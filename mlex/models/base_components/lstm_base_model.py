import torch.nn as nn
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from mlex.features.sequences import SequenceDataset
from copy import deepcopy


class LSTMBaseModel(nn.Module):
    def __init__(
        self,
        validation_data,
        input_size=None,
        hidden_size=10,
        num_layers=1,
        output_size=1,
        seq_length=30,
        batch_size=32,
        shuffle_dataloader=True,
        learning_rate=1e-03,
        alpha=.9,
        eps=1e-07,
        weight_decay=.0,
        epochs=30,
        patience=5,
        group_index=-1,
        random_seed=42,
        device=None,
        **kwargs
    ):
        super().__init__()
        # Model architecture parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # Data parameters
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.shuffle_dataloader = shuffle_dataloader

        # Training parameters
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.validation_data = validation_data
        self.group_index = group_index
        self.random_seed = random_seed

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.linear = nn.Linear(hidden_size, self.output_size)
        self.sigmoid = nn.Sigmoid()

        self.to(device=self.device)

    def __forward(self, x):
        # LSTM forward pass
        # lstm_out: (batch_size, seq_length, hidden_size)
        # hidden: (num_layers, batch_size, hidden_size)
        lstm_out, hidden = self.lstm(x)

        # Take the output from the last time step
        # lstm_out[:, -1, :] has shape (batch_size, hidden_size)
        last_output = lstm_out[:, -1, :]

        # Pass through linear layer
        # linear_out: (batch_size, output_size)
        linear_out = self.linear(last_output)

        # Apply sigmoid activation
        # output: (batch_size, output_size)
        output = self.sigmoid(linear_out)

        return output

    @property
    def name(self):
        return "LSTMBaseModel"

    def fit(self, X, y):
        torch.cuda.manual_seed(self.random_seed)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        self.__fit_core(X, y)
        self.fitted_ = True

    def predict_proba(self, X):
        self.eval()
        with torch.no_grad():
            probs = self.__forward(X).cpu().numpy()
        return probs

    def predict(self, X):
        return self.score_samples(X)

    def score_samples(self, X):
        test_loader = self._create_dataloader(X, None, shuffle_dataloader=False)
        y_pred = []
        for x_batch in test_loader:
            x = x_batch.to(self.device)
            outputs = self.predict_proba(x)
            y_pred.extend(outputs.flatten())
        return y_pred

    def __fit_core(self, X, y):
        train_loader = self._create_dataloader(X, y, self.shuffle_dataloader)
        val_loader = self._create_dataloader(self.validation_data[0], self.validation_data[1], self.shuffle_dataloader)

        return self.__train_epochs(train_loader, val_loader)

    def __train_epochs(self, train_loader, val_loader):
        optimizer = torch.optim.RMSprop(
            self.parameters(), 
            lr=self.learning_rate,
            alpha=self.alpha,
            eps=self.eps,
            weight_decay=self.weight_decay
        )
        criterion = nn.BCELoss()
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train': [], 'val': [], 'epoch': []}

        for epoch in range(self.epochs):
            # Training phase
            self.train()
            train_loss = 0
            total_samples = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                current_batch_size = batch_x.size(0)

                optimizer.zero_grad()
                outputs = self.__forward(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * current_batch_size
                total_samples += current_batch_size

            # Validation phase
            val_loss = 0
            total_samples_val = 0
            self.eval()
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    current_batch_size = batch_x.size(0)
                    outputs = self.__forward(batch_x)
                    val_loss += criterion(outputs, batch_y).item() * current_batch_size
                    total_samples_val += current_batch_size

            # Record history
            avg_train_loss = train_loss / total_samples
            avg_val_loss = val_loss / total_samples_val
            history['train'].append(avg_train_loss)
            history['val'].append(avg_val_loss)
            history['epoch'].append(epoch+1)

            print(f"Epoch {epoch + 1}/{self.epochs} - "
                f"Train Loss: {avg_train_loss:.4f} - "
                f"Val Loss: {avg_val_loss:.4f}")

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_weights = deepcopy(self.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}\n\n")
                    break

        # Load best weights
        self.load_state_dict(best_weights)
        return best_weights, history


    def __create_dataset(self, X, y):
        return SequenceDataset(X, y, self.seq_length, self.group_index)


    def _create_dataloader(self, X, y, shuffle_dataloader):
        if y is not None:
            y = y.values if hasattr(y, 'values') else y
        return DataLoader(self.__create_dataset(X, y), batch_size=self.batch_size, shuffle=shuffle_dataloader)
