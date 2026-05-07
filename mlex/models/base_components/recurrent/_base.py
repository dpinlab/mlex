import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mlex.features.length_strategy import LengthStrategy
from mlex.features.sequences import (
    DynamicLengthBatchSampler,
    DynamicSequenceDataset,
    SequenceDataset,
)
from mlex.models.base_components.recurrent.params import RecurrentModelParams


class _RecurrentBaseModel(nn.Module):
    LAYER_CLS: type = None
    BIDIRECTIONAL: bool = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.LAYER_CLS is None:
            raise TypeError(f"{cls.__name__} must define LAYER_CLS")

    def __init__(self, validation_data=None, **kwargs):
        super().__init__()
        params = RecurrentModelParams(validation_data=validation_data, **kwargs)
        self.params = params

        for field_name in RecurrentModelParams.model_fields:
            setattr(self, field_name, getattr(params, field_name))

        self.device = params.device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        layer_kwargs = dict(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        if self.BIDIRECTIONAL:
            layer_kwargs['bidirectional'] = True

        self.recurrent_layer = self.LAYER_CLS(**layer_kwargs)
        linear_in = self.hidden_size * (2 if self.BIDIRECTIONAL else 1)
        self.linear = nn.Linear(linear_in, self.output_size)
        self.sigmoid = nn.Sigmoid()

        self.to(device=self.device)

        self.fitted_ = False
        self.predict_end_indices = []
        self.activations = {'train': [], 'validation': [], 'predict': []}
        self.epoch_observers = list(params.epoch_observers) if params.epoch_observers else []
        self.history = {'train': [], 'val': [], 'epoch': []}

    def _activation_mode(self):
        mode = 'train' if not self.fitted_ and self.training else 'validation'
        mode = 'predict' if self.fitted_ and not self.training else mode
        return mode

    def _forward_logits(self, x):
        layer_out, _hidden = self.recurrent_layer(x)
        last_output = layer_out[:, -1, :]
        linear_out = self.linear(last_output)

        if self.collect_activations:
            self.activations[self._activation_mode()].append({
                'hidden_states': layer_out.detach().cpu().numpy(),
                'last_hidden': last_output.detach().cpu().numpy(),
                'linear_out': linear_out.detach().cpu().numpy(),
            })

        return linear_out

    def __forward(self, x):
        linear_out = self._forward_logits(x)
        output = self.sigmoid(linear_out)

        if self.collect_activations:
            self.activations[self._activation_mode()][-1]['output'] = (
                output.detach().cpu().numpy()
            )

        return output

    @property
    def name(self):
        return type(self).__name__

    def fit(self, X, y):
        if self.random_seed is not None:
            torch.cuda.manual_seed(self.random_seed)
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)

        self.__fit_core(X, y)

    def predict_proba(self, X):
        self.eval()
        with torch.no_grad():
            probs = self.__forward(X).cpu().numpy()
        return probs

    def predict(self, X):
        self.activations['predict'] = []
        if self.dynamic_length_strategy is None:
            return self._predict_fixed(X)
        return self._predict_ensemble(X)

    def _predict_fixed(self, X):
        test_loader = self._create_dataloader(X, None, shuffle_dataloader=False)
        self.predict_end_indices = test_loader.dataset.valid_end_indices
        y_pred = []
        for x_batch in test_loader:
            x = x_batch.to(self.device)
            outputs = self.predict_proba(x)
            y_pred.extend(outputs.flatten())
        return y_pred

    def _predict_ensemble(self, X):
        lengths = sorted({int(L) for L in self.dynamic_length_strategy.lengths})
        per_length = {}

        self.eval()
        for L in lengths:
            loader = self._create_dataloader(X, None, shuffle_dataloader=False, seq_length=L)
            end_indices = list(loader.dataset.valid_end_indices)
            per_idx = {}
            cursor = 0
            with torch.no_grad():
                for x_batch in loader:
                    x = x_batch.to(self.device)
                    logits = self._forward_logits(x).cpu()
                    for j in range(logits.shape[0]):
                        per_idx[end_indices[cursor + j]] = logits[j]
                    cursor += logits.shape[0]
            per_length[L] = per_idx

        covered = sorted(set.union(*(set(d) for d in per_length.values())))
        self.predict_end_indices = covered
        y_pred = []
        with torch.no_grad():
            for end_idx in covered:
                contributing = [
                    per_length[L][end_idx]
                    for L in lengths
                    if end_idx in per_length[L]
                ]
                avg_logit = torch.stack(contributing).mean(dim=0)
                prob = self.sigmoid(avg_logit).numpy()
                y_pred.extend(prob.flatten())
        return y_pred

    def __fit_core(self, X, y):
        train_loader = self._create_fit_dataloader(
            X, y, shuffle=self.shuffle_dataloader, drop_last=self.dynamic_drop_last
        )
        val_loader = self._create_fit_dataloader(
            self.validation_data[0], self.validation_data[1],
            shuffle=False, drop_last=False,
        )
        return self.__train_epochs(train_loader, val_loader)

    def __train_epochs(self, train_loader, val_loader):
        optimizer = torch.optim.RMSprop(
            self.parameters(),
            lr=self.learning_rate,
            alpha=self.alpha,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )
        criterion = nn.BCELoss()
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        history = {'train': [], 'val': [], 'epoch': []}

        for epoch in range(self.epochs):
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

            val_loss = 0
            total_samples_val = 0
            val_outputs_all = []
            val_targets_all = []
            self.eval()
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    current_batch_size = batch_x.size(0)
                    outputs = self.__forward(batch_x)
                    val_loss += criterion(outputs, batch_y).item() * current_batch_size
                    total_samples_val += current_batch_size
                    val_outputs_all.append(outputs.detach().cpu().numpy().flatten())
                    val_targets_all.append(batch_y.detach().cpu().numpy().flatten())

            avg_train_loss = train_loss / total_samples
            avg_val_loss = val_loss / total_samples_val
            history['train'].append(avg_train_loss)
            history['val'].append(avg_val_loss)
            history['epoch'].append(epoch + 1)

            print(f"Epoch {epoch + 1}/{self.epochs} - "
                  f"Train Loss: {avg_train_loss:.4f} - "
                  f"Val Loss: {avg_val_loss:.4f}")

            if self.epoch_observers:
                val_outputs_arr = np.concatenate(val_outputs_all) if val_outputs_all else np.array([])
                val_targets_arr = np.concatenate(val_targets_all) if val_targets_all else np.array([])
                observer_context = {
                    'epoch': epoch + 1,
                    'model': self,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'val_outputs': val_outputs_arr,
                    'val_targets': val_targets_arr,
                }
                for observer in self.epoch_observers:
                    observer.on_epoch_end(observer_context)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_weights = deepcopy(self.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}\n\n")
                    break

        if best_weights is not None:
            self.load_state_dict(best_weights)
        self.history = history
        return best_weights, history

    def __create_dataset(self, X, y, seq_length=None):
        L = seq_length if seq_length is not None else self.seq_length
        return SequenceDataset(X, y, L, self.group_index)

    def _loader_kwargs(self):
        kwargs = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if self.num_workers > 0 and self.persistent_workers:
            kwargs["persistent_workers"] = True
        return kwargs

    def _create_dataloader(self, X, y, shuffle_dataloader, seq_length=None):
        if y is not None:
            y = y.values if hasattr(y, 'values') else y
        return DataLoader(
            self.__create_dataset(X, y, seq_length),
            batch_size=self.batch_size,
            shuffle=shuffle_dataloader,
            **self._loader_kwargs(),
        )

    def _create_fit_dataloader(self, X, y, shuffle, drop_last):
        if self.dynamic_length_strategy is None:
            return self._create_dataloader(X, y, shuffle)

        if y is not None:
            y = y.values if hasattr(y, 'values') else y

        strategy: LengthStrategy = self.dynamic_length_strategy
        dataset = DynamicSequenceDataset(
            X=X,
            y=y,
            sequence_lengths=strategy.lengths,
            group_column_index=self.group_index,
        )
        batch_sampler = DynamicLengthBatchSampler(
            dataset=dataset,
            batch_size=self.batch_size,
            strategy=strategy,
            shuffle=shuffle,
            drop_last=drop_last,
            random_seed=self.random_seed,
        )
        return DataLoader(dataset, batch_sampler=batch_sampler, **self._loader_kwargs())
