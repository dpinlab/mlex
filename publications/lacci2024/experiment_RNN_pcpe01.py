import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__ , "..", "..", "..")))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from copy import deepcopy
from mlex.features.sequences import SequenceTransformer
from mlex import PastFutureSplit
from mlex import PreProcessingTransformer
from mlex import StandardEvaluator
from mlex import QuantileThresholdStrategy
from mlex.models.models import RNNModel


threshold_strategy = 'quantile'
threshold_selection = QuantileThresholdStrategy()
sequence_lengths = [20, 50]
batch_size = 32
column_to_stratify = 'CPF_CNPJ_TITULAR'
hidden_size = 10
num_layers = 1
num_classes = 1
epochs = 30
patience = 3
target_column = 'I-d'
# filter_data = {'NATUREZA_LANCAMENTO': 'C'}
sequences_compositions = ['temporal']
sequence_column_dict = {'temporal': None, 'account': 'CONTA_TITULAR', 'individual': 'CPF_CNPJ_TITULAR'}
iterations = 1
path = r'/data/pcpe/pcpe_01.csv'

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# reader = DataReader(path, target_columns=[target_column])
# X = reader.fit_transform(X=None)
# y = reader.get_target()

df = pd.read_csv(path, delimiter=';', decimal=',')
df['I-d'].fillna(0, inplace=True)
df['I-d'] = df['I-d'].astype(int)
X = df.drop(columns=[target_column], axis=1)
y = df[[target_column]]

final_models = []

for sequence_composition in sequences_compositions:
    print(f"experiment sequence {sequence_composition}")
    sequence_column = sequence_column_dict[sequence_composition]
    for sequence_length in sequence_lengths:
        print(f"sequence length {sequence_length}")
        for i in range(iterations):
            print(f"iteration {i+1}")
            # Initialize splitter
            splitter_tt = PastFutureSplit()
            splitter_tt.fit(X, y)
            # Get splits
            X_train, y_train, X_test, y_test = splitter_tt.transform(X, y)

            splitter_tv = PastFutureSplit(proportion=0.7)
            splitter_tv.fit(X_train, y_train)
            # Get splits
            X_train, y_train, X_val, y_val = splitter_tv.transform(X_train, y_train)

            group_train, group_val, group_test = (None, None, None)
            if sequence_composition != 'temporal':
                group_train = X_train.loc[:,sequence_column].values
                group_val = X_val.loc[:,sequence_column].values
                group_test = X_test.loc[:, sequence_column].values

            preprocessor = PreProcessingTransformer(target_column=[target_column])

            X_train_array = preprocessor.transform(X_train, y_train)
            y_train_array = preprocessor.get_target()
            features_names_train = preprocessor.get_feature_names_out()

            X_val_array = preprocessor.transform(X_val, y_val)
            y_val_array = preprocessor.get_target()
            features_names_val = preprocessor.get_feature_names_out()

            X_test_array = preprocessor.transform(X_test, y_test)
            y_test_array = preprocessor.get_target()
            features_names_test = preprocessor.get_feature_names_out()

            features_names_common = np.intersect1d(np.intersect1d(features_names_train, features_names_val),
                                                   features_names_test)
            mask_train = np.isin(features_names_train, features_names_common)
            mask_val = np.isin(features_names_val, features_names_common)
            mask_test = np.isin(features_names_test, features_names_common)

            X_train_array = X_train_array[:, mask_train]
            X_val_array = X_val_array[:, mask_val]
            X_test_array = X_test_array[:, mask_test]

            sequence_transformer = SequenceTransformer(
                sequence_length=sequence_length,
                batch_size=batch_size,
                shuffled=False
            )

            train_loader = sequence_transformer.transform(X_train_array, y_train_array, column_to_stratify=group_train)
            val_loader = sequence_transformer.transform(X_val_array, y_val_array, column_to_stratify=group_val)
            test_loader = sequence_transformer.transform(X_test_array, y_test_array, column_to_stratify=group_test)

            input_size = X_train_array.shape[1]
            model = RNNModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes)
            model.to(device=device)
            optimizer = torch.optim.RMSprop(params=model.parameters(), lr=.001, alpha=.9, eps=1e-07)

            loss_fn = nn.BCELoss()
            best_val_loss = float('inf')
            patience_counter = 0
            best_weights = None
            history = {'train': [], 'val': [], 'epoch': []}

            for epoch in range(epochs):
                model.train()
                train_loss = 0.0
                total_samples = 0
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    current_batch_size = batch_x.size(0)

                    optimizer.zero_grad()
                    output = model.forward(batch_x)
                    loss = loss_fn(output, batch_y)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * current_batch_size
                    total_samples += current_batch_size

                # Validation phase
                val_loss = 0.0
                total_samples_val = 0
                model.eval()
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(device)
                        batch_y = batch_y.to(device)
                        current_batch_size = batch_x.size(0)
                        outputs = model.forward(batch_x)
                        val_loss += loss_fn(outputs, batch_y).item() * current_batch_size
                        total_samples_val += current_batch_size

                # Record history
                avg_train_loss = train_loss / total_samples
                avg_val_loss = val_loss / total_samples_val
                history['train'].append(avg_train_loss)
                history['val'].append(avg_val_loss)
                history['epoch'].append(epoch+1)

                print(f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {avg_train_loss:.4f} - "
                    f"Val Loss: {avg_val_loss:.4f}")

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_weights = deepcopy(model.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}\n\n")
                        break

            model.load_state_dict(best_weights)

            # Prediction and evaluation
            list_ys_true_pred = []
            model.eval()
            y_pred_score = []
            y_true = []
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x = x_batch.to(device)
                    outputs = model(x).cpu().numpy()
                    y_pred_score.extend(outputs.flatten())
                    y_true.extend(np.array(y_batch, dtype="int4").flatten())

            assert len(y_pred_score) == len(y_true)

            evaluator = StandardEvaluator(f"RNN_Layers-{num_layers}_HiddenSize-{hidden_size}_SequenceLength-{sequence_length}_{sequence_composition}_{threshold_strategy}_Iteration-{i+1}",
                                        threshold_selection)
            evaluator.evaluate(np.array(y_true), [], y_pred_score)
            print(evaluator.summary())
            print('\n')

            evaluator.save('evaluation.parquet')
            evaluator.save('evaluation.json')

            final_models.append(('RNN',
                                 sequence_composition,
                                 i+1,
                                 model.state_dict(),
                                 history,
                                 input_size,
                                 threshold_strategy,
                                 num_layers,
                                 hidden_size,
                                 sequence_length,
                                 batch_size,
                                 epochs,
                                 patience,
                                 optimizer.state_dict(),
                                 features_names_train))


df = pd.DataFrame(final_models, columns=['MODEL_NAME', 'SEQUENCE_COMPOSITION', 'ITERATION', 'MODEL_STATE_DICT', 'HISTORY', 'INPUT_SIZE', 'THRESHOLD_STRATEGY', 'NUM_LAYERS', 'HIDDEN_SIZE', 'SEQUENCE_LENGTH', 'BATCH_SIZE', 'EPOCHS', 'PATIENCE', 'OPTIMIZER_STATE_DICT', 'FEATURES_NAMES'])
df.to_pickle(f'df_results_RNN_NumLayer-{num_layers}_HiddenSize-{hidden_size}.pkl')
