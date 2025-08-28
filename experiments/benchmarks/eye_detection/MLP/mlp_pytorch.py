import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import shap
from sklearn import metrics
from itertools import cycle
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve
from mlex import PreProcessingTransformer
from mlex import PreProcessingTransformer

#from mlex.publications.lacci2025.analysis import LacciAnalysis
from mlex import DataReader
#from mlex import CpfStratifiedSplit
from mlex.evaluation.evaluator import StandardEvaluator
from mlex.evaluation.threshold import F1MaxThresholdStrategy
from mlex import PastFutureSplit
target_column = 'eyeDetection'
filter_data = {}
threshold_selection = F1MaxThresholdStrategy()
threshold_str = "f1max"

# Define PyTorch models
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=[10], dropout_rate=0.2):
        super().__init__()
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = size
        layers.append(nn.Linear(prev_size, 1))  # Output layer with 1 neuron
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze(-1)  # Shape: [batch_size, 1]


def get_y_pred_actual(y_pred_score, y_test):
    y_pred = y_pred_score > np.quantile(y_pred_score, 0.95)
    return y_test, y_pred


def get_y_pred_actual_f1(y_pred_score, y_test):
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_score)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
    best_threshold = thresholds[np.argmax(f1_scores)]
    y_pred = (y_pred_score > best_threshold).astype(int)
    return y_test, y_pred, best_threshold


def roc_samples(y_t, y_s):
    N = len(y_t)
    rocs = []
    for b in range(30):
        choices = np.random.choice(N, size=N)
        ys_t = y_t[choices]
        ys_p = y_s[choices]
        fpr, tpr, _ = metrics.roc_curve(ys_t, ys_p)
        auc = metrics.auc(fpr, tpr)
        rocs.append(auc)
    return rocs


def kfold_by_account(X_train, X_test, accounts_train, y_train, y_test, fold_results):
    # Convert labels to 1D tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).squeeze()  # Remove extra dimensions
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).squeeze()     # Remove extra dimensions

    # Create test DataLoader
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    data_test = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

    # KFold split by accounts
    unique_accounts = np.unique(accounts_train)
    unique_accounts = np.unique(accounts_train)
    np.random.shuffle(unique_accounts)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(unique_accounts)):
        train_accounts = unique_accounts[train_idx]
        val_accounts = unique_accounts[val_idx]

        # Create masks
        train_mask = np.isin(accounts_train, train_accounts)
        val_mask = np.isin(accounts_train, val_accounts)

        # Slice data
        X_train_fold = X_train_tensor[train_mask]
        y_train_fold = y_train_tensor[train_mask]
        X_val_fold = X_train_tensor[val_mask]
        y_val_fold = y_train_tensor[val_mask]

        # Create DataLoaders
        train_dataset = torch.utils.data.TensorDataset(X_train_fold, y_train_fold)
        val_dataset = torch.utils.data.TensorDataset(X_val_fold, y_val_fold)
        data_train = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        data_val = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

        fold_results[fold_idx] = (data_train, data_val)
        # break

    return data_test, fold_results


def train_model(model, data_train, data_val, device, patience=3, epochs=30):
    optimizer = optim.RMSprop(model.parameters())
    criterion = nn.BCELoss()
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0
    history = {'loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        total = 0
        for x_batch, y_batch in data_train:
            x = x_batch.to(device)
            y = y_batch.to(device)

            # Debugging: Check shapes
            # print(f"Input shape: {x.shape}, Target shape: {y.shape}")

            optimizer.zero_grad()
            outputs = model(x)
            # print(f"Output shape: {outputs.shape}, Target shape: {y.shape}")
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            total += x.size(0)

        train_loss /= total
        history['loss'].append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        total_val = 0
        with torch.no_grad():
            for x_val, y_val in data_val:
                x = x_val.to(device)
                y = y_val.to(device)

                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item() * x.size(0)
                total_val += x.size(0)

        val_loss /= total_val
        history['val_loss'].append(val_loss)

        print(f'Epoch {epoch + 1}/{epochs} - '
              f'Train Loss: {train_loss:.4f} - '
              f'Val Loss: {val_loss:.4f}')

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

    model.load_state_dict(best_model)
    return model, history


# Main execution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path_file = r"/data/isa/EEG_Eye_State_com_timestamp.arff"
#df = pd.read_csv(path_file, delimiter=';', decimal=',', low_memory=False)

reader = DataReader(path_file, target_columns=[target_column], filter_dict=filter_data)
X = reader.fit_transform(X=None)
y = reader.get_target().astype(int)

X = X.drop(columns=["Timestamp"])

#####

X = X.drop([10386,11509,898]) 
y = y.drop([10386,11509,898]) 

# df = df.sort_values(by='DATA_LANCAMENTO', ascending=True).reset_index(drop=True)

# df = df.sort_values(by=['CONTA_TITULAR', 'DATA_LANCAMENTO'], ascending=True).reset_index(drop=True)

# df = df.sort_values(by=['COMU_ID', 'DATA_LANCAMENTO'], ascending=True).reset_index(drop=True)

# lacci_analysis = LacciAnalysis(df)
# df_descriptive = lacci_analysis.get_results_descriptive()
# print(df_descriptive.to_latex())



# Split data
splitter_tt = PastFutureSplit(proportion=0.75)
splitter_tt.fit(X, y)
                # Get splits
X_train, y_train, X_test, y_test = splitter_tt.transform(X, y)

splitter_tv = PastFutureSplit(proportion=0.66)
splitter_tv.fit(X_train, y_train)
                # Get splits
X_train, y_train, X_val, y_val = splitter_tv.transform(X_train, y_train)

############
#preprocessing

X_train['GROUP'] = 'Unknown'
y_train['GROUP'] = 'Unknown'

X_val['GROUP'] = 'Unknown'
y_val['GROUP'] = 'Unknown'

X_test['GROUP'] = 'Unknown'
y_test['GROUP'] = 'Unknown'


preprocessor = PreProcessingTransformer(target_columns=[target_column], numeric_features=X.columns,categorical_features=[], handle_unknown='ignore')
preprocessor.fit(X_train, y_train)

X_train = preprocessor.transform(X_train, y_train)
#y_train = preprocessor.get_target()

X_val = preprocessor.transform(X_val, y_val)
#y_val = preprocessor.get_target()

X_test = preprocessor.transform(X_test, y_test)
#y_test = preprocessor.get_target()


#################

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).squeeze()

X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).squeeze()

X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).squeeze()

# DataLoaders
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)

data_train = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
data_val = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
data_test = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# KFold and DataLoaders


# fold_results = {}
# data_test, fold_results = kfold_by_account(X_train, X_test, accounts_train, y_train, y_test, fold_results)

# Train MLP
model_names = ["MLP"]
input_size = X_train.shape[1]

# final_models = []

# for model_name in model_names:
#     print(f"\nTraining {model_name}")
#     for fold_idx, (data_train, data_val) in fold_results.items():
#         print(f'\nFold {fold_idx + 1}/{len(fold_results)}')
#         model = MLPModel(input_size).to(device)
#         trained_model, history = train_model(model, data_train, data_val, device)
#         final_models.append((model_name, fold_idx, trained_model, history))

model = MLPModel(input_size).to(device)
trained_model, history = train_model(model, data_train, data_val, device)
# Select best model per fold
# best_models = {}
# for model_name in model_names:
#     best_model = min(final_models, key=lambda x: min(x[3]['val_loss']))[2]
#     best_models[model_name] = best_model

# Evaluate on test set
list_ys_true_pred = []
#for model_name, model in best_models.items():
model.eval()
y_pred_score = []
y_true = []
with torch.no_grad():
    for x_batch, y_batch in data_test:
        outputs = model(x_batch.to(device)).cpu().numpy()
        y_pred_score.extend(outputs.flatten())
        y_true.extend(np.array(y_batch, dtype="int32").flatten())

assert len(y_pred_score) == len(y_true)

evaluator = StandardEvaluator(f"MLP_f1max_{threshold_str}",
                                threshold_selection)
evaluator.evaluate(np.array(y_true), [], y_pred_score)  # Evaluate
print(evaluator.summary())

evaluator.save("evaluation.parquet")  # Saving results
evaluator.save("evaluation.json")  # Saving results

# # Plotting and metrics
# names_ = ['MLP']
# name_cycler = cycle(names_)

# plotter = Plotter()
# for y_true, y_pred, _ in list_ys_true_pred:
#     plotter.plot_matrix(y_true, y_pred, name_cycler, filename=f"confusion_MLP")

# # ROC plotting
# title = "ROC"
# fig, ax = plt.subplots()
# ax.plot([0, 1], [0, 1], "k--", linewidth=4, label='random classifier')
# for y_true, y_pred, y_pred_score in list_ys_true_pred:
#     fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred_score)
#     auc = metrics.auc(fpr, tpr)
#     ax.plot(fpr, tpr, linewidth=4, label=f"{next(name_cycler)} (AUC = {round(auc, 2)})")
# ax.set_xlim([0.0, 1.0])
# ax.set_ylim([0.0, 1.05])
# ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=16)
# ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=16)
# ax.set_title(f"Receiver Operating Characteristic \n {title}", fontsize=18)
# ax.legend(loc="lower right")
# plt.savefig(f"roc_MLP.pdf")
# plt.show()
# plt.clf()
# plt.close()

# SHAP analysis
# background_samples = []
# for x_batch, _ in fold_results[0][0]:  # First fold's training data
#     background_samples.append(x_batch.numpy())  # Convert to NumPy
# background = np.concatenate(background_samples)[:100]  # Use NumPy array

# test_samples = []
# for x_batch, _ in data_test:
#     test_samples.append(x_batch.numpy())  # Convert to NumPy
# test_samples = np.concatenate(test_samples)[:50]


background = X_train.values[:100]
test_samples = X_test.values[:50]

def model_predict(x):
    model.eval()
    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)  # Convert back to tensor
    with torch.no_grad():
        outputs = model(x_tensor).cpu().numpy()  # Return NumPy array
    return outputs


#for model_name, model in best_models.items():
explainer = shap.KernelExplainer(
    model_predict,  # Prediction function (not the model object)
    background  # NumPy array, shape [n_samples, input_size]
)

# Calculate SHAP values
shap_values = explainer.shap_values(test_samples)
feature_names = X_train.columns.tolist()

shap.summary_plot(shap_values, test_samples, feature_names=feature_names, plot_type="bar")
plt.savefig("shap_MLP.pdf")
