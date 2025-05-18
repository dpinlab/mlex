import sys
sys.path.append("../..")
sys.path.append("../../..")

import torch
import torch.nn.functional as F
import torch.nn as nn
from mlex.features.sequences import SequenceTransformer
from mlex import FeatureStratifiedSplit
from mlex import PreProcessingTransformer
from mlex import DataReader
from mlex.models.models import RNNModel, GRUModel, LSTMModel


sequence_length = 10
batch_size = 32
column_to_stratify = 'CONTA_TITULAR'
hidden_size = 10
num_layers = 1
num_classes = 1
epochs = 100
target_column = 'I-d'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = r'/data/pcpe/pcpe_03.csv'
reader = DataReader(path, target_columns=[target_column])

X = reader.fit_transform(X=None)
y = reader.get_target()

# Initialize splitter
splitter = FeatureStratifiedSplit(column_to_stratify=column_to_stratify, test_proportion=0.3)
splitter.fit(X, y)

# Get splits
X_train, y_train, X_test, y_test = splitter.transform(X, y)
group_train, group_test = splitter.get_groups(X)

preprocessor = PreProcessingTransformer(target_columns=[target_column])

X_train_array = preprocessor.transform(X_train, y_train)
y_train_array = preprocessor.get_target()
features_names = preprocessor.get_feature_names_out()

X_test_array = preprocessor.transform(X_test, y_test)
y_test_array = preprocessor.get_target()


sequence_transformer = SequenceTransformer(
    column_to_stratify= group_train,
    sequence_length=sequence_length,
    batch_size=batch_size,
    shuffled=True
)

train_loader = sequence_transformer.transform(X_train_array, y_train_array)
test_loader = sequence_transformer.transform(X_test_array, y_test_array)

rnn = RNNModel(input_size=X_train_array.shape[1], hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes)
rnn.to(device=device)
rnn_optimizer = torch.optim.RMSprop(params=rnn.parameters(), lr=.001, alpha=.9, eps=1e-07)

lstm = LSTMModel(input_size=X_train_array.shape[1], hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes)
lstm.to(device=device)
lstm_optimizer = torch.optim.RMSprop(params=lstm.parameters(), lr=.001, alpha=.9, eps=1e-07)

gru = GRUModel(input_size=X_train_array.shape[1], hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes)
gru.to(device=device)
gru_optimizer =  torch.optim.RMSprop(params=gru.parameters(), lr=.001, alpha=.9, eps=1e-07)


loss_fn = nn.BCELoss()

for _ in range(epochs):
    for X_batch, y_batch in train_loader:
        rnn_optimizer.zero_grad()
        output = rnn.forward(X_batch)
        loss = loss_fn(output, y_batch)
        print(loss)
        loss.backward()
        rnn_optimizer.step()

