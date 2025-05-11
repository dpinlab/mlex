import sys
sys.path.append("../..")
sys.path.append("../../..")

import torch
from mlex.features.sequences import SequenceTransformer
from mlex import FeatureStratifiedSplit
from mlex import PreProcessingTransformer
from mlex import DataReader
from mlex.models.composite import MLEXComposite, MLEXLeafComponent
from mlex.models.models import RNNModule, GRUModule, LSTMModule


sequence_length = 10
batch_size = 32
column_to_stratify = 'CONTA_TITULAR'
hidden_size = 10
num_layers = 1
num_classes = 1
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

rnn = RNNModule(input_size=X_train.shape[1], hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes)
lstm = LSTMModule(input_size=X_train.shape[1], hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes)
gru = GRUModule(input_size=X_train.shape[1], hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes)
leaf1 = MLEXLeafComponent(torch.optim.RMSprop, model=rnn)
leaf2 = MLEXLeafComponent(torch.optim.RMSprop, model=lstm)
leaf3 = MLEXLeafComponent(torch.optim.RMSprop, model=gru)
composite = MLEXComposite()
composite.add_module(name='leaf1',module=leaf1)
composite.add_module(name='leaf2',module=leaf2)
composite.add_module(name='leaf3',module=leaf3)
composite.to(device=device)

for X_batch, y_batch in train_loader:
    composite.optimize_module(x=X_batch, target=y_batch)

