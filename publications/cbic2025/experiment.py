import sys
sys.path.append("../..")
sys.path.append("../../..")

import torch
from mlex.features.sequences import SequenceTransformer
from mlex.utils.split import FeatureStratifiedSplit
from mlex.models.composite import MLEXComposite, MLEXLeafComponent
from mlex.models.models import RNNModule, GRUModule, LSTMModule


sequence_length = 10
batch_size = 32
column_to_stratify = 'CONTA_TITULAR'
hidden_size = 10
num_layers = 1
num_classes = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = "/data/pcpe/pcpe_03.csv"

splitter = FeatureStratifiedSplit(path)
X_train, X_test, y_train, y_test = splitter.train_test_split(column_to_stratify = column_to_stratify)

sequence_transformer = SequenceTransformer(
    column_to_stratify= splitter.group_train,
    sequence_length=sequence_length,
    batch_size=batch_size,
    shuffled=True
)

train_loader = sequence_transformer.transform(X_train, y_train)

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

