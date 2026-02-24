import sys

sys.path.append("..")

from mlex.utils.sorter import AccountBalanceSorter
from mlex.utils.network_strategy import EulerianPathStrategy
from mlex.utils.datareader import (
    DataReader,
    sorter_decorator,
    pcpe_preprocessing_read_func as pre_fn,
)
import pdb

pdb.set_trace()
network_strategy = EulerianPathStrategy()
sorter = AccountBalanceSorter(network_strategy=network_strategy)
reader = DataReader(
    data_path="data/PCPE/pcpe_04.csv",
    target_columns=["I-d"],
    preprocessing_func=sorter_decorator(pre_fn),
)
df = reader.read_df()
X = df.reset_index()[
    ["index", "CONTA_TITULAR", "TIPO", "TIMESTEP", "VALOR_SALDO", "SALDO_ANTERIOR"]
].to_numpy()
sorted_transactions, inconsistent_transactions = sorter.sort(X=X)
