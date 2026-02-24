import abc
from mlex.utils.network_strategy import NetworkPathStrategy


class BaseTestNetworkStrategy(abc.ABC):
    @abc.abstractmethod
    def get_network_strategy(self) -> NetworkPathStrategy:
        pass

    @abc.abstractmethod
    def test_create_digraph_backwards(self):
        pass

    @abc.abstractmethod
    def test_create_digraph_forwards(self):
        pass

    @abc.abstractmethod
    def test_find_path(self):
        pass

    @abc.abstractmethod
    def test_get_transactions(self):
        pass
