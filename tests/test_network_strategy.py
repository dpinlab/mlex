import unittest

import networkx as nx
import numpy as np
from .data.sort_fixtures import NetworkStrategyFixture

from mlex.utils.network_strategy import EulerianPathStrategy, HamiltonianPathStrategy
from tests.test_base import BaseTestNetworkStrategy


class TestEulerianPathStrategy(BaseTestNetworkStrategy, unittest.TestCase):
    def setUp(self):
        self.fixture = NetworkStrategyFixture()
        self.network_strategy = self.get_network_strategy()

    def get_network_strategy(self):
        return EulerianPathStrategy()

    def test_create_digraph_backwards(self):
        root_node, transactions = self.fixture.get_data_create_digraph_backwards()
        values = np.array(transactions)
        actual_multi_digraph = self.network_strategy.create_digraph(
            root_node=root_node, values=values
        )
        expected_nodes = [15000, 10000, 5000]
        expected_edges = [(15000, 10000, 0), (10000, 5000, 0)]
        assert list(actual_multi_digraph.nodes) == expected_nodes
        assert list(actual_multi_digraph.edges) == expected_edges
        self.assertIsInstance(actual_multi_digraph, nx.MultiDiGraph)

    def test_create_digraph_forwards(self):
        root_node, transactions = self.fixture.get_data_create_digraph_forwards()
        values = np.array(transactions)
        actual_multi_digraph = self.network_strategy.create_digraph(
            root_node=root_node, values=values, backwards=False
        )
        expected_nodes = [5000, 15000, 10000]
        expected_edges = [(5000, 15000, 0), (15000, 10000, 0)]
        assert list(actual_multi_digraph.nodes) == expected_nodes
        assert list(actual_multi_digraph.edges) == expected_edges
        self.assertIsInstance(actual_multi_digraph, nx.MultiDiGraph)

    def test_find_path(self):
        source = 0
        step = 10000
        edges = [(i, i + step, 0) for i in range(0, 50000, step)]
        D = nx.MultiDiGraph()
        D.add_edges_from(edges)
        expected_path = [(i, i + step, 0) for i in range(0, 50000, step)]
        actual_path = self.network_strategy.find_path(G=D, source=source)
        assert actual_path == expected_path

    def test_get_transactions(self):
        step = 10000
        expected_transactions = self.fixture.get_transactions_data()
        path = [(i, i + step, 0) for i in range(0, 50000, step)]
        D = nx.MultiDiGraph()
        D.add_edges_from(
            [(u, v, k, {"transaction":att}) for (u, v, k), att in zip(path, expected_transactions)]
        )
        actual_transactions = self.network_strategy.get_transactions(G=D, path=path)
        assert actual_transactions == expected_transactions

    def test_get_source(self):
        source_node = self.fixture.get_source_data()
        actual_source = self.network_strategy.get_source(node=source_node)
        expected_source = 10000
        assert expected_source == actual_source

    def test_get_source_backwards(self):
        source_node = self.fixture.get_source_data()
        actual_source = self.network_strategy.get_source(
            node=source_node, backwards=True
        )
        expected_source = 15000
        assert expected_source == actual_source


class TestHamiltonianPathStrategy(BaseTestNetworkStrategy, unittest.TestCase):
    def setUp(self):
        self.fixture = NetworkStrategyFixture()
        self.network_strategy = self.get_network_strategy()

    def get_network_strategy(self):
        return HamiltonianPathStrategy()

    def test_create_digraph_backwards(self):
        root_node, transactions = self.fixture.get_data_create_digraph_backwards()
        values = np.array(transactions)
        actual_digraph = self.network_strategy.create_digraph(
            root_node=root_node, values=values
        )
        expected_nodes = [0, 1]
        expected_edges = [(0, 1)]
        assert list(actual_digraph.nodes) == expected_nodes
        assert list(actual_digraph.edges) == expected_edges
        self.assertIsInstance(actual_digraph, nx.DiGraph)

    def test_create_digraph_forwards(self):
        root_node, transactions = self.fixture.get_data_create_digraph_forwards()
        values = np.array(transactions)
        actual_digraph = self.network_strategy.create_digraph(
            root_node=root_node, values=values, backwards=False
        )
        expected_nodes = [0, 1]
        expected_edges = [(0, 1)]
        assert list(actual_digraph.nodes) == expected_nodes
        assert list(actual_digraph.edges) == expected_edges
        self.assertIsInstance(actual_digraph, nx.DiGraph)

    def test_find_path(self):
        source = 0
        edges = [(i, i + 1) for i in range(4)]
        D = nx.DiGraph()
        D.add_edges_from(edges)
        expected_path = [i for i in range(5)]
        actual_path = self.network_strategy.find_path(G=D, source=source)
        assert actual_path == expected_path

    def test_get_transactions(self):
        expected_transactions = self.fixture.get_transactions_data()
        path = [i for i in range(5)]
        D = nx.DiGraph()
        D.add_nodes_from(
            [(node, {"transaction": att}) for node, att in zip(path, expected_transactions)]
        )
        D.add_edges_from([(i, i + 1) for i in range(4)])
        actual_transactions = self.network_strategy.get_transactions(G=D, path=path)
        assert actual_transactions == expected_transactions

    def test_get_source(self):
        source_node = self.fixture.get_source_data()
        actual_source = self.network_strategy.get_source(node=source_node)
        expected_source = 0
        assert expected_source == actual_source

    def test_get_source_backwards(self):
        source_node = self.fixture.get_source_data()
        actual_source = self.network_strategy.get_source(
            node=source_node, backwards=True
        )
        expected_source = 0
        assert expected_source == actual_source
