import abc
import unittest
from itertools import chain, islice, repeat
from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from parameterized import parameterized

from mlex.utils.network_strategy import (
    EulerianPathStrategy,
    HamiltonianPathStrategy,
    NetworkPathStrategy,
)
from mlex.utils.schema import Transaction
from mlex.utils.sorter import AccountBalanceSorter

from .data.sort_fixtures import TransactionSortFixture


def compose_integrity_scenario():
    fixture = TransactionSortFixture.get_data_transaction_integrity()
    find_path_behaviors = [
        [None, [[]]],
        chain(islice(repeat(None), 3), [[]]),
        [[]],
        chain([None], repeat([])),
        chain([None], repeat([])),
    ]
    get_transaction_behaviors = [
        [range(1, 5)],
        [range(1, 5)],
        [range(1, 5)],
        [[1], range(2, 6), range(6, 10), range(10, 14)],
        [[13], range(11, 7, -1), range(7, 3, -1), range(3,-1, -1)]
    ]

    return [
        (*f, path_b, get_behavior)
        for f, path_b, get_behavior in zip(
            fixture, find_path_behaviors, get_transaction_behaviors
        )
    ]


class TestSorterInternalLogic(unittest.TestCase):
    def setUp(self):
        self.network_strategy = MagicMock(spec=NetworkPathStrategy)
        self.sorter = AccountBalanceSorter(network_strategy=self.network_strategy)

    def test_sort_empty_values(self):
        X = np.empty((0, 6))
        with pytest.raises(ValueError, match="Input data X is empty"):
            self.sorter.sort(X=X)

    def test_sort_invalid_dimension_array(self):
        X = np.zeros((1, 7))
        with pytest.raises(
            ValueError, match="Column size of 7 is greater than max column size 6"
        ):
            self.sorter.sort(X=X)

    def test_sort_timestep_with_single_transaction(self):
        transactions = [[0, "ACCOUNT-ID1", 0, 0, 100.0, 50.0]]
        X = np.array(transactions)
        sorted_transactions, inconsistent_transactions = self.sorter.sort(X=X)
        expected_sorted_transactions = np.array([0])
        expected_inconsistent_transactions = np.empty(0)
        assert_array_equal(sorted_transactions, expected_sorted_transactions)
        assert_array_equal(
            inconsistent_transactions, expected_inconsistent_transactions
        )


class SorterContract(abc.ABC):
    @abc.abstractmethod
    def get_network_strategy(self):
        pass

    def setUp(self):
        self.network_strategy = self.get_network_strategy()
        self.sorter = AccountBalanceSorter(network_strategy=self.network_strategy)
        self.fixture = TransactionSortFixture()

    def setup_scenario_mixed_consistent_and_inconsistent(self):
        return self.fixture.get_data_mixed_consistent_and_inconsistent()

    def test_sort_mixed_consistent_and_inconsistent(self):
        transactions = self.setup_scenario_mixed_consistent_and_inconsistent()
        X = np.array(transactions)
        sorted_transactions, inconsistent_transactions = self.sorter.sort(X=X)
        expected_sorted_indices = np.array([i for i in range(4)])
        expected_inconsistent_indices = np.array([i for i in range(4, 8, 1)])
        assert_array_equal(inconsistent_transactions, expected_inconsistent_indices)
        assert_array_equal(sorted_transactions, expected_sorted_indices)

    def setup_scenario_alternating_integrity_segments(self):
        return self.fixture.get_data_alternating_integrity_segments()

    def test_sort_alternating_integrity_segments(self):
        transactions = self.setup_scenario_alternating_integrity_segments()
        X = np.array(transactions)
        expected_sorted_indices = np.array(
            [i for i in range(4)] + [i for i in range(8, 12)]
        )
        expected_inconsistent_indices = np.array(
            [i for i in range(4, 8, 1)] + [i for i in range(12, 16, 1)]
        )

        sorted_transactions, inconsistent_transactions = self.sorter.sort(X=X)
        assert_array_equal(inconsistent_transactions, expected_inconsistent_indices)
        assert_array_equal(sorted_transactions, expected_sorted_indices)

    def setup_scenario_missing_balance_adjacent_timesteps(self):
        return self.fixture.get_data_missing_balance_adjacent_timesteps()

    def test_sort_missing_balance_adjacent_timesteps(self):
        transactions = self.setup_scenario_missing_balance_adjacent_timesteps()
        X = np.array(transactions)
        expected_sorted_indices = np.array([i for i in range(8)])
        expected_inconsistent_indices = np.empty(0)
        sorted_transactions, inconsistent_transactions = self.sorter.sort(X=X)
        assert_array_equal(inconsistent_transactions, expected_inconsistent_indices)
        assert_array_equal(sorted_transactions, expected_sorted_indices)

    def setup_scenario_account_grouping(self):
        return self.fixture.get_data_account_grouping()

    def test_sort_account_grouping(self):
        transactions = self.setup_scenario_account_grouping()
        X = np.array(transactions)
        expected_sorted_indices = np.array([i for i in range(8)])
        expected_inconsistent_indices = np.empty(0)
        sorted_transactions, inconsistent_transactions = self.sorter.sort(X=X)
        assert_array_equal(inconsistent_transactions, expected_inconsistent_indices)
        assert_array_equal(sorted_transactions, expected_sorted_indices)

    def setup_scenario_balance_type_grouping(self):
        return self.fixture.get_data_balance_type_grouping()

    def test_sort_balance_type_grouping(self):
        transactions = self.setup_scenario_balance_type_grouping()
        X = np.array(transactions)
        expected_sorted_indices = np.array([0, 1, 4, 5, 2, 3, 6, 7])
        expected_inconsistent_indices = np.empty(0)
        sorted_transactions, inconsistent_transactions = self.sorter.sort(X=X)
        assert_array_equal(inconsistent_transactions, expected_inconsistent_indices)
        assert_array_equal(sorted_transactions, expected_sorted_indices)

    @parameterized.expand(TransactionSortFixture.get_data_transaction_integrity)
    def test_transaction_integrity(self, transactions, expected_indices):
        X = np.array(transactions)
        sorted_transactions, inconsistent_transactions = self.sorter.sort(X)
        expected_inconsistent_indices = np.empty(0)
        assert_array_equal(inconsistent_transactions, expected_inconsistent_indices)
        assert_array_equal(sorted_transactions, expected_indices)


class TestSorterMockStrategy(SorterContract, unittest.TestCase):
    def get_network_strategy(self):
        return MagicMock(spec=NetworkPathStrategy)

    def setUp(self):
        super().setUp()
        self.mock_graph = MagicMock(name="TransactionDigraph")

    def setup_scenario_mixed_consistent_and_inconsistent(self):
        transactions = self.fixture.get_data_mixed_consistent_and_inconsistent()
        transactions_obj = [
            Transaction(**dict(zip(Transaction.model_fields.keys(), t)))
            for t in transactions
        ]
        self.network_strategy.create_digraph.return_value = self.mock_graph
        self.network_strategy.find_path.side_effect = chain(
            [[0, 1, 2, 3]], repeat(None)
        )
        self.network_strategy.get_source.return_value = 0
        self.network_strategy.get_transactions.return_value = transactions_obj[1:4]
        return transactions

    def setup_scenario_alternating_integrity_segments(self):
        transactions = self.fixture.get_data_alternating_integrity_segments()
        transactions_obj = [
            Transaction(**dict(zip(Transaction.model_fields.keys(), t)))
            for t in transactions
        ]
        self.network_strategy.create_digraph.return_value = self.mock_graph
        self.network_strategy.get_source.return_value = 0
        mock_path = []
        self.network_strategy.find_path.side_effect = chain(
            [mock_path], islice(repeat(None), 5), [mock_path], repeat(None)
        )

        self.network_strategy.get_transactions.side_effect = chain(
            [transactions_obj[1:4]], [transactions_obj[9:12]]
        )
        return transactions

    def setup_scenario_missing_balance_adjacent_timesteps(self):
        transactions = self.fixture.get_data_missing_balance_adjacent_timesteps()
        transactions_obj = [
            Transaction(**dict(zip(Transaction.model_fields.keys(), t)))
            for t in transactions
        ]
        self.network_strategy.create_digraph.return_value = self.mock_graph
        self.network_strategy.get_source.return_value = 0
        mock_path = [i for i in range(4)]
        self.network_strategy.find_path.side_effect = chain(
            [mock_path], [None], [mock_path]
        )
        self.network_strategy.get_transactions.side_effect = chain(
            [transactions_obj[1:4]], [transactions_obj[5:8]]
        )
        return transactions

    def setup_scenario_account_grouping(self):
        transactions = self.fixture.get_data_account_grouping()
        transactions_obj = [
            Transaction(**dict(zip(Transaction.model_fields.keys(), t)))
            for t in transactions
        ]
        self.network_strategy.create_digraph.return_value = self.mock_graph
        self.network_strategy.get_source.return_value = 0
        mock_path = [i for i in range(4)]
        self.network_strategy.find_path.side_effect = chain(repeat(mock_path))
        self.network_strategy.get_transactions.side_effect = chain(
            [transactions_obj[1:4]], [transactions_obj[5:8]]
        )
        return transactions

    def setup_scenario_balance_type_grouping(self):
        transactions = self.fixture.get_data_balance_type_grouping()
        transactions_obj = [
            Transaction(**dict(zip(Transaction.model_fields.keys(), t)))
            for t in transactions
        ]
        self.network_strategy.create_digraph.return_value = self.mock_graph
        self.network_strategy.get_source.return_value = 0
        mock_path = [i for i in range(4)]
        self.network_strategy.find_path.side_effect = chain(repeat(mock_path))
        self.network_strategy.get_transactions.side_effect = [
            [transactions_obj[i] for i in [1, 4, 5]],
            [transactions_obj[i] for i in [3, 6, 7]],
        ]
        return transactions

    @parameterized.expand(compose_integrity_scenario())
    def test_transaction_integrity(
        self, transactions, expected_indices, path_behavior, get_tx_behavior
    ):
        X = np.array(transactions)
        transactions_obj = [
            Transaction(**dict(zip(Transaction.model_fields.keys(), t)))
            for t in transactions
        ]
        self.network_strategy.create_digraph.return_value = self.mock_graph
        self.network_strategy.get_source.return_value = 0
        self.network_strategy.find_path.side_effect = path_behavior
        self.network_strategy.get_transactions.side_effect = [[transactions_obj[expected_indices[i]] for i in get_tx] for get_tx in get_tx_behavior]
        sorted_transactions, inconsistent_transactions = self.sorter.sort(X)
        expected_inconsistent_indices = np.empty(0)
        assert_array_equal(inconsistent_transactions, expected_inconsistent_indices)
        assert_array_equal(sorted_transactions, expected_indices)


class TestSorterHamiltonianStrategy(SorterContract, unittest.TestCase):
    def get_network_strategy(self):
        return HamiltonianPathStrategy()


class TestSorterEulerianStrategy(SorterContract, unittest.TestCase):
    def get_network_strategy(self):
        return EulerianPathStrategy()


if __name__ == "__main__":
    unittest.main()
