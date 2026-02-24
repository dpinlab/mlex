import os
from collections import deque
from operator import attrgetter

import numpy as np

from .network_strategy import NetworkPathStrategy
from .schema import Transaction

PATH = os.path.dirname(os.path.abspath(__file__))

INDEX = 0
ACCOUNT_ID = 1
BALANCE_TYPE = 2
TIMESTEP = 3
BALANCE_AMOUNT = 4
PREVIOUS_BALANCE = 5
MAX_COLUMNS = 6


class AccountBalanceSorter:
    def __init__(self, network_strategy: NetworkPathStrategy):
        self.output_path = [
            os.path.join("outputs", "sorted.csv"),
            os.path.join("outputs", "inconsistent.csv"),
        ]
        self.network_strategy = network_strategy
        self.__sorted_transactions = []
        self.__inconsistent_transactions = set()

    def sort(self, X: np.ndarray):
        """
        Parameters
        ----------
            X : numpy.ndarray
             Input data array with shape (n_samples, 6). Specific index usage:
                - X[0] (int): An arbitrary index used to link data to its original position.
                - X[1] (str): A unique account identifier used to group transactions.
                - X[2] (str): The balance type applied to the transaction.
                - X[3] (int): The timestep that indicates the temporal order of the transaction.
                - X[4] (float): The account balance at the time of the transaction.
                - X[5] (float): The previous balance, calculated using the transaction value, the current balance, and the operation.
        Returns
        ----------
            sorted_transactions : numpy.ndarray
                A one-dimensional array that specifies the sequence of indices in the input data X.
            inconsistent_transactions : numpy.ndarray
                A one-dimensional array listing all indices in the input data X where an order could not be determined.
        """
        if X.size == 0:
            raise ValueError("Input data X is empty")
        if X[0].size > 6:
            raise ValueError(
                f"Column size of {X[0].size} is greater than max column size {MAX_COLUMNS}"
            )
        if len(X) == 1:
            self.__sorted_transactions.append(int(X[0][0]))
            return (np.array(self.__sorted_transactions), np.empty(0))
        X_transform = X.copy()
        accounts = np.unique(X_transform[:, ACCOUNT_ID])
        for account in accounts:
            account_transactions = X_transform[X_transform[:, ACCOUNT_ID] == account]
            balance_types = np.unique(account_transactions[:, BALANCE_TYPE])
            for balance_type in balance_types:
                transactions = account_transactions[
                    account_transactions[:, BALANCE_TYPE] == balance_type
                ]
                transactions = transactions[np.argsort(transactions[:, TIMESTEP])]
                transactions_obj = np.array(
                    [
                        Transaction(**dict(zip(Transaction.model_fields.keys(), t)))
                        for t in transactions
                    ]
                )
                sorted_indices = self.__sort_account(transactions_obj)
                if sorted_indices is not None:
                    self.__sorted_transactions.extend(sorted_indices)
        return self.__process_transactions()

    def __sort_account(self, transactions_obj: np.ndarray):
        timesteps = np.fromiter(
            map(attrgetter("timestep"), transactions_obj),
            dtype=int,
            count=len(transactions_obj),
        )
        root_timestep_index = self.__get_root(timesteps=timesteps)
        root_timestep = timesteps[root_timestep_index]
        root_transactions = transactions_obj[timesteps == root_timestep]
        if len(root_transactions) > 1:
            S = self.__resolve_transaction_precendence(
                root_transactions=root_transactions
            )
        else:
            S = deque([root_transactions[0]])
        if S is None:
            previous_remaining_values = transactions_obj[timesteps < root_timestep]
            next_remaining_values = transactions_obj[timesteps > root_timestep]
            previous_result = (
                self.__sort_account(previous_remaining_values)
                if previous_remaining_values.size > 0
                else None
            )
            next_result = (
                self.__sort_account(next_remaining_values)
                if next_remaining_values.size > 0
                else None
            )
            return (previous_result or []) + (next_result or [])

        if root_timestep_index > 0:
            previous_dates = np.unique(timesteps[:root_timestep_index])
            self.sort_previous_transactions(
                S=S,
                previous_timesteps=previous_dates,
                values=transactions_obj[:root_timestep_index],
            )

        next_timesteps_indices = timesteps > root_timestep
        next_timesteps = np.unique(timesteps[next_timesteps_indices])
        if len(next_timesteps) > 0:
            self.sort_next_transactions(
                S=S,
                next_timesteps=next_timesteps,
                values=transactions_obj[next_timesteps_indices],
            )

        result = [transaction.index for transaction in S]

        if len(transactions_obj) != len(result):
            S_first_date = S[0].timestep
            S_last_date = S[-1].timestep
            previous_remaining_values = transactions_obj[timesteps < S_first_date]
            next_remaining_values = transactions_obj[timesteps > S_last_date]
            if len(previous_remaining_values) > 1:
                previous_result = self.__sort_account(previous_remaining_values)
                if previous_result is not None:
                    result = previous_result + result
            if len(next_remaining_values) > 1:
                next_result = self.__sort_account(next_remaining_values)
                if next_result is not None:
                    result = result + next_result
        self.__inconsistent_transactions.difference_update(set(result))
        return result

    def __resolve_transaction_precendence(self, root_transactions: np.ndarray):
        next_timesteps = [root_transactions[0].timestep]
        for i, transaction in enumerate(root_transactions):
            values = np.delete(root_transactions, i, axis=0)
            S = deque([transaction])
            self.sort_next_transactions(
                S=S, next_timesteps=next_timesteps, values=values
            )
            if len(S) == len(root_transactions):
                return S
        self.__inconsistent_transactions |= set(
            map(attrgetter("index"), root_transactions)
        )
        return None

    def sort_previous_transactions(
        self, S: deque, previous_timesteps: np.ndarray, values: np.ndarray
    ):
        timesteps = np.fromiter(
            map(attrgetter("timestep"), values), dtype=int, count=len(values)
        )
        for timestep in reversed(previous_timesteps):
            target_transactions = values[timesteps == timestep]
            balance_D = self.network_strategy.create_digraph(
                values=target_transactions, root_node=S[0], backwards=True
            )
            source = self.network_strategy.get_source(node=S[0], backwards=True)
            path = self.network_strategy.find_path(G=balance_D, source=source)
            if path is None:
                self.__inconsistent_transactions |= set(
                    map(attrgetter("index"), target_transactions)
                )
                return
            S.extendleft(
                self.network_strategy.get_transactions(G=balance_D, path=path[1:])
            )

    def sort_next_transactions(
        self, S: deque, next_timesteps: np.ndarray, values: np.ndarray
    ):
        timesteps = np.fromiter(
            map(attrgetter("timestep"), values), dtype=int, count=len(values)
        )
        for timestep in next_timesteps:
            target_transactions = values[timesteps == timestep]
            balance_D = self.network_strategy.create_digraph(
                values=target_transactions, root_node=S[-1], backwards=False
            )
            source = self.network_strategy.get_source(node=S[-1], backwards=False)
            path = self.network_strategy.find_path(G=balance_D, source=source)
            if path is None:
                self.__inconsistent_transactions |= set(
                    map(attrgetter("index"), target_transactions)
                )
                return
            S.extend(self.network_strategy.get_transactions(G=balance_D, path=path[1:]))

    def __process_transactions(self):
        return (
            np.array(self.__sorted_transactions),
            np.array(list(self.__inconsistent_transactions)),
        )

    def __get_root(self, timesteps: np.ndarray):
        unique_dates, date_counts = np.unique(timesteps, return_counts=True)
        minimum_date_count = np.argmin(date_counts)
        minimum_date = unique_dates[minimum_date_count]
        return np.argmax(timesteps == minimum_date)
