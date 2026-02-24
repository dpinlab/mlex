from mlex.utils.schema import Transaction


class TransactionSortFixture:
    def get_data_mixed_consistent_and_inconsistent(self):
        return [
            [0, "ACCOUNT-ID1", "0", 0, 100.0, 50.0],
            [1, "ACCOUNT-ID1", "0", 0, 250.0, 100.0],
            [2, "ACCOUNT-ID1", "0", 0, 200.0, 250.0],
            [3, "ACCOUNT-ID1", "0", 0, 300.0, 200.0],
            [4, "ACCOUNT-ID1", "0", 1, 100.0, 25.0],
            [5, "ACCOUNT-ID1", "0", 1, 200.0, 175.0],
            [6, "ACCOUNT-ID1", "0", 1, 300.0, 125.0],
            [7, "ACCOUNT-ID1", "0", 1, 400.0, 50.0],
        ]

    def get_data_alternating_integrity_segments(self):
        return [
            [0, "ACCOUNT-ID1", "0", 0, 100.0, 50.0],
            [1, "ACCOUNT-ID1", "0", 0, 200.0, 100.0],
            [2, "ACCOUNT-ID1", "0", 0, 300.0, 200.0],
            [3, "ACCOUNT-ID1", "0", 0, 400.0, 300.0],
            [4, "ACCOUNT-ID1", "0", 1, 100.0, 25.0],
            [5, "ACCOUNT-ID1", "0", 1, 200.0, 175.0],
            [6, "ACCOUNT-ID1", "0", 1, 300.0, 125.0],
            [7, "ACCOUNT-ID1", "0", 1, 400.0, 50.0],
            [8, "ACCOUNT-ID1", "0", 2, 400.0, 500.0],
            [9, "ACCOUNT-ID1", "0", 2, 300.0, 400.0],
            [10, "ACCOUNT-ID1", "0", 2, 200.0, 300.0],
            [11, "ACCOUNT-ID1", "0", 2, 100.0, 200.0],
            [12, "ACCOUNT-ID1", "0", 3, 100.0, 25.0],
            [13, "ACCOUNT-ID1", "0", 3, 200.0, 175.0],
            [14, "ACCOUNT-ID1", "0", 3, 300.0, 125.0],
            [15, "ACCOUNT-ID1", "0", 3, 400.0, 50.0],
        ]

    def get_data_missing_balance_adjacent_timesteps(self):
        return [
            [0, "ACCOUNT-ID1", "0", 0, 100.0, 50.0],
            [1, "ACCOUNT-ID1", "0", 0, 200.0, 100.0],
            [2, "ACCOUNT-ID1", "0", 0, 300.0, 200.0],
            [3, "ACCOUNT-ID1", "0", 0, 400.0, 300.0],
            [4, "ACCOUNT-ID1", "0", 1, 600.0, 500.0],
            [5, "ACCOUNT-ID1", "0", 1, 700.0, 600.0],
            [6, "ACCOUNT-ID1", "0", 1, 800.0, 700.0],
            [7, "ACCOUNT-ID1", "0", 1, 900.0, 800.0],
        ]

    def get_data_account_grouping(self):
        return [
            [0, "ACCOUNT-ID1", "0", 0, 100.0, 50.0],
            [1, "ACCOUNT-ID1", "0", 0, 200.0, 100.0],
            [2, "ACCOUNT-ID1", "0", 0, 300.0, 200.0],
            [3, "ACCOUNT-ID1", "0", 0, 400.0, 300.0],
            [4, "ACCOUNT-ID2", "0", 0, 600.0, 500.0],
            [5, "ACCOUNT-ID2", "0", 0, 700.0, 600.0],
            [6, "ACCOUNT-ID2", "0", 0, 800.0, 700.0],
            [7, "ACCOUNT-ID2", "0", 0, 900.0, 800.0],
        ]

    def get_data_balance_type_grouping(self):
        return [
            [0, "ACCOUNT-ID1", "0", 0, 100.0, 50.0],
            [1, "ACCOUNT-ID1", "0", 0, 200.0, 100.0],
            [2, "ACCOUNT-ID1", "1", 0, 500.0, 600.0],
            [3, "ACCOUNT-ID1", "1", 0, 400.0, 500.0],
            [4, "ACCOUNT-ID1", "0", 0, 300.0, 200.0],
            [5, "ACCOUNT-ID1", "0", 0, 400.0, 300.0],
            [6, "ACCOUNT-ID1", "1", 0, 300.0, 400.0],
            [7, "ACCOUNT-ID1", "1", 0, 200.0, 300.0],
        ]

    @staticmethod
    def get_data_transaction_integrity():
        return [
            (
                [
                    [0, "ACCOUNT-ID1", "0", 0, 200.0, 100.0],
                    [1, "ACCOUNT-ID1", "0", 0, 100.0, 0.0],
                    [2, "ACCOUNT-ID1", "0", 0, 400.0, 300.0],
                    [3, "ACCOUNT-ID1", "0", 0, 300.0, 200.0],
                    [4, "ACCOUNT-ID1", "0", 0, 500.0, 400.0],
                ],
                [1, 0, 3, 2, 4],
            ),
            (
                [
                    [0, "ACCOUNT-ID1", "0", 0, 300.0, 200.0],
                    [1, "ACCOUNT-ID1", "0", 0, 400.0, 300.0],
                    [2, "ACCOUNT-ID1", "0", 0, 500.0, 400.0],
                    [3, "ACCOUNT-ID1", "0", 0, 100.0, 0.0],
                    [4, "ACCOUNT-ID1", "0", 0, 200.0, 100.0],
                ],
                [3, 4, 0, 1, 2],
            ),
            (
                [
                    [0, "ACCOUNT-ID1", "0", 0, 100.0, 0.0],
                    [1, "ACCOUNT-ID1", "0", 0, 300.0, 200.0],
                    [2, "ACCOUNT-ID1", "0", 0, 200.0, 100.0],
                    [3, "ACCOUNT-ID1", "0", 0, 500.0, 400.0],
                    [4, "ACCOUNT-ID1", "0", 0, 400.0, 300.0],
                ],
                [0, 2, 1, 4, 3],
            ),
            (
                [
                    [0, "ACCOUNT-ID1", "0", 0, 400.0, 300.0],
                    [1, "ACCOUNT-ID1", "0", 0, 300.0, 200.0],
                    [2, "ACCOUNT-ID1", "0", 1, 0.0, 100.0],
                    [3, "ACCOUNT-ID1", "0", 1, 100.0, 200.0],
                    [4, "ACCOUNT-ID1", "0", 1, 200.0, 300.0],
                    [5, "ACCOUNT-ID1", "0", 1, 300.0, 400.0],
                    [6, "ACCOUNT-ID1", "0", 2, 200.0, 100.0],
                    [7, "ACCOUNT-ID1", "0", 2, 100.0, 0.0],
                    [8, "ACCOUNT-ID1", "0", 2, 400.0, 300.0],
                    [9, "ACCOUNT-ID1", "0", 2, 300.0, 200.0],
                    [10, "ACCOUNT-ID1", "0", 3, 200.0, 300.0],
                    [11, "ACCOUNT-ID1", "0", 3, 300.0, 400.0],
                    [12, "ACCOUNT-ID1", "0", 3, 0.0, 100.0],
                    [13, "ACCOUNT-ID1", "0", 3, 100.0, 200.0],
                ],
                [1, 0, 5, 4, 3, 2, 7, 6, 9, 8, 11, 10, 13, 12],
            ),
            (
                [
                    [0, "ACCOUNT-ID1", "0", 0, 200.0, 100.0],
                    [1, "ACCOUNT-ID1", "0", 0, 100.0, 0.0],
                    [2, "ACCOUNT-ID1", "0", 0, 400.0, 300.0],
                    [3, "ACCOUNT-ID1", "0", 0, 300.0, 200.0],
                    [4, "ACCOUNT-ID1", "0", 1, 0.0, 100.0],
                    [5, "ACCOUNT-ID1", "0", 1, 100.0, 200.0],
                    [6, "ACCOUNT-ID1", "0", 1, 200.0, 300.0],
                    [7, "ACCOUNT-ID1", "0", 1, 300.0, 400.0],
                    [8, "ACCOUNT-ID1", "0", 2, 400.0, 300.0],
                    [9, "ACCOUNT-ID1", "0", 2, 100.0, 0.0],
                    [10, "ACCOUNT-ID1", "0", 2, 300.0, 200.0],
                    [11, "ACCOUNT-ID1", "0", 2, 200.0, 100.0],
                    [12, "ACCOUNT-ID1", "0", 3, 200.0, 300.0],
                    [13, "ACCOUNT-ID1", "0", 3, 300.0, 400.0],
                ],
                [1, 0, 3, 2, 7, 6, 5, 4, 9, 11, 10, 8, 13, 12],
            ),
        ]


class NetworkStrategyFixture:
    def get_data_create_digraph_backwards(self):
        root_transaction = [
            1,
            "ACCOUNT-ID1",
            "0",
            0,
            150.00,
            100.00,
        ]
        root_node = Transaction(
            **dict(zip(Transaction.model_fields.keys(), root_transaction))
        )
        transactions = [
            [
                2,
                "ACCOUNT-ID1",
                "0",
                0,
                100.0,
                50.0,
            ]
        ]
        transaction_nodes = [
            Transaction(**dict(zip(Transaction.model_fields.keys(), t)))
            for t in transactions
        ]
        return root_node, transaction_nodes

    def get_data_create_digraph_forwards(self):
        root_transaction = [
            1,
            "ACCOUNT-ID1",
            "0",
            0,
            150.00,
            50.00,
        ]
        root_node = Transaction(
            **dict(zip(Transaction.model_fields.keys(), root_transaction))
        )
        transactions = [
            [
                2,
                "ACCOUNT-ID1",
                "0",
                0,
                100.0,
                150.0,
            ]
        ]
        transactions_nodes = [
            Transaction(**dict(zip(Transaction.model_fields.keys(), t)))
            for t in transactions
        ]

        return root_node, transactions_nodes

    def get_transactions_data(self):
        transactions = [
            [0, "ACCOUNT-ID1", "0", 0, 100.0, 0.0],
            [1, "ACCOUNT-ID1", "0", 0, 200.0, 100.0],
            [2, "ACCOUNT-ID1", "0", 0, 300.0, 200.0],
            [3, "ACCOUNT-ID1", "0", 0, 400.0, 300.0],
            [4, "ACCOUNT-ID1", "0", 0, 500.0, 400.0],
        ]
        return [
            Transaction(**dict(zip(Transaction.model_fields.keys(), t)))
            for t in transactions
        ]

    def get_source_data(self):
        source_transaction = [
            1,
            "ACCOUNT-ID1",
            "0",
            0,
            150.00,
            100.00,
        ]
        return Transaction(
            **dict(zip(Transaction.model_fields.keys(), source_transaction))
        )
