import pytest
from mlex.utils.schema import TransactionFlow
from datetime import datetime
import json
import networkx as nx


@pytest.fixture
def debug_network():
    def _create_json_schema(D: nx.multidigraph):
        network = {
            "kind": {"graph": True},
            "nodes": [
                {
                    "id": str(node),
                    "label": str(node),
                    "color": "green" if node >= 0 else "red",
                    "shape": "box",
                }
                for node in D
            ],
            "edges": [
                {
                    "from": str(u),
                    "to": str(v),
                    "color": "red" if d['tx_value'] < 0 else "green",
                    "label": f"{k}: {d['tx_value']}",
                }
                for (u, v, k, d) in D.edges(data=True, keys=True)
            ],
        }
        return json.dumps(network)

    return _create_json_schema


@pytest.fixture
def single_transaction_flow():
    transaction_flows = [
        TransactionFlow(
            tx_index=0,
            balance_amount=10.0,
            tx_value=1.0,
            timestamp=datetime(year=2026, month=1, day=1),
        )
    ]
    return transaction_flows


@pytest.fixture
def simple_nontrivial_connected_transaction_flows():
    transaction_flows = [
        TransactionFlow(
            tx_index=0,
            balance_amount=10.0,
            tx_value=1.0,
            timestamp=datetime(year=2026, month=1, day=1),
        ),
        TransactionFlow(
            tx_index=1,
            balance_amount=11.0,
            tx_value=1.0,
            timestamp=datetime(year=2026, month=1, day=1),
        ),
    ]
    return transaction_flows


@pytest.fixture
def simple_nontrivial_disconnected_transaction_flows():
    transaction_flows = [
        TransactionFlow(
            tx_index=0,
            balance_amount=10.0,
            tx_value=1.0,
            timestamp=datetime(year=2026, month=1, day=1),
        ),
        TransactionFlow(
            tx_index=1,
            balance_amount=12.0,
            tx_value=1.0,
            timestamp=datetime(year=2026, month=1, day=1),
        ),
    ]
    return transaction_flows


@pytest.fixture
def no_eulerian_trail_transaction_flows():
    transaction_flows = [
        TransactionFlow(
            tx_index=0,
            balance_amount=40.0,
            tx_value=-20.0,
            timestamp=datetime(year=2026, month=1, day=1),
        ),
        TransactionFlow(
            tx_index=1,
            balance_amount=20.0,
            tx_value=-40.0,
            timestamp=datetime(year=2026, month=1, day=1),
        ),
        TransactionFlow(
            tx_index=2,
            balance_amount=100.0,
            tx_value=40.0,
            timestamp=datetime(year=2026, month=1, day=1),
        ),
        TransactionFlow(
            tx_index=3,
            balance_amount=20.0,
            tx_value=-20.0,
            timestamp=datetime(year=2026, month=1, day=1),
        ),
        TransactionFlow(
            tx_index=4,
            balance_amount=100.0,
            tx_value=80.0,
            timestamp=datetime(year=2026, month=1, day=1),
        ),
    ]
    return transaction_flows


@pytest.fixture
def open_eulerian_trail():
    transaction_flows = [
        TransactionFlow(
            tx_index=0,
            balance_amount=100.0,
            tx_value=-100.0,
            timestamp=datetime(year=2026, month=1, day=1),
        ),
        TransactionFlow(
            tx_index=1,
            balance_amount=150.0,
            tx_value=50.0,
            timestamp=datetime(year=2026, month=1, day=1),
        ),
        TransactionFlow(
            tx_index=2,
            balance_amount=225.0,
            tx_value=75.0,
            timestamp=datetime(year=2026, month=1, day=1),
        ),
        TransactionFlow(
            tx_index=3,
            balance_amount=185.0,
            tx_value=-40.0,
            timestamp=datetime(year=2026, month=1, day=1),
        ),
        TransactionFlow(
            tx_index=4,
            balance_amount=200.0,
            tx_value=15.0,
            timestamp=datetime(year=2026, month=1, day=1),
        ),
        TransactionFlow(
            tx_index=5,
            balance_amount=225.0,
            tx_value=25.0,
            timestamp=datetime(year=2026, month=1, day=1),
        ),
    ]
    expected_trail = [i for i in range(6)]
    return transaction_flows, expected_trail


@pytest.fixture
def closed_eulerian_trail():
    transaction_flows = [
        TransactionFlow(
            tx_index=0,
            balance_amount=200.0,
            tx_value=100.0,
            timestamp=datetime(year=2026, month=1, day=1),
        ),
        TransactionFlow(
            tx_index=1,
            balance_amount=100.0,
            tx_value=100.0,
            timestamp=datetime(year=2026, month=1, day=1),
        ),
        TransactionFlow(
            tx_index=2,
            balance_amount=400.0,
            tx_value=100.0,
            timestamp=datetime(year=2026, month=1, day=1),
        ),
        TransactionFlow(
            tx_index=3,
            balance_amount=300.0,
            tx_value=100.0,
            timestamp=datetime(year=2026, month=1, day=1),
        ),
        TransactionFlow(
            tx_index=4,
            balance_amount=500.0,
            tx_value=100.0,
            timestamp=datetime(year=2026, month=1, day=1),
        ),
        TransactionFlow(
            tx_index=5,
            balance_amount=0.0,
            tx_value=-500.0,
            timestamp=datetime(year=2026, month=1, day=1),
        ),
    ]
    expected_trail = [0, 3, 2, 4, 5, 1]
    return transaction_flows, expected_trail
