from venv import create

import networkx as nx
import pytest

from mlex.analysis.network import (
    DisconnectedMultiDigraphError,
    NoEulerianTrailError,
    TransactionMultiDigraph,
    Trail,
)


class TestTransactionMultiDigraph:

    def test_base_case_empty_transaction_list(self):
        with pytest.raises(
            ValueError,
            match=r"No transaction flow received\. Transaction flow network must have at least one flow\.",
        ):
            TransactionMultiDigraph.create_multigraph(
                transaction_flows=[],
            )

    def test_single_transaction_flow_multidigraph_attributes(
        self, single_transaction_flow
    ):
        D = TransactionMultiDigraph.create_multigraph(
            transaction_flows=single_transaction_flow,
        )
        flow = single_transaction_flow[0]
        flow_previous_balance, flow_balance, flow_index = (
            flow.balance_amount - flow.tx_value,
            flow.balance_amount,
            flow.tx_index,
        )

        assert D.number_of_edges() == 1
        assert D.number_of_nodes() == 2
        assert flow_previous_balance in D.nodes
        assert flow_balance in D.nodes

        flow_data = D.get_edge_data(
            u=flow_previous_balance, v=flow.balance_amount, key=flow_index
        )
        assert flow_data["tx_value"] == flow.tx_value
        assert flow_data["timestamp"] == flow.timestamp

    def test_simple_nontrivial_connected_multidigraph(
        self, simple_nontrivial_connected_transaction_flows
    ):
        D = TransactionMultiDigraph.create_multigraph(
            transaction_flows=simple_nontrivial_connected_transaction_flows,
        )
        flow1, flow2 = simple_nontrivial_connected_transaction_flows
        connected_components = list(nx.weakly_connected_components(D))
        flow1_previous_balance = flow1.balance_amount - flow1.tx_value
        flow2_balance_amount = flow2.balance_amount
        assert len(connected_components) == 1
        assert D.number_of_nodes() == 3
        assert D.number_of_edges() == 2
        assert nx.has_path(D, flow1_previous_balance, flow2_balance_amount)

    def test_simple_nontrivial_disconnected_multidigraph(
        self, simple_nontrivial_disconnected_transaction_flows
    ):
        D = TransactionMultiDigraph.create_multigraph(
            transaction_flows=simple_nontrivial_disconnected_transaction_flows,
        )
        flow1, flow2 = simple_nontrivial_disconnected_transaction_flows
        connected_components = list(nx.weakly_connected_components(D))
        flow1_previous_balance = flow1.balance_amount - flow1.tx_value
        flow2_balance_amount = flow2.balance_amount
        assert len(connected_components) == 2
        assert D.number_of_nodes() == 4
        assert D.number_of_edges() == 2
        assert not nx.has_path(D, flow1_previous_balance, flow2_balance_amount)


class TestEulerianTrail:

    def test_base_case_disconnected_multidigraph(
        self, simple_nontrivial_disconnected_transaction_flows
    ):
        D = TransactionMultiDigraph.create_multigraph(
            transaction_flows=simple_nontrivial_disconnected_transaction_flows,
        )
        with pytest.raises(
            DisconnectedMultiDigraphError,
            match="Cannot create Eulerian Trail on a disconnected multigraph.",
        ):
            trail = Trail.eulerian_trail(D)

    def test_no_eulerian_trail_multidigraph(
        self, no_eulerian_trail_transaction_flows, debug_network
    ):
        D = TransactionMultiDigraph.create_multigraph(
            transaction_flows=no_eulerian_trail_transaction_flows,
        )
        with pytest.raises(
            NoEulerianTrailError,
            match=f"Transaction MultiDigraph is not pseudosymmetric and has no open eulerian trail",
        ):
            trail = Trail.eulerian_trail(D)

    def test_open_eulerian_trail(self, open_eulerian_trail, debug_network):
        transaction_flows, expected_trail = open_eulerian_trail
        D = TransactionMultiDigraph.create_multigraph(
            transaction_flows=transaction_flows
        )
        trail = Trail.eulerian_trail(D)
        assert trail == expected_trail

    def test_closed_eulerian_trail(self, closed_eulerian_trail, debug_network):
        transaction_flows, expected_trail = closed_eulerian_trail
        D = TransactionMultiDigraph.create_multigraph(
            transaction_flows=transaction_flows
        )
        trail = Trail.eulerian_trail(D)
        assert trail == expected_trail
