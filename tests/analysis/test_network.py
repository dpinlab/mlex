from datetime import datetime

import networkx as nx
import pytest

from mlex.analysis.network import (
    DisconnectedMultiDigraphError,
    TransactionMultiDigraph,
    Trail,
)


class TestTransactionMultiDigraph:
    def test_base_case_empty_transaction_list(self):
        with pytest.raises(
            ValueError,
            match=r"No transaction flow received\. Transaction flow network must have at least one flow\.",
        ):
            TransactionMultiDigraph.create_multidigraph(
                transaction_flows=[],
            )

    def test_single_transaction_flow_multidigraph_attributes(
        self, single_transaction_flow
    ):
        G = TransactionMultiDigraph.create_multidigraph(
            transaction_flows=single_transaction_flow,
        )
        flow = single_transaction_flow[0]
        flow_previous_balance, flow_balance, flow_index = (
            flow.balance_amount - flow.tx_value,
            flow.balance_amount,
            flow.tx_index,
        )

        assert G.number_of_edges() == 1
        assert G.number_of_nodes() == 2
        assert flow_previous_balance in G.nodes
        assert flow_balance in G.nodes

        flow_data = G.get_edge_data(
            u=flow_previous_balance, v=flow.balance_amount, key=flow_index
        )
        assert flow_data["tx_value"] == flow.tx_value
        assert flow_data["timestamp"] == flow.timestamp

    def test_simple_nontrivial_connected_multidigraph(
        self, simple_nontrivial_connected_transaction_flows
    ):
        G = TransactionMultiDigraph.create_multidigraph(
            transaction_flows=simple_nontrivial_connected_transaction_flows,
        )
        flow1, flow2 = simple_nontrivial_connected_transaction_flows
        connected_components = list(nx.weakly_connected_components(G))
        flow1_previous_balance = flow1.balance_amount - flow1.tx_value
        flow2_balance_amount = flow2.balance_amount
        assert len(connected_components) == 1
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 2
        assert nx.has_path(G, flow1_previous_balance, flow2_balance_amount)

    def test_simple_nontrivial_disconnected_multidigraph(
        self, simple_nontrivial_disconnected_transaction_flows
    ):
        transaction_flows, s = simple_nontrivial_disconnected_transaction_flows
        G = TransactionMultiDigraph.create_multidigraph(
            transaction_flows=transaction_flows,
        )
        flow1, flow2 = transaction_flows
        connected_components = list(nx.weakly_connected_components(G))
        flow1_previous_balance = flow1.balance_amount - flow1.tx_value
        flow2_balance_amount = flow2.balance_amount
        assert len(connected_components) == 2
        assert G.number_of_nodes() == 4
        assert G.number_of_edges() == 2
        assert not nx.has_path(G, flow1_previous_balance, flow2_balance_amount)


class TestEulerianTrail:
    def test_base_case_disconnected_multidigraph(
        self, simple_nontrivial_disconnected_transaction_flows
    ):
        disconnected_transaction_flows, s = (
            simple_nontrivial_disconnected_transaction_flows
        )
        G = TransactionMultiDigraph.create_multidigraph(
            transaction_flows=disconnected_transaction_flows,
        )
        with pytest.raises(
            DisconnectedMultiDigraphError,
            match="Cannot create Eulerian Circuit on a not strongly connected multidigraph.",
        ):
            circuit = Trail.eulerian_circuit(G, s)

    def test_no_eulerian_trail_multidigraph(
        self, not_pseudo_symmetric_single_component_transaction_flows, debug_network
    ):
        not_eulerian_transaction_flows, s = (
            not_pseudo_symmetric_single_component_transaction_flows
        )
        G = TransactionMultiDigraph.create_multidigraph(
            transaction_flows=not_eulerian_transaction_flows,
        )
        with pytest.raises(
            DisconnectedMultiDigraphError,
            match=f"Cannot create Eulerian Circuit on a not strongly connected multidigraph.",
        ):
            circuit = Trail.eulerian_circuit(G, s)

    def test_open_eulerian_trail(self, open_eulerian_trail, debug_network):
        transaction_flows, expected_trail, dummy_flow = open_eulerian_trail
        G = TransactionMultiDigraph.create_multidigraph(
            transaction_flows=transaction_flows
        )

        G.add_edge(
            u_for_edge=dummy_flow.balance_amount - dummy_flow.tx_value,
            v_for_edge=dummy_flow.balance_amount,
            key=dummy_flow.tx_index,
            **{
                "tx_value": dummy_flow.tx_value,
                "timestamp": dummy_flow.timestamp,
            },
        )
        trail = Trail.eulerian_circuit(
            G,
            s=dummy_flow.balance_amount - dummy_flow.tx_value,
            e=(
                dummy_flow.balance_amount - dummy_flow.tx_value,
                dummy_flow.balance_amount,
                dummy_flow.tx_index,
            ),
        )
        assert list(trail)[1:] == expected_trail

    def test_closed_eulerian_trail(self, closed_eulerian_trail, debug_network):
        transaction_flows, expected_trail, s = closed_eulerian_trail
        G = TransactionMultiDigraph.create_multidigraph(
            transaction_flows=transaction_flows
        )
        trail = Trail.eulerian_circuit(G, s)
        assert list(trail) == expected_trail
