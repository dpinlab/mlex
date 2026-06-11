from datetime import datetime

import pytest
from mlex.analysis.consistency import IntraTimestamp
from mlex import TransactionFlow


class TestIntraTimestamp:
    def test_empty_transaction_flow_list(self):
        with pytest.raises(
            ValueError, match=r"At least one TransactionFlow list must be provided."
        ):
            IntraTimestamp.balance_consistency(T=[])

    def test_transaction_sequence_not_unique_timestamp(self):
        with pytest.raises(
            ValueError,
            match=r"Transaction sequence at index 0 has multiple timestamps. Each sequence must have unique timestamp.",
        ):
            flow_1 = TransactionFlow(
                tx_index=1,
                balance_amount=100.0,
                tx_value=100.0,
                timestamp=datetime(year=2026, month=1, day=1),
            )
            flow_2 = TransactionFlow(
                tx_index=1,
                balance_amount=100.0,
                tx_value=100.0,
                timestamp=datetime(year=2026, month=1, day=2),
            )
            IntraTimestamp.balance_consistency(T=[[flow_1, flow_2]])

    @pytest.mark.parametrize(
        "fixture_name", ["open_eulerian_trail", "closed_eulerian_trail"]
    )
    def test_single_consistent_timestamp(self, fixture_name, request):
        fixture_values = request.getfixturevalue(fixture_name)
        transaction_flows, expected_trail, s = fixture_values
        T = [transaction_flows]
        S_expected = [expected_trail]
        W_expected = []
        S, W = IntraTimestamp.balance_consistency(T)
        assert S == S_expected
        assert W == W_expected

    def test_single_inconsistent_timestamp_disconnected_open_eulerian_components(
        self, inconsisten_timestamp_multiple_open_eulerian_components
    ):
        transaction_flows, expected_component_sequences = (
            inconsisten_timestamp_multiple_open_eulerian_components
        )
        T = [transaction_flows]
        S_expected = []
        W_expected = [(set(), {sequence for sequence in expected_component_sequences})]
        S, W = IntraTimestamp.balance_consistency(T=T)
        assert S == S_expected
        assert W == W_expected

    def test_single_inconsistent_timestamp_disconnected_closed_eulerian_components(
        self, inconsistent_timestamp_multiple_closed_eulerian_components
    ):
        transaction_flows, expected_component_sequences = (
            inconsistent_timestamp_multiple_closed_eulerian_components
        )
        T = [transaction_flows]
        S_expected = []
        W_expected = [(set(), {sequence for sequence in expected_component_sequences})]
        S, W = IntraTimestamp.balance_consistency(T=T)
        assert S == S_expected
        assert W == W_expected

    def test_single_inconsistent_timestamp_connected_asymetrical_single_component(
        self, inconsistent_timestamp_single_asymetrical_component
    ):
        transaction_flows, expected_transaction_indices = (
            inconsistent_timestamp_single_asymetrical_component
        )
        A_1, A_2 = expected_transaction_indices
        A = set()
        A.add((A_1, A_2))
        T = [transaction_flows]
        S_expected = []
        W_expected = [(A, set())]
        S, W = IntraTimestamp.balance_consistency(T=T)
        assert S == S_expected
        assert W == W_expected

    def test_mixed_consistent_inconsistent_multiple_timestamps(
        self,
        open_eulerian_trail,
        closed_eulerian_trail,
        inconsisten_timestamp_multiple_open_eulerian_components,
        inconsistent_timestamp_multiple_closed_eulerian_components,
        inconsistent_timestamp_single_asymetrical_component,
    ):
        open_eulerian_trail_transaction_flows, open_eulerian_expected_trail, _ = (
            open_eulerian_trail
        )
        closed_eulerian_trail_transaction_flows, closed_eulerian_expected_trail, _ = (
            closed_eulerian_trail
        )
        (
            multiple_open_eulerian_components_transaction_flows,
            open_eulerian_components_expected_sequence,
        ) = inconsisten_timestamp_multiple_open_eulerian_components
        (
            multiple_closed_eulerian_components_transaction_flows,
            closed_eulerian_components_expected_sequence,
        ) = inconsistent_timestamp_multiple_closed_eulerian_components
        single_asymetrical_component_transaction_flows, expected_transaction_indices = (
            inconsistent_timestamp_single_asymetrical_component
        )
        A_1, A_2 = expected_transaction_indices
        A = set()
        A.add((A_1, A_2))
        T = [
            open_eulerian_trail_transaction_flows,
            closed_eulerian_trail_transaction_flows,
            multiple_open_eulerian_components_transaction_flows,
            multiple_closed_eulerian_components_transaction_flows,
            single_asymetrical_component_transaction_flows,
        ]
        S_expected = [open_eulerian_expected_trail, closed_eulerian_expected_trail]
        W_expected = [
            (
                set(),
                {sequence for sequence in open_eulerian_components_expected_sequence},
            ),(
                set(),
                {sequence for sequence in closed_eulerian_components_expected_sequence},
            ),
            (A, set()),
        ]
        S, W = IntraTimestamp.balance_consistency(T=T)
        assert S == S_expected
        assert W == W_expected
