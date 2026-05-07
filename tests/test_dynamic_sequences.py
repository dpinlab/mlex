import unittest
from collections import Counter

import numpy as np

from mlex.features.length_strategy import (
    LengthStrategy,
    UniformRandomLengthStrategy,
)
from mlex.features.sequences import (
    DynamicLengthBatchSampler,
    DynamicSequenceDataset,
)


def _make_data(n_samples=200, n_features=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y = rng.integers(0, 2, size=n_samples).astype(np.float32)
    return X, y


def _make_grouped_data(group_sizes, n_features=3, seed=0):
    rng = np.random.default_rng(seed)
    total = sum(group_sizes)
    features = rng.standard_normal((total, n_features)).astype(np.float32)
    groups = np.concatenate([np.full(size, idx) for idx, size in enumerate(group_sizes)])
    X = np.column_stack([features, groups.astype(np.float32)])
    y = rng.integers(0, 2, size=total).astype(np.float32)
    return X, y, X.shape[1] - 1


class DynamicSequenceDatasetTests(unittest.TestCase):
    def test_requires_non_empty_lengths(self):
        X, y = _make_data()
        with self.assertRaises(ValueError):
            DynamicSequenceDataset(X, y, sequence_lengths=[])

    def test_valid_indices_per_length(self):
        X, y = _make_data(n_samples=100)
        dataset = DynamicSequenceDataset(X, y, sequence_lengths=[5, 20])

        self.assertEqual(len(dataset.valid_indices_for(5)), 100 - 5 + 1)
        self.assertEqual(len(dataset.valid_indices_for(20)), 100 - 20 + 1)

        self.assertEqual(dataset.valid_end_indices_for(5)[0], 4)
        self.assertEqual(dataset.valid_end_indices_for(20)[-1], 99)

    def test_getitem_shapes(self):
        X, y = _make_data(n_samples=50, n_features=6)
        dataset = DynamicSequenceDataset(X, y, sequence_lengths=[3, 10])

        x5, y5 = dataset[(3, 0)]
        self.assertEqual(tuple(x5.shape), (3, 6))
        self.assertEqual(y5.item(), float(y[2]))

        x10, y10 = dataset[(10, 0)]
        self.assertEqual(tuple(x10.shape), (10, 6))
        self.assertEqual(y10.item(), float(y[9]))

    def test_respects_group_boundaries(self):
        X, y, group_col = _make_grouped_data([6, 6, 6])
        dataset = DynamicSequenceDataset(
            X, y, sequence_lengths=[4], group_column_index=group_col
        )

        # Only windows that stay inside a single group of size 6 are valid for length 4.
        # Each group of 6 yields 6 - 4 + 1 = 3 valid windows, so 3 groups * 3 = 9.
        self.assertEqual(len(dataset.valid_indices_for(4)), 9)


class DynamicLengthBatchSamplerTests(unittest.TestCase):
    def setUp(self):
        X, y = _make_data(n_samples=300, seed=1)
        self.dataset = DynamicSequenceDataset(X, y, sequence_lengths=[5, 10, 20])

    def test_each_batch_has_uniform_length(self):
        sampler = DynamicLengthBatchSampler(
            self.dataset,
            batch_size=8,
            strategy=UniformRandomLengthStrategy([5, 10, 20]),
            shuffle=True,
            drop_last=True,
            random_seed=42,
        )

        batches = list(sampler)
        self.assertGreater(len(batches), 0)
        for batch in batches:
            lengths_in_batch = {length for length, _ in batch}
            self.assertEqual(len(lengths_in_batch), 1)
            self.assertEqual(len(batch), 8)

    def test_length_varies_across_batches(self):
        sampler = DynamicLengthBatchSampler(
            self.dataset,
            batch_size=8,
            strategy=UniformRandomLengthStrategy([5, 10, 20]),
            random_seed=42,
        )

        lengths_per_batch = [batch[0][0] for batch in sampler]
        counts = Counter(lengths_per_batch)
        self.assertGreaterEqual(len(counts), 2)

    def test_seed_reproducibility(self):
        def run():
            sampler = DynamicLengthBatchSampler(
                self.dataset,
                batch_size=8,
                strategy=UniformRandomLengthStrategy([5, 10, 20]),
                random_seed=123,
            )
            return [list(batch) for batch in sampler]

        self.assertEqual(run(), run())

    def test_drop_last_false_emits_partial_batches(self):
        sampler_drop = DynamicLengthBatchSampler(
            self.dataset, batch_size=8, drop_last=True, random_seed=0,
        )
        sampler_keep = DynamicLengthBatchSampler(
            self.dataset, batch_size=8, drop_last=False, random_seed=0,
        )

        dropped_total = sum(len(b) for b in sampler_drop)
        kept_total = sum(len(b) for b in sampler_keep)

        expected_kept = sum(len(self.dataset.datasets[length])
                            for length in self.dataset.sequence_lengths)
        self.assertEqual(kept_total, expected_kept)
        self.assertLess(dropped_total, kept_total)

    def test_custom_strategy_plug_point(self):
        calls = []

        class CyclingStrategy(LengthStrategy):
            def __init__(self, lengths):
                super().__init__(lengths)
                self.i = 0

            def next_length(self, rng):
                length = self.lengths[self.i % len(self.lengths)]
                self.i += 1
                calls.append(length)
                return length

        strategy = CyclingStrategy([5, 10, 20])
        sampler = DynamicLengthBatchSampler(
            self.dataset, batch_size=8, strategy=strategy, random_seed=0,
        )
        batches = list(sampler)

        # Strategy was invoked at least once per yielded batch.
        self.assertGreaterEqual(len(calls), len(batches))
        self.assertGreater(len(batches), 0)

    def test_rejects_unknown_strategy_length(self):
        with self.assertRaises(ValueError):
            DynamicLengthBatchSampler(
                self.dataset,
                batch_size=8,
                strategy=UniformRandomLengthStrategy([99]),
            )


class DynamicBatchingEndToEndTests(unittest.TestCase):
    def test_dataloader_iteration_yields_uniform_batches(self):
        import torch
        from torch.utils.data import DataLoader

        X, y = _make_data(n_samples=160, n_features=5, seed=3)
        dataset = DynamicSequenceDataset(X, y, sequence_lengths=[4, 12])
        sampler = DynamicLengthBatchSampler(
            dataset,
            batch_size=8,
            strategy=UniformRandomLengthStrategy([4, 12]),
            random_seed=7,
        )
        loader = DataLoader(dataset, batch_sampler=sampler)

        seen_lengths = set()
        for batch_x, batch_y in loader:
            self.assertEqual(batch_x.ndim, 3)
            self.assertEqual(batch_x.shape[0], 8)
            self.assertEqual(batch_x.shape[2], 5)
            self.assertIn(int(batch_x.shape[1]), {4, 12})
            seen_lengths.add(int(batch_x.shape[1]))
            self.assertEqual(batch_y.shape, torch.Size([8]))

        self.assertTrue(seen_lengths.issubset({4, 12}))
        self.assertGreaterEqual(len(seen_lengths), 1)

    def test_lstm_base_model_trains_with_dynamic_lengths(self):
        import torch
        from mlex.models.base_components.recurrent import LSTMBaseModel

        X_train, y_train = _make_data(n_samples=120, n_features=3, seed=11)
        X_val, y_val = _make_data(n_samples=60, n_features=3, seed=12)

        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)

        lengths = [5, 10]
        strategy = UniformRandomLengthStrategy(lengths)

        model = LSTMBaseModel(
            validation_data=(X_val, y_val),
            input_size=3,
            hidden_size=4,
            num_layers=1,
            output_size=1,
            seq_length=5,
            batch_size=8,
            shuffle_dataloader=True,
            epochs=1,
            patience=1,
            group_index=None,
            random_seed=42,
            device=torch.device("cpu"),
            dynamic_length_strategy=strategy,
            dynamic_drop_last=True,
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        # Ensemble path (union with partial averaging): predictions cover every
        # end-index that any length produces — i.e. starting at end_idx =
        # min(lengths) - 1. Indices where only some lengths produced a window
        # are averaged over those lengths only.
        expected_predictions = len(X_val) - min(lengths) + 1
        self.assertEqual(len(preds), expected_predictions)
        self.assertEqual(len(model.predict_end_indices), expected_predictions)
        self.assertEqual(model.predict_end_indices[0], min(lengths) - 1)

        preds_arr = np.asarray(preds, dtype=float)
        self.assertTrue(np.all(preds_arr >= 0.0))
        self.assertTrue(np.all(preds_arr <= 1.0))

        model.eval()

        def _logit_at(end_idx, L):
            window = torch.from_numpy(
                X_val[end_idx - L + 1 : end_idx + 1]
            ).unsqueeze(0)
            with torch.no_grad():
                return model._forward_logits(window).cpu().numpy().flatten()[0]

        # At end_idx = min(lengths) - 1, only the shortest length contributes.
        end_idx_short = min(lengths) - 1
        expected_short = 1.0 / (1.0 + np.exp(-_logit_at(end_idx_short, min(lengths))))
        idx_short = model.predict_end_indices.index(end_idx_short)
        self.assertAlmostEqual(float(preds_arr[idx_short]),
                               float(expected_short), places=5)

        # At end_idx = max(lengths) - 1, every length contributes — the
        # prediction equals sigmoid(mean(per-length linear_out)).
        end_idx_full = max(lengths) - 1
        logits_full = [_logit_at(end_idx_full, L) for L in lengths]
        expected_full = 1.0 / (1.0 + np.exp(-np.mean(logits_full)))
        idx_full = model.predict_end_indices.index(end_idx_full)
        self.assertAlmostEqual(float(preds_arr[idx_full]),
                               float(expected_full), places=5)


if __name__ == "__main__":
    unittest.main()
