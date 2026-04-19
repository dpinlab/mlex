import unittest

import numpy as np


def _make_data(n_samples=120, n_features=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y = rng.integers(0, 2, size=n_samples).astype(np.float32).reshape(-1, 1)
    return X, y


def _make_base_model(epoch_observers, epochs=3, patience=10, seed=42):
    import torch
    from mlex.models.base_components.lstm_base_model import LSTMBaseModel

    X_val, y_val = _make_data(n_samples=60, seed=12)
    return LSTMBaseModel(
        validation_data=(X_val, y_val),
        input_size=3,
        hidden_size=4,
        num_layers=1,
        output_size=1,
        seq_length=5,
        batch_size=8,
        shuffle_dataloader=True,
        epochs=epochs,
        patience=patience,
        group_index=None,
        random_seed=seed,
        device=torch.device("cpu"),
        epoch_observers=epoch_observers,
    ), X_val, y_val


class AUCROCObserverTests(unittest.TestCase):
    def test_val_observer_records_auc_per_epoch(self):
        from mlex.observers.observers import AUCROCObserver

        obs = AUCROCObserver(name="val")
        model, _, _ = _make_base_model([obs], epochs=3, patience=10)

        X_train, y_train = _make_data(n_samples=120, seed=11)
        model.fit(X_train, y_train)

        self.assertEqual(len(obs.auc_history), len(model.history["epoch"]))
        self.assertGreater(len(obs.auc_history), 0)
        for entry in obs.auc_history:
            self.assertIn("epoch", entry)
            self.assertIn("auc", entry)
            self.assertTrue(np.isnan(entry["auc"]) or 0.0 <= entry["auc"] <= 1.0)

    def test_held_out_observer_uses_its_own_data(self):
        from mlex.observers.observers import AUCROCObserver

        X_test, y_test = _make_data(n_samples=80, seed=99)
        val_obs = AUCROCObserver(name="val")
        test_obs = AUCROCObserver(name="test", data=(X_test, y_test))
        model, _, _ = _make_base_model([val_obs, test_obs], epochs=2)

        X_train, y_train = _make_data(n_samples=120, seed=11)
        model.fit(X_train, y_train)

        self.assertEqual(len(test_obs.auc_history), len(val_obs.auc_history))
        self.assertGreater(len(test_obs.auc_history), 0)
        # Held-out set is different from validation, so AUCs should differ at
        # least once across epochs.
        val_aucs = [e["auc"] for e in val_obs.auc_history]
        test_aucs = [e["auc"] for e in test_obs.auc_history]
        self.assertNotEqual(val_aucs, test_aucs)

    def test_no_observers_is_backwards_compatible(self):
        model, _, _ = _make_base_model(None, epochs=1)
        X_train, y_train = _make_data(n_samples=100, seed=1)
        model.fit(X_train, y_train)
        self.assertEqual(model.epoch_observers, [])
        self.assertGreater(len(model.history["epoch"]), 0)

    def test_custom_observer_receives_expected_context(self):
        from mlex.observers.observers import EpochObserver

        captured = []

        class RecordingObserver(EpochObserver):
            def on_epoch_end(self, context):
                captured.append({k: v for k, v in context.items() if k != "model"})

        model, _, _ = _make_base_model([RecordingObserver()], epochs=2)
        X_train, y_train = _make_data(n_samples=100, seed=5)
        model.fit(X_train, y_train)

        self.assertGreater(len(captured), 0)
        ctx = captured[0]
        self.assertIn("epoch", ctx)
        self.assertIn("train_loss", ctx)
        self.assertIn("val_loss", ctx)
        self.assertIn("val_outputs", ctx)
        self.assertIn("val_targets", ctx)
        self.assertEqual(len(ctx["val_outputs"]), len(ctx["val_targets"]))


if __name__ == "__main__":
    unittest.main()
