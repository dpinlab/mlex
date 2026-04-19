import numpy as np
from sklearn.metrics import roc_auc_score


class EpochObserver:
    """Base class for per-epoch observers attached to recurrent models.

    Subclasses override ``on_epoch_end(context)``. The ``context`` dict contains:
        - epoch:         int, 1-indexed
        - model:         the base model (torch.nn.Module) — use for predict_proba
        - train_loss:    float, averaged training BCE loss for the epoch
        - val_loss:      float, averaged validation BCE loss for the epoch
        - val_outputs:   np.ndarray, concatenated validation sigmoid outputs
        - val_targets:   np.ndarray, concatenated validation targets
    """

    def on_epoch_end(self, context):
        raise NotImplementedError


class AUCROCObserver(EpochObserver):
    """Records AUC-ROC per epoch on validation (default) or a held-out set.

    If ``data`` is None, uses cached validation outputs/targets from the
    training loop — zero extra cost. If ``data=(X, y)`` is provided (already
    preprocessed, same shape as the base model's ``validation_data``), runs
    a forward pass each epoch and computes AUC against the sequence-end
    targets yielded by the base model's dataloader.
    """

    def __init__(self, name="val", data=None):
        self.name = name
        self.data = data
        self.auc_history = []

    def on_epoch_end(self, context):
        epoch = context["epoch"]

        if self.data is None:
            y_true = np.asarray(context["val_targets"]).flatten()
            y_score = np.asarray(context["val_outputs"]).flatten()
        else:
            y_true, y_score = self._score_held_out(context["model"])

        if len(np.unique(y_true)) < 2:
            auc = float("nan")
        else:
            auc = float(roc_auc_score(y_true, y_score))
        self.auc_history.append({"epoch": epoch, "auc": auc})

    def _score_held_out(self, model):
        import torch

        X, y = self.data
        y_arr = y.values if hasattr(y, "values") else y
        loader = model._create_dataloader(X, y_arr, shuffle_dataloader=False)

        was_training = model.training
        model.eval()
        outputs_all, targets_all = [], []
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(model.device)
                probs = model.predict_proba(batch_x)
                outputs_all.append(np.asarray(probs).flatten())
                targets_all.append(batch_y.cpu().numpy().flatten())
        if was_training:
            model.train()

        return np.concatenate(targets_all), np.concatenate(outputs_all)
