"""Training loop with early stopping and metric-driven checkpointing."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import TrainConfig
from .evaluation import MantelResult


# --------------------------------------------------------------------------- #
# Device selection (MPS first, per the target MacBook Pro)
# --------------------------------------------------------------------------- #
def resolve_device(pref: str) -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_mse: float
    mantel: Optional[MantelResult] = None
    per_feature_r: Optional[float] = None


@dataclass
class TrainingHistory:
    records: List[EpochRecord] = field(default_factory=list)
    best_epoch: int = -1
    best_score: float = float("nan")


MantelEvalFn = Callable[[nn.Module], MantelResult]
PerFeatureFn = Callable[[nn.Module], float]


class Trainer:
    """Trains a single :class:`ImputationTransformer`.

    Model selection is driven by ``cfg.selection_metric``:
      * "mantel"  -> keep the checkpoint with the highest Mantel statistic.
      * "val_mse" -> keep the checkpoint with the lowest validation MSE.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: TrainConfig,
        name: str = "model",
        mantel_eval_fn: Optional[MantelEvalFn] = None,
        per_feature_fn: Optional[PerFeatureFn] = None,
    ):
        self.cfg = cfg
        self.name = name
        self.device = resolve_device(cfg.device)
        self.model = model.to(self.device)
        self.mantel_eval_fn = mantel_eval_fn
        self.per_feature_fn = per_feature_fn
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.history = TrainingHistory()
        self._best_state: Optional[Dict[str, torch.Tensor]] = None

        if cfg.selection_metric == "mantel" and mantel_eval_fn is None:
            raise ValueError(
                "selection_metric='mantel' requires a mantel_eval_fn callback."
            )
        if cfg.selection_metric == "per_feature_r" and per_feature_fn is None:
            raise ValueError(
                "selection_metric='per_feature_r' requires a per_feature_fn callback."
            )

    # ----- single epoch ----------------------------------------------------- #
    def _train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(xb)
            loss = self.loss_fn(pred, yb)
            loss.backward()
            if self.cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        return total / max(n, 1)

    @torch.no_grad()
    def _val_mse(self, loader: DataLoader) -> float:
        self.model.eval()
        total, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            pred = self.model(xb)
            total += self.loss_fn(pred, yb).item() * xb.size(0)
            n += xb.size(0)
        return total / max(n, 1)

    # ----- selection logic -------------------------------------------------- #
    def _score(
        self,
        val_mse: float,
        mantel: Optional[MantelResult],
        per_feature_r: Optional[float],
    ) -> float:
        # Always expressed so that HIGHER is better.
        if self.cfg.selection_metric == "mantel":
            return mantel.statistic if mantel is not None else float("-inf")
        if self.cfg.selection_metric == "per_feature_r":
            if per_feature_r is None or per_feature_r != per_feature_r:  # None/NaN
                return float("-inf")
            return per_feature_r
        return -val_mse

    # ----- full training ---------------------------------------------------- #
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> TrainingHistory:
        set_seed(self.cfg.seed)
        best_score = float("-inf")
        epochs_no_improve = 0

        for epoch in range(1, self.cfg.max_epochs + 1):
            train_loss = self._train_one_epoch(train_loader)
            val_mse = self._val_mse(val_loader)
            mantel = self.mantel_eval_fn(self.model) if self.mantel_eval_fn else None
            per_feature_r = (
                self.per_feature_fn(self.model) if self.per_feature_fn else None
            )

            rec = EpochRecord(epoch, train_loss, val_mse, mantel, per_feature_r)
            self.history.records.append(rec)
            pstr = (
                f" | per_feat_r={per_feature_r:+.4f}"
                if per_feature_r is not None
                else ""
            )
            mstr = f" | {mantel}" if mantel is not None else ""
            star = "*" if self.cfg.selection_metric == "per_feature_r" else ""
            sstar = "*" if self.cfg.selection_metric == "val_mse" else ""
            print(
                f"[{self.name}] epoch {epoch:3d} "
                f"train_mse={train_loss:.5f} val_mse={val_mse:.5f}{sstar}"
                f"{pstr}{star}{mstr}"
            )

            score = self._score(val_mse, mantel, per_feature_r)
            improved = score > best_score
            if improved:
                best_score = score
                self._best_state = copy.deepcopy(self.model.state_dict())
                self.history.best_epoch = epoch
                self.history.best_score = best_score
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if (
                epoch >= self.cfg.min_epochs
                and epochs_no_improve >= self.cfg.patience
            ):
                print(
                    f"[{self.name}] early stopping at epoch {epoch} "
                    f"(best epoch {self.history.best_epoch})."
                )
                break

        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
        return self.history

    # ----- persistence ------------------------------------------------------ #
    def save(self, path: Optional[str] = None) -> str:
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        path = path or os.path.join(self.cfg.checkpoint_dir, f"{self.name}.pt")
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "best_epoch": self.history.best_epoch,
                "best_score": self.history.best_score,
                "selection_metric": self.cfg.selection_metric,
            },
            path,
        )
        return path

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        xb = torch.tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        return self.model(xb).cpu().numpy()
