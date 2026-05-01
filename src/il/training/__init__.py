"""Training entrypoints and evaluators.

Keep imports lazy to avoid import-time dependency failures when Hydra resolves a
single target (e.g. `il.training.mlp_train.MLPTrainer`).
"""

from .trainer import TrainerBase

__all__ = ["MLPTrainer", "MLPCloseEvaluator", "DFTrainer", "TrainerBase"]


def __getattr__(name: str):
    if name == "MLPTrainer":
        from .mlp_train import MLPTrainer

        return MLPTrainer
    if name == "MLPCloseEvaluator":
        from .mlp_close_eval import MLPCloseEvaluator

        return MLPCloseEvaluator
    if name == "DFTrainer":
        from .df_train import DFTrainer

        return DFTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
