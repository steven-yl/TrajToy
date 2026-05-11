import torch
import torch.optim.lr_scheduler


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    lr_scheduler: str,
    cosine_t_max: int,
    cosine_eta_min: float,
    warmup_steps: int,
    warmup_start_factor: float,
    lr_step_size: int,
    lr_gamma: float,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """按名称构造学习率调度器；步数均指 TrainFlow 每次 ``optimizer.step()`` 对应的 scheduler step。"""
    kind = lr_scheduler.strip().lower()
    if kind in ("none", "null", ""):
        return None
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_t_max, eta_min=cosine_eta_min
        )
    if kind == "cosine_warmup":
        ws = max(0, int(warmup_steps))
        if ws == 0:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cosine_t_max, eta_min=cosine_eta_min
            )
        warm = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=float(warmup_start_factor),
            end_factor=1.0,
            total_iters=ws,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_t_max, eta_min=cosine_eta_min
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warm, cosine], milestones=[ws]
        )
    if kind == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(lr_step_size), gamma=float(lr_gamma)
        )
    raise ValueError(
        f"Unknown lr_scheduler {lr_scheduler!r}; "
        "expected none, cosine, cosine_warmup, step."
    )
