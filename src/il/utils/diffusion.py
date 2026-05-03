import math
from itertools import pairwise

import torch
import numpy as np
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from types import SimpleNamespace
from typing import Optional, Dict, Tuple

class Schedule:
    '''Diffusion noise schedules parameterized by sigma'''
    def __init__(self, sigmas: torch.FloatTensor):
        self.sigmas = sigmas

    def __getitem__(self, i) -> torch.FloatTensor:
        return self.sigmas[i]

    def __len__(self) -> int:
        return len(self.sigmas)

    def sample_sigmas(self, steps: int) -> torch.FloatTensor:
        '''Called during sampling to get a decreasing sigma schedule with a
        specified number of sampling steps:
          - Spacing is "trailing" as in Table 2 of https://arxiv.org/abs/2305.08891
          - Includes initial and final sigmas
            i.e. len(schedule.sample_sigmas(steps)) == steps + 1
        '''
        indices = list((len(self) * (1 - np.arange(0, steps)/steps))
                       .round().astype(np.int64) - 1)
        return self[indices + [0]]

    def sample_batch(self, x0: torch.FloatTensor) -> torch.FloatTensor:
        '''Called during training to get a batch of randomly sampled sigma values
        '''
        batchsize = x0.shape[0]
        return self[torch.randint(len(self), (batchsize,))].to(x0)

def sigmas_from_betas(betas: torch.FloatTensor):
    return (1/torch.cumprod(1.0 - betas, dim=0) - 1).sqrt()

# Simple log-linear schedule works for training many diffusion models
class ScheduleLogLinear(Schedule):
    def __init__(self, N: int, sigma_min: float=0.02, sigma_max: float=10):
        super().__init__(torch.logspace(math.log10(sigma_min), math.log10(sigma_max), N))

# Default parameters recover schedule used in most diffusion models
class ScheduleDDPM(Schedule):
    def __init__(self, N: int=1000, beta_start: float=0.0001, beta_end: float=0.02):
        super().__init__(sigmas_from_betas(torch.linspace(beta_start, beta_end, N)))

# Default parameters recover schedule used in most latent diffusion models, e.g. Stable diffusion
class ScheduleLDM(Schedule):
    def __init__(self, N: int=1000, beta_start: float=0.00085, beta_end: float=0.012):
        super().__init__(sigmas_from_betas(torch.linspace(beta_start**0.5, beta_end**0.5, N)**2))

# Sigmoid schedule used in GeoDiff
class ScheduleSigmoid(Schedule):
    def __init__(self, N: int=1000, beta_start: float=0.0001, beta_end: float=0.02):
        betas = torch.sigmoid(torch.linspace(-6, 6, N)) * (beta_end - beta_start) + beta_start
        super().__init__(sigmas_from_betas(betas))

# Cosine schedule used in Nichol and Dhariwal 2021
class ScheduleCosine(Schedule):
    def __init__(self, N: int=1000, beta_start: float=0.0001, beta_end: float=0.02, max_beta: float=0.999):
        alpha_bar = lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
        betas = [min(1 - alpha_bar((i+1)/N)/alpha_bar(i/N), max_beta)
                 for i in range(N)]
        super().__init__(sigmas_from_betas(torch.tensor(betas, dtype=torch.float32)))

# Given a batch of data
#   x0   : Either a data tensor or a tuple of (data, labels)
# Returns
#   eps  : i.i.d. normal with same shape as x0
#   sigma: uniformly sampled from schedule, with shape Bx1x..x1 for broadcasting
def _batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move tensor values in a dataloader batch dict onto ``device``."""
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def generate_train_sample(x0: torch.FloatTensor,
                          schedule: Schedule):
    sigma = schedule.sample_batch(x0)
    while len(sigma.shape) < len(x0.shape):
        sigma = sigma.unsqueeze(-1)
    eps = torch.randn_like(x0)
    return x0, sigma, eps

# Model objects
# Always called with (x, sigma):
#   If x.shape == [B, D1, ..., Dk], sigma.shape == [] or [B, 1, ..., 1].
#   If sigma.shape == [], model will be called with the same sigma for each x0
#   Otherwise, x[i] will be paired with sigma[i] when calling model
# Have a `rand_input` method for generating random xt during sampling

# def training_loop(loader      : DataLoader,
#                   model       : nn.Module,
#                   schedule    : Schedule,
#                   accelerator : Optional[Accelerator] = None,
#                   epochs      : int = 100,
#                   lr          : float = 1e-3,
#                   conditional : bool = False):
#     accelerator = accelerator or Accelerator()
#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
#     model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
#     for _ in (pbar := tqdm(range(epochs))):
#         for x0 in loader:
#             # Support both plain tensor batches and dict batches from trajectory dataset.
#             if isinstance(x0, dict):
#                 x = x0["future"]                         # (B, F, D)
#                 future_mask = x0.get("future_mask", None)  # (B, F), True means valid
#             else:
#                 x = x0
#                 future_mask = None

#             model.train()
#             optimizer.zero_grad()
#             x, sigma, eps, cond = generate_train_sample(x, schedule, conditional)

#             if future_mask is None:
#                 loss = model.get_loss(x, sigma, eps, cond=cond)
#             else:
#                 # Ignore padded future steps when computing diffusion MSE.
#                 mask = future_mask.to(x.device).to(x.dtype).unsqueeze(-1)  # (B, F, 1)
#                 mask = mask.expand_as(eps)                                  # (B, F, D)
#                 pred_eps = model.predict_eps(x + sigma * eps, sigma, cond=cond)
#                 sq_err = (pred_eps - eps) ** 2
#                 loss = (sq_err * mask).sum() / mask.sum().clamp_min(1.0)

#             yield SimpleNamespace(**locals()) # For extracting training statistics
#             accelerator.backward(loss)
#             optimizer.step()


def training_loop(loader      : DataLoader,
                  model       : nn.Module,
                  schedule    : Schedule,
                  epochs      : int = 100,
                  lr          : float = 1e-3,
                  conditional : bool = False):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    device = next(model.parameters()).device
    for epoch in (pbar := tqdm(range(epochs), desc="Epochs")):
        for batch in loader:
            batch = _batch_to_device(batch, device)
            x = batch["future"]
            future_mask = batch["future_mask"]
            cond = batch

            model.train()
            optimizer.zero_grad()
            x, sigma, eps = generate_train_sample(x, schedule)

            if future_mask is None:
                loss = model.get_loss(x, sigma, eps, cond=cond)
            else:
                # Ignore padded future steps when computing diffusion MSE.
                mask = future_mask.to(x.device).to(x.dtype).unsqueeze(-1)  # (B, F, 1)
                mask = mask.expand_as(eps)                                  # (B, F, D)
                pred_eps = model.predict_eps(x + sigma * eps, sigma, cond=cond)
                sq_err = (pred_eps - eps) ** 2
                loss = (sq_err * mask).sum() / mask.sum().clamp_min(1.0)

            loss.backward()
            optimizer.step()
            yield SimpleNamespace(**locals())  # after step: OK for EMA / checkpoint timing


# Generalizes most commonly-used samplers:
#   DDPM       : gam=1, mu=0.5
#   DDIM       : gam=1, mu=0
#   Accelerated: gam=2, mu=0
@torch.no_grad()
def samples(model      : nn.Module,
            sigmas     : torch.FloatTensor, # Iterable with N+1 values for N sampling steps
            gam        : float = 1.,        # Suggested to use gam >= 1
            mu         : float = 0.,        # Requires mu in [0, 1)
            cfg_scale  : int = 0.,          # 0 means no classifier-free guidance
            batchsize  : int = 1,
            xt         : Optional[torch.FloatTensor] = None,
            cond       : Optional[torch.Tensor] = None):
    model.eval()
    device = next(model.parameters()).device
    sigmas = sigmas.to(device)
    xt = model.rand_input(batchsize) * sigmas[0] if xt is None else xt
    if cond is not None:
        assert cond.shape[0] == xt.shape[0], 'cond must have same shape as x!'
        cond = cond.to(xt.device)
    eps = None
    for i, (sig, sig_prev) in enumerate(pairwise(sigmas)):
        eps_prev, eps = eps, model.predict_eps_cfg(xt, sig.to(xt), cond, cfg_scale)
        eps_av = eps * gam + eps_prev * (1-gam)  if i > 0 else eps
        sig_p = (sig_prev/sig**mu)**(1/(1-mu)) # sig_prev == sig**mu sig_p**(1-mu)
        eta = (sig_prev**2 - sig_p**2).sqrt()
        xt = xt - (sig - sig_p) * eps_av + eta * model.rand_input(xt.shape[0]).to(xt)
        yield xt
