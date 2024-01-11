from typing import Any, Dict

import torch
from torch import optim


class ReduceLROnPlateauWrapper(optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(
        self,
        scheduler: optim.lr_scheduler.LRScheduler,
        optimiser: optim.Optimizer,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
    ) -> None:
        self.scheduler = scheduler
        self.optimizer = optimiser

        self.factor = factor
        self.patience = patience

        assert mode in ["min", "max"]

        self.comparator = torch.lt if mode == "min" else torch.gt
        self.best_score = torch.inf if mode == "min" else -torch.inf
        self.misses = 0
        self.modifier = 1

    def state_dict(self) -> Dict[str, Any]:
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ("optimiser", "scheduler")}
        state_dict["scheduler"] = self.scheduler.state_dict()

        return state_dict

    def load_state_dict(self, state_dict) -> None:
        self.__dict__.update(state_dict)

        scheduler = state_dict.pop("scheduler")
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict["scheduler"] = scheduler

        self.scheduler.load_state_dict(scheduler)

    def step(self, score: float) -> None:
        self.scheduler.step()

        if self.is_better(score):
            self.misses = 0
            self.best_score = score
        else:
            self.misses += 1
            if self.misses >= self.patience:
                self.misses = 0
                self.modifier = self.modifier * self.factor

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * self.modifier

    def is_better(self, score: torch.Tensor) -> bool:
        return self.comparator(score, self.best_score)
