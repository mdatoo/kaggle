"""ReduceLROnPlateau PyTorch scheduler."""

from typing import Any, Dict, Optional

from torch import optim


class ReduceLROnPlateauWrapper(optim.lr_scheduler.ReduceLROnPlateau):
    """ReduceLROnPlateau PyTorch scheduler.

    Reimplementation of PyTorch's ReduceLROnPlateau scheduler that accepts a scheduler to wrap.
    This exists as ReduceLROnPlateau does not play nice with SequentialLR and CompositeScheduler.
    See:
    - https://github.com/pytorch/pytorch/issues/68978
    - https://github.com/pytorch/pytorch/issues/110761
    """

    def __init__(  # pylint: disable=super-init-not-called
        self,
        scheduler: optim.lr_scheduler.LRScheduler,
        optimiser: optim.Optimizer,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
    ) -> None:
        """Initialises object.

        Args:
            scheduler: LR scheduler to wrap
            optimiser: Optimiser to modify
            mode: Whether lower or higher best
            factor: LR multiplier
            patience: Epochs with no improvement to wait for before decreasing LR
        """
        self.scheduler = scheduler
        self.optimizer = optimiser

        self.factor = factor
        self.patience = patience

        assert mode in ["min", "max"]

        self.mode = mode
        self.best_score: Optional[float] = None
        self.misses = 0
        self.modifier = 1.0

    def state_dict(self) -> Dict[str, Any]:
        """Save state of scheduler."""

        state_dict = {key: value for key, value in self.__dict__.items() if key not in ("optimiser", "scheduler")}
        state_dict["scheduler"] = self.scheduler.state_dict()

        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state of scheduler.

        Args:
            state_dict: Data to load
        """
        self.__dict__.update(state_dict)

        scheduler = state_dict.pop("scheduler")
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict["scheduler"] = scheduler

        self.scheduler.load_state_dict(scheduler)

    def step(self, metrics: Any, _: Optional[int] = None) -> None:
        """Run scheduler iteration.

        Args:
            metrics: Value of tracked metric
        """
        score = float(metrics)

        self.scheduler.step()

        if self.is_best(score):
            self.misses = 0
            self.best_score = score
        else:
            self.misses += 1
            if self.misses >= self.patience:
                self.misses = 0
                self.modifier = self.modifier * self.factor

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * self.modifier

    def is_best(self, score: float) -> bool:
        """Is given score the best score so far

        Args:
            score: Score to compare
        """
        if not self.best_score:
            return True
        if self.mode == "min":
            return score < self.best_score
        return score > self.best_score
