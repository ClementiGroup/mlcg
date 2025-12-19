from torch.optim.lr_scheduler import _LRScheduler
import warnings


class CustomStepLR(_LRScheduler):
    """Custom Learning rate that allows to apply different
    gamma and step size to different parameter groups defined in
    the optimizer."""

    def __init__(
        self, optimizer, step_sizes=[], gammas=[], last_epoch=-1, verbose=False
    ):
        # Ensure step_sizes and gammas match the number of param groups
        if not len(step_sizes) == len(gammas):
            raise ValueError("The length of step_sizes and gammas must match.")

        if not len(gammas) == len(optimizer.param_groups):
            warnings.warn(
                "The length of step_sizes and gammas must match the number of parameter groups.",
                DeprecationWarning,
            )

        self.step_sizes = step_sizes
        self.gammas = gammas
        super(CustomStepLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """Compute learning rate using step decay schedule with different parameters for each param group."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )
        new_lrs = []
        for i, (base_lr, step_size, gamma) in enumerate(
            zip(self.base_lrs, self.step_sizes, self.gammas)
        ):
            if (self.last_epoch == 0) or (self.last_epoch % step_size != 0):
                # Not a step epoch for this param group
                new_lrs.append(self.optimizer.param_groups[i]["lr"])
            else:
                # Step epoch for this param group
                new_lrs.append(
                    base_lr * gamma ** (self.last_epoch // step_size)
                )

        return new_lrs
