import torch
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F
from typing import Optional, List

from ..data._keys import FORCE_KEY
from ..data import AtomicData


class Loss(torch.nn.Module):
    def __init__(
        self, losses: List[_Loss], weights: Optional[List[float]] = None
    ) -> None:
        super(Loss, self).__init__()
        if weights is None:
            weights = torch.ones((len(losses)))
        else:
            weights = torch.tensor(weights)
        assert len(weights) == len(losses)
        self.register_buffer("weights", weights)
        self.losses = losses

    def forward(self, data: AtomicData) -> torch.Tensor:
        loss = torch.zeros((len(self.losses)))
        for ii, loss_fn in enumerate(self.losses):
            loss[ii] = loss_fn(data) * self.weights[ii]
        return loss.sum()


class ForceRMSE(_Loss):
    def __init__(
        self,
        force_kwd: str = FORCE_KEY,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
    ) -> None:

        super(ForceRMSE, self).__init__(
            size_average=size_average, reduce=reduce, reduction=reduction
        )

        self.force_kwd = force_kwd

    def forward(self, data: AtomicData) -> torch.Tensor:

        if self.force_kwd not in data.out:
            raise RuntimeError(
                f"target property {self.force_kwd} has not been computed in data.out {list(data.out.keys())}"
            )
        if self.force_kwd not in data:
            raise RuntimeError(
                f"target property {self.force_kwd} has no reference in data {list(data.keys())}"
            )

        return torch.sqrt(
            F.mse_loss(
                data.out[self.force_kwd],
                data[self.force_kwd],
                reduction=self.reduction,
            )
        )


class ForceMSE(_Loss):
    r"""Force mean square error loss, as defined by:
    .. math::
        L\left(f,\hat{f}\right) = \frac{1}{Nd}\sum_{i}^{N} \left\Vert f_i - \hat{f}_i \right\Vert ^2
    where :math:`f` are predicted forces, :math:`\hat{f}` are reference forces, :math:`N` is
    the number of examples/structures, and :math:`d` is the real space dimensionality
    (eg, :math:`d=3` for proteins)
    Parameters
    ----------
    force_kwd:
        string to specify the force key in an AtomicData instance
    size_average:
        If True, the loss is normalized by the batch size
    reduce:
        If True, the loss is reduced to a scalar
    reduction:
        Specifies the method of reduction. See
        https://github.com/pytorch/pytorch/blob/acb035f5130fabe258ff27049c73a15ba3a52dbd/torch/nn/modules/loss.py#L69
    """

    def __init__(
        self,
        force_kwd: str = FORCE_KEY,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
    ) -> None:

        super(ForceMSE, self).__init__(
            size_average=size_average, reduce=reduce, reduction=reduction
        )

        self.force_kwd = force_kwd

    def forward(self, data: AtomicData) -> torch.Tensor:

        if self.force_kwd not in data.out:
            raise RuntimeError(
                f"target property {self.force_kwd} has not been computed in data.out {list(data.out.keys())}"
            )
        if self.force_kwd not in data:
            raise RuntimeError(
                f"target property {self.force_kwd} has no reference in data {list(data.keys())}"
            )

        return F.mse_loss(
            data.out[self.force_kwd],
            data[self.force_kwd],
            reduction=self.reduction,
        )
