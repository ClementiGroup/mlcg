import torch
import warnings
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F
from typing import Optional, List

from ..data._keys import FORCE_KEY, ENERGY_KEY
from ..data import AtomicData


class Loss(torch.nn.Module):
    """Generalized loss function class that can be
    used to combine more than one loss function

    Parameters
    ----------
    losses:
        List of torch loss modules
    weights:
        List of corresponding weights for each loss in
        the losses list. By default, each loss is weighted
        equally by 1.0.
    """

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
        """Forward pass that sums up the (weighted) contributions
        of each loss function

        Parameters
        ----------
        data:
            AtomicData instance containing the quantities needed
            for each loss

        Returns
        -------
        loss:
            The scalar losses aggreagted from each loss function
            over the entire AtomicData instance
        """

        loss = torch.zeros((len(self.losses)), device=self.weights.device)
        for ii, loss_fn in enumerate(self.losses):
            loss[ii] = loss_fn(data) * self.weights[ii]
        return loss.sum()


class ForceRMSE(_Loss):
    r"""Force root-mean-square error loss, as defined by:

    .. math::

        L\left(f,\hat{f}\right) = \sqrt{ \frac{1}{Nd}\sum_{i}^{N} \left\Vert f_i - \hat{f}_i \right\Vert ^2 }

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
        Specifies the method of reduction. See `here <https://github.com/pytorch/pytorch/blob/acb035f5130fabe258ff27049c73a15ba3a52dbd/torch/nn/modules/loss.py#L69>`_
        for more options.
    """

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
        """Forward pass through the RMSE loss

        Parameters
        ----------
        data:
            AtomicData instance containing the force keyword in the 'out' field
            as well as a base attribute

        Returns
        -------
        loss:
            Root-mean-square force Loss reduced according to the specified strategy
        """

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
        Specifies the method of reduction. See `here <https://github.com/pytorch/pytorch/blob/acb035f5130fabe258ff27049c73a15ba3a52dbd/torch/nn/modules/loss.py#L6L9>`_
        for more options.
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


class EnergyMSE(_Loss):
    r"""Energy mean square error per atom loss, as defined by:

    .. math::

        L\left(E,\hat{E}\right) = \frac{1}{N}\sum_{i}^{N}\frac{1}{n_i} \left\Vert E_i - \hat{E}_i \right\Vert ^2

    where :math:`E` are predicted energy, :math:`\hat{E}` are reference energy, :math:`n_i` is
    the number of atoms for examples/structures, :math:`N` is the total number of examples/structures.

    Parameters
    ----------
    energy_kwd:
        string to specify the energy key in an AtomicData instance
    force_kwd:
        (deprecated) old key for energy. Will be mapped to `energy_kwd` if provided.
    size_average:
        If True, the loss is normalized by the batch size
    reduce:
        If True, the loss is reduced to a scalar
    reduction:
        Specifies the method of reduction. See `here <https://github.com/pytorch/pytorch/blob/acb035f5130fabe258ff27049c73a15ba3a52dbd/torch/nn/modules/loss.py#L6L9>`_
        for more options.
    """

    def __init__(
        self,
        energy_kwd: Optional[str] = ENERGY_KEY,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        force_kwd: Optional[str] = None,
        **kwargs,
    ) -> None:
        super(EnergyMSE, self).__init__(
            size_average=size_average, reduce=reduce, reduction="none"
        )

        if force_kwd is not None:
            warnings.warn(
                "`force_kwd` is deprecated and will be removed in a future version. "
                "Please use `energy_kwd` instead.",
                UserWarning,
                stacklevel=2,
            )
            if energy_kwd is not ENERGY_KEY:
                energy_kwd = force_kwd

        self.energy_kwd = energy_kwd

    def forward(self, data: AtomicData) -> torch.Tensor:
        if self.energy_kwd not in data.out:
            raise RuntimeError(
                f"target property {self.energy_kwd} has not been computed in data.out {list(data.out.keys())}"
            )
        if self.energy_kwd not in data:
            raise RuntimeError(
                f"target property {self.energy_kwd} has no reference in data {list(data.keys())}"
            )

        return (
            F.mse_loss(
                data.out[self.energy_kwd],
                data[self.energy_kwd],
                reduction="none",
            )
            / data.n_atoms
        ).mean()


class VarianceRegularizedMSE(_Loss):
    r"""MSE loss with a variance regularization term to avoid narrow distributions:

    .. math::

        L(f,\hat{f}) = \text{MSE}(f,\hat{f}) + \lambda \max(0, \sigma_{min}^2 - \text{Var}(f))

    where :math:`f` are predicted values, :math:`\hat{f}` are reference values,
    :math:`\lambda` is the regularization strength, and :math:`\sigma_{min}^2` is
    the minimum desired variance.

    Parameters
    ----------
    target_key:
        string to specify the target key in an AtomicData instance
    min_variance:
        minimum desired variance of the predictions
    lambda_reg:
        strength of the variance regularization term
    size_average:
        If True, the loss is normalized by the batch size
    reduce:
        If True, the loss is reduced to a scalar
    reduction:
        Specifies the method of reduction
    """

    def __init__(
        self,
        target_key: str,
        min_variance: float = 1e-3,
        lambda_reg: float = 0.1,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
    ) -> None:
        super(VarianceRegularizedMSE, self).__init__(
            size_average=size_average, reduce=reduce, reduction=reduction
        )
        self.target_key = target_key
        self.min_variance = min_variance
        self.lambda_reg = lambda_reg

    def forward(self, data: AtomicData) -> torch.Tensor:
        """Forward pass through the regularized MSE loss

        Parameters
        ----------
        data:
            AtomicData instance containing the target keyword in the 'out' field
            and as a base attribute

        Returns
        -------
        loss:
            Regularized MSE loss that penalizes narrow distributions
        """
        if self.target_key not in data.out:
            raise RuntimeError(
                f"target property {self.target_key} has not been computed in data.out {list(data.out.keys())}"
            )
        if self.target_key not in data:
            raise RuntimeError(
                f"target property {self.target_key} has no reference in data {list(data.keys())}"
            )

        # Calculate standard MSE loss
        mse_loss = F.mse_loss(
            data.out[self.target_key],
            data[self.target_key],
            reduction=self.reduction,
        )

        # Calculate variance of predictions
        pred_variance = torch.var(data.out[self.target_key])

        # Add penalty if variance is too small
        variance_penalty = self.lambda_reg * torch.nn.functional.relu(
            self.min_variance - pred_variance
        )

        return mse_loss + variance_penalty


class DistributionMatchingMSE(_Loss):
    r"""MSE loss that automatically matches the target distribution's variance:

    .. math::

        L(f,\hat{f}) = \text{MSE}(f,\hat{f}) + \lambda(\sigma_f - \sigma_{\hat{f}})^2

    where :math:`f` are predicted values, :math:`\hat{f}` are reference values,
    and :math:`\sigma` represents the standard deviation.

    Parameters
    ----------
    target_key:
        string to specify the target key in an AtomicData instance
    lambda_match:
        weight for the variance matching term
    reduction:
        Specifies the method of reduction
    """

    def __init__(
        self,
        target_key: str,
        lambda_match: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super(DistributionMatchingMSE, self).__init__(reduction=reduction)
        self.target_key = target_key
        self.lambda_match = lambda_match

    def forward(self, data: AtomicData) -> torch.Tensor:
        if self.target_key not in data.out:
            raise RuntimeError(
                f"target property {self.target_key} has not been computed in data.out {list(data.out.keys())}"
            )
        if self.target_key not in data:
            raise RuntimeError(
                f"target property {self.target_key} has no reference in data {list(data.keys())}"
            )

        # Calculate standard MSE loss
        mse_loss = F.mse_loss(
            data.out[self.target_key],
            data[self.target_key],
            reduction=self.reduction,
        )

        # Calculate standard deviations
        pred_std = torch.std(data.out[self.target_key])
        target_std = torch.std(data[self.target_key])

        # Add penalty for std mismatch
        std_penalty = self.lambda_match * (pred_std - target_std)**2

        return mse_loss + std_penalty