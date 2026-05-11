import torch
from typing import Union, List

from .base import _RadialBasis


class SymmetricTensor(torch.nn.Module):
    r"""
    Implementation of a symmetric tensor via upper triangular parameters.
    Upper triangular parameters are initialized to `init_val`.

    Parameters:
    ----------
    N : int
        Maximum value of the selection index.
    size : int
        Size of the tensor.
    init_val : float, optional
        Initial value for the upper triangular parameters (default is 1.0).

    """

    def __init__(self, N, size: int, init_val=1.0):
        super().__init__()
        self.N = N
        self.size = size
        self.w = torch.nn.Parameter(
            torch.full((N * (N + 1) // 2, size), init_val)
        )

    def forward(self, i, j):
        r"""
        Given indices i,j return the corresponding symmetric tensor entry
        of shape (i.shape, size).
        Here i,j are indeces tensors of same shape.
        """
        i_, j_ = torch.minimum(i, j), torch.maximum(i, j)
        k = ((2 * self.N - i_ + 1) * i_) // 2 + (j_ - i_)
        return self.w[k]


class RegularizedBasis(torch.nn.Module):
    r"""
    Utility class for applying regularization to provided radial basis function.

    Parameters:
    ----------
    basis_function: _RadialBasis
        The radial basis function to be regularized. This should be an instance of a class
        that inherits from _RadialBasis and have cutoff and num_rbf attributes.
    types: int
        The number of atom types. This is used to generate separate regularization parameters
        for each atom type couple.
    n_basis_set: int
        The number of basis to output for each indeces couple.
        This is useful for example when using different interaction blocks
        within the same model and a different basis set is needed for each block.
    independent_regularizations: bool = False
        If set to True, independent parameters are used for each basis set.
    init_val: Union[float, List] = 1.0
        Initial value(s) for the regularization parameters. If independent_regularizations is True,
        this can be a list of floats with length equal to n_basis_set, while if a single float is provided,
        it will be used for all basis sets. If independent_regularizations is False,
        this should be a single float value.

    """

    def __init__(
        self,
        basis_function: _RadialBasis,
        types: int,
        n_basis_set: int,
        independent_regularizations: bool = False,
        init_val: Union[float, List] = 1.0,
    ):
        super().__init__()
        self.basis_function = basis_function
        self.types = types
        self.n_basis_set = n_basis_set
        self.independent_regularizations = independent_regularizations
        self.num_rbf = basis_function.num_rbf
        self.cutoff = basis_function.cutoff

        if self.independent_regularizations:
            init_val = (
                init_val
                if isinstance(init_val, list)
                else [init_val] * n_basis_set
            )
            self.regularization = torch.nn.ModuleList(
                [
                    SymmetricTensor(
                        N=types,
                        size=self.num_rbf,
                        init_val=init_val[idx],
                    )
                    for idx in range(n_basis_set)
                ]
            )
            self.get_regularization_parameters = (
                self._get_independent_regularization_parameters
            )
            self._compute_regularization_params_fn = (
                self._compute_independent_regularization_params
            )

        else:
            if isinstance(init_val, list):
                if len(init_val) != 1:
                    raise ValueError(
                        "When using shared regularization parameters, "
                        "init_val should be a single float value."
                    )
                init_val = init_val[0]
            self.regularization = SymmetricTensor(
                N=types,
                size=self.num_rbf,
                init_val=init_val,
            )
            self.get_regularization_parameters = (
                self._get_shared_regularization_parameters
            )
            self._compute_regularization_params_fn = (
                self._compute_shared_regularization_params
            )

    def forward(
        self,
        distances: torch.Tensor,
        type_i: torch.Tensor,
        type_j: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Apply the radial basis function and the regularization.

        Parameters:
        ----------
        distances : torch.Tensor
            Tensor of shape (num_edges,) containing the distances between atom pairs.
        type_i : torch.Tensor
            Tensor of shape (num_edges,) containing the types of the first atom in each pair.
        type_j : torch.Tensor
            Tensor of shape (num_edges,) containing the types of the second atom in each pair.

        Returns:
        -------
        torch.Tensor
            Tensor of shape (n_basis_set, num_edges, num_rbf) containing the regularized radial basis function values.

        """
        rbf_values = self.basis_function(distances)

        reg_params = self._compute_regularization_params_fn(type_i, type_j)
        return rbf_values.unsqueeze(0) * reg_params

    def _compute_independent_regularization_params(self, type_i, type_j):
        r"""
        Compute the regualrization parameters for the case were regularization
        parameters are specific for different basis sets.
        """
        return torch.clamp(
            torch.stack(
                [r(type_i, type_j) for r in self.regularization], dim=0
            ),
            min=0.0,
            max=1.0,
        )

    def _compute_shared_regularization_params(self, type_i, type_j):
        r"""
        Compute the regualrization parameters for the case were regularization
        parameters are shared between different basis sets.
        """
        return torch.clamp(
            self.regularization(type_i, type_j)
            .unsqueeze(0)
            .expand(self.n_basis_set, -1, -1),
            min=0.0,
            max=1.0,
        )

    def _get_independent_regularization_parameters(self) -> torch.Tensor:
        r"""
        Get the current regularization parameters for the case were regularization
        parameters are specific for different basis sets.

        Returns:
        -------
        torch.Tensor
            A tensor of shape (independent_regularizations, types*(types+1)/2, num_rbf) containing
            the regularization parameters for each independent regularization.
        """

        return torch.stack(
            [
                regularization_tensor.w
                for regularization_tensor in self.regularization
            ]
        )

    def _get_shared_regularization_parameters(self) -> torch.Tensor:
        r"""
        Get the current regularization parameters for the case were regularization
        parameters are shared across different basis sets.

        Returns:
        -------
        torch.Tensor
            A tensor of shape (types*(types+1)/2, num_rbf) containing
            the regularization parameters for each independent regularization.
        """

        return self.regularization.w

    def reset_parameters(self):
        pass  # Added for compatibility: reset is done in the initialization

    def plot(self, i, j, layer=0, ax=None, **kwargs):
        r"""Method for quickly visualizing a specific basis. This is useful for
        inspecting the distance coverage of basis functions for non-default lower
        and upper cutoffs.

        Parameters:
        ----------
        distances : torch.Tensor
            Tensor of shape (num_edges,) containing the distances between atom pairs.
        i : torch.Tensor
            Tensor of shape (num_edges,) containing the types of the first atom in each pair.
        j : torch.Tensor
            Tensor of shape (num_edges,) containing the types of the second atom in each pair.

        """

        import matplotlib.pyplot as plt

        distances = torch.linspace(
            self.cutoff.cutoff_lower - 1,
            self.cutoff.cutoff_upper + 1,
            1000,
        )
        i, j = torch.as_tensor(i), torch.as_tensor(j)
        expanded_distances = self(distances, i, j)

        if ax is None:
            fig, ax = plt.subplots()

        for _i in range(expanded_distances.shape[-1]):
            ax.plot(
                distances.numpy(),
                expanded_distances[layer, :, _i].detach().numpy(),
                **kwargs,
            )

        return ax
