import torch
from typing import Optional, List, Final, Callable, Union

from ._module_init import init_xavier_uniform


class MLP(torch.nn.Module):
    """Multilayer Perceptron for regression.

    Parameters
    ----------
    layer_widths:
        the width of outputs after passing through
        each linear layer. Eg, for an width specification:
        ..code-block::python
            [10, 10, 1]
        an n-dimensional example will be transformed
        to a 10-dimensional feature, then a second
        10-dimensional feature, then finally to a
        1-dimensional scalar feature.
    activation_func:
        The (non-linear) activation function to apply
        to the output of each linear transformation,
        with the exception of the final output, which
        is simply the output of a linear transformation
        with no bias
    last_bias: bool
        add a bias term to the last layer of the MLP
    """

    def __init__(
        self,
        layer_widths: List[int] = None,
        activation_func: torch.nn.Module = torch.torch.nn.Tanh(),
        last_bias: bool = True,
    ):
        super(MLP, self).__init__()
        if layer_widths is None:
            layer_widths = [10, 10, 1]
        layers = []
        for w_in, w_out in zip(layer_widths[:-2], layer_widths[1:-1]):
            layers.append(torch.nn.Linear(w_in, w_out, bias=True))
            layers.append(activation_func)
        # last layer without activation function and bias
        layers.append(
            torch.nn.Linear(layer_widths[-2], layer_widths[-1], bias=last_bias)
        )

        self.layers = torch.nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            init_xavier_uniform(layer)

    def forward(self, x, data=None):
        """Forward pass"""
        return self.layers(x)


class TypesMLP(torch.nn.Module):
    """
    The local energy model :math:`\epsilon` are multi-layer perceptron (MLP)
    that use as en input a representation of the atomic environment. There can
    be distinct models for each central atomic species.

    Parameters
    ----------
    layer_widths (list):
        List of the widths of the MLP's hidden layers. The input and output width are set by the calculator size and 1 respectively
    activation (nn.Module, optional):
        The activation function to use
            (default: :obj:`torch.nn.Tanh()`)
    species (torch.Tensor, optional):
        use a different set of weights for each type of atoms
            (default: :obj:`None`)

    """

    name: Final[str] = "TypesMLP"

    def __init__(
        self,
        layer_widths: List[int],
        activation: torch.nn.Module = torch.nn.Tanh(),
        species: Optional[torch.Tensor] = None,
    ):
        super(TypesMLP, self).__init__()

        self.weights_per_species = False
        if species is not None:
            self.weights_per_species = True
            species = torch.unique(species)
            self.register_buffer("species", species)
        else:
            self.species = species

        if self.weights_per_species:
            self.mlp = torch.nn.ModuleList()
            for _ in self.species:
                self.mlp.append(MLP(layer_widths, activation))
        else:
            self.mlp = MLP(layer_widths, activation)

    def reset_parameters(self):
        if self.weights_per_species:
            for mod in self.mlp:
                mod.reset_parameters()
        else:
            self.mlp.reset_parameters()

    def forward(self, features, data):
        # predict atomic energies
        yi = torch.zeros_like(data.batch, dtype=features.dtype).view(-1, 1)
        if self.weights_per_species:
            for ii, mlp in enumerate(self.mlp):
                mask = self.species[ii] == data.atom_types
                yi[mask] = mlp(features[mask])
        else:
            yi = self.mlp(features)

        return yi
