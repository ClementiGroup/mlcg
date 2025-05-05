import torch
from ...data import AtomicData


class _Prior(torch.nn.Module):
    r"""Abstract prior class

    Priors are torch.nn.Module objects that should represent an energy term that has
    a traditional functional form, like a harmonic potential or a fourier series.

    They should called as functions with AtomicData objects as parameters and then
    they should populate their `.out` field with their energy, similar to how
    SchNet modules do this.

    The different parameters needed for different beads in this interactions should
    be stored in `torch.nn.parameter.Buffer` objects to avoid any backwards passes
    over this parameters.
    """

    def __init__(self) -> None:
        r"""
        This is the class initialization for the prior.

        It should populate the parameter buffers, if needed, and all of the
        other relevant parameters need to be able to call the `self.forward`
        method.
        """
        super(_Prior, self).__init__()

    def forward(self, data: AtomicData) -> AtomicData:
        r"""
        Method used to evaluate the prior object over a structure

        it should populate the data.out field with the energy
        predictions from this model
        """
        raise NotImplementedError

    @staticmethod
    def data2features(self, data: AtomicData) -> torch.Tensor:
        """
        Method that returns a tensor of all the (physical) features
        related to a prior for the given Atomic Data instance

        For example, for a bond prior it should return a tensor with the
        lenghts of all the bonds that this prior should evaluate to
        """
        raise NotImplementedError

    def data2parameters(self, data: AtomicData):
        r"""
        Method used to obtain the prior parameters for all the given features
        associated to this prior.

        For example, for a prior representing a harmonic function restraining
        the bonds of a system, of the form f(x) = k(x-x_0)^2, it should return
        the values of k and x_0 that will be used for each of the features
        given by the `data2features` object.
        """
        raise NotImplementedError
