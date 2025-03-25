from typing import List, Tuple, Any, Dict, Sequence, Union
import torch
from torch.distributions.normal import Normal
import numpy as np
import warnings
from copy import deepcopy

from ..data.atomic_data import AtomicData
from ..data._keys import (
    MASS_KEY,
    VELOCITY_KEY,
    POSITIONS_KEY,
    ENERGY_KEY,
)
from .base import _Simulation

torch_pi = torch.tensor(np.pi)


class LangevinSimulation(_Simulation):
    r"""Langevin simulatin class for trained models.

    The following `BAOAB integration scheme <https://doi.org/10.1007/978-3-319-16375-8>`_ is used, where::

        B = deterministic velocity update
        A = deterministic position update
        O = stochastic velocity update

    We have implemented the following update so as to only calculate
    forces once per timestep:

    .. math::
        [B]\;& V_{t+1/2} = V_t + \frac{\Delta t}{2m}  F(X_t) \\
        [A]\;& X_{t+1/2} = X_t + \frac{\Delta t}{2}V_{t+1/2}  \\
        [O]\;& \tilde{V}_{t+1/2} = \epsilon V_{t+1/2} + \alpha dW_t \\
        [A]\;& X_{t+1} = X_{t+1/2} + \frac{\Delta t}{2} \tilde{V}_{t+1/2}  \\
        [B]\;& V_{t+1} = \tilde{V}_{t+1/2} + \frac{\Delta t}{2m}  F(X_{t+1})

    Where, :math:`dW_t` is a noise drawn from :math:`\mathcal{N}(0,1)`,
    :math:`\eta` is the friction, :math:`\epsilon` is the velocity scale,
    :math:`\alpha` is the noise scale, and:

    .. math::
        F(X_t) =& - \nabla  U(X_t)  \\
        \epsilon =& \exp(-\eta \; \Delta t) \\
        \alpha =& \sqrt{1 - \epsilon^2}

    A diffusion constant :math:`D` can be back-calculated using
    the Einstein relation:

    .. math::
        D = 1 / (\beta  \eta)

    Initial velocities are set to zero if not provided.

    Parameters
    ----------

    friction :
        Friction value for Langevin scheme, in units of inverse time.

    """

    def __init__(self, friction: float = 1e-3, **kwargs: Any):
        super(LangevinSimulation, self).__init__(**kwargs)

        assert friction > 0
        self.friction = friction

        self.vscale = np.exp(-self.dt * self.friction)
        self.noisescale = np.sqrt(1 - self.vscale * self.vscale)

    @staticmethod
    def sample_maxwell_boltzmann(
        betas: torch.Tensor, masses: torch.Tensor
    ) -> torch.Tensor:
        """Returns n_samples atomic velocites according to Maxwell-Boltzmann
        distribution at the corresponding temperature and mass values.

        Parameters
        ----------
        n_samples:
            Number of atoms to generate velocites for
        betas:
            The inverse thermodynamic temperature of each atom
        masses:
            Them masses of each atom
        """
        assert all([m > 0 for m in masses])
        scale = torch.sqrt(1 / (betas * masses))
        dist = Normal(loc=0.00, scale=scale)
        velocities = dist.sample((3,)).t()
        return velocities

    def timestep(
        self, data: AtomicData, forces: torch.Tensor
    ) -> Tuple[AtomicData, torch.Tensor, torch.Tensor]:
        """Timestep method for Langevin dynamics
        Parameters
        ----------
        data:
            atomic structure at t
        forces:
            forces evaluated at t
        Returns
        -------
        data:
            atomic structure at t+1
        forces:
            forces evaluated at t+1
        potential:
            potential evaluated at t+1
        """
        v_old = data[VELOCITY_KEY]
        masses = data[MASS_KEY]
        x_old = data[POSITIONS_KEY]
        # B
        v_new = v_old + 0.5 * self.dt * forces / masses[:, None]

        # A (position update)
        x_new = x_old + v_new * self.dt * 0.5

        # O (noise)
        noise = self.beta_mass_ratio * torch.randn(
            size=x_new.size(),
            dtype=x_new.dtype,
            generator=self.rng,
            device=self.device,
        )
        v_new = v_new * self.vscale + self.noisescale * noise
        # A
        x_new = x_new + v_new * self.dt * 0.5
        data[POSITIONS_KEY] = x_new
        potential, forces = self.calculate_potential_and_forces(data)
        # B
        v_new = v_new + 0.5 * self.dt * forces / masses[:, None]
        data[VELOCITY_KEY] = v_new

        return data, potential, forces

    def _attach_configurations(
        self, configurations: List[AtomicData], beta: Union[float, List[float]]
    ):
        """Setup the starting atomic configurations.

        Parameters
        ----------
        configurations : List[AtomicData]
            List of AtomicData instances representing initial structures for
            parallel simulations.
        beta:
            Desired temperature(s) of the simulation
        """
        super(LangevinSimulation, self)._attach_configurations(
            configurations, beta
        )

        # Initialize velocities according to Maxwell-Boltzmann distribution
        if VELOCITY_KEY not in self.initial_data:
            # initialize velocities at zero
            self.initial_data[VELOCITY_KEY] = (
                LangevinSimulation.sample_maxwell_boltzmann(
                    self.beta.repeat_interleave(self.n_atoms),
                    self.initial_data[MASS_KEY],
                ).to(self.dtype)
            )
        assert (
            self.initial_data[VELOCITY_KEY].shape
            == self.initial_data[POSITIONS_KEY].shape
        )
        self.beta_mass_ratio = torch.sqrt(
            1.0
            / self.beta.repeat_interleave(self.n_atoms)
            / self.initial_data[MASS_KEY]
        )[:, None].to(self.dtype)

    def _set_up_simulation(self, overwrite: bool = False):
        """Method to setup up saving and logging options"""
        super()._set_up_simulation(overwrite)

        if self.save_energies:
            self.simulated_kinetic_energies = torch.zeros(
                self._save_size, self.n_sims
            )
        else:
            self.simulated_kinetic_energies = None

    def save(
        self,
        data: AtomicData,
        forces: torch.Tensor,
        potential: torch.Tensor,
        t: int,
    ):
        """Utilities to store saved values of coordinates and, if relevant,
        also forces, potential, and/or kinetic energy
        Parameters
        ----------
        x_new :
            current coordinates
        v_new :
            current velocities
        forces:
            current forces
        potential :
            current potential
        masses :
            atom masses for kinetic energy calculation
        t :
            current timestep
        """
        super().save(data, forces, potential, t)

        v_new = data[VELOCITY_KEY].view(-1, self.n_atoms, self.n_dims)
        masses = data.masses.view(self.n_sims, self.n_atoms)

        save_ind = (
            t // self.save_interval
        ) - self._npy_file_index * self._save_size

        if self.save_energies:
            kes = 0.5 * torch.sum(
                torch.sum(masses[:, :, None] * v_new**2, dim=2), dim=1
            )
            self.simulated_kinetic_energies[save_ind, :] = kes

    def write(self):
        """Utility to save numpy arrays"""
        key = self._get_numpy_count()
        if self.save_energies:
            kinetic_energies_to_export = self.simulated_kinetic_energies
            kinetic_energies_to_export = self._swap_and_export(
                kinetic_energies_to_export
            )
            np.save(
                "{}_kineticenergy_{}.npy".format(self.filename, key),
                kinetic_energies_to_export,
            )

            # Reset simulate_kinetic_energies
            self.simulated_kinetic_energies = torch.zeros(
                self._save_size, self.n_sims
            )

        super().write()

    def reshape_output(self):
        super().reshape_output()
        if self.save_energies:
            self.simulated_kinetic_energies = self._swap_and_export(
                self.simulated_kinetic_energies
            )

    def attach_configurations(
        self, configurations: List[AtomicData], beta: Union[float, List[float]]
    ):
        warnings.warn(
            "using 'attach_configurations' is deprecated, use 'attach_model_and_configurations' instead.",
            DeprecationWarning,
        )
        self._attach_configurations(configurations, beta)


# pipe the doc from the base class into the child class so that it's properly
# displayed by sphinx
LangevinSimulation.__doc__ += _Simulation.__doc__


class OverdampedSimulation(_Simulation):
    r"""Overdamped Langevin simulation class for trained models.

    The following Brownian motion scheme is used:

    .. math::

        dX_t = - \nabla U( X_t )   D  \Delta t + \sqrt{( 2  D *\Delta t / \beta )} dW_t

    for coordinates :math:`X_t` at time :math:`t`, potential energy :math:`U`,
    diffusion :math:`D`, thermodynamic inverse temperature :math:`\beta`,
    time step :math:`\Delta t`, and stochastic Weiner process :math:`W`.

    Parameters
    ----------
    diffusion :
        The constant diffusion parameter :math:`D`. By default, the diffusion
        is set to unity and is absorbed into the :math:`\Delta t` argument.
        However, users may specify separate diffusion and :math:`\Delta t`
        parameters in the case that they have some estimate of the diffusion.
    """

    def __init__(self, friction: float = 1.0, **kwargs: Any):
        super(OverdampedSimulation, self).__init__(**kwargs)

        assert friction > 0
        self.friction = friction

    def _attach_configurations(
        self, configurations: List[AtomicData], beta: Union[float, List[float]]
    ):
        """Setup the starting atomic configurations.

        Parameters
        ----------
        configurations :
            List of AtomicData instances representing initial structures for
            parallel simulations.
        beta:
            Desired temperature(s) of the simulation.
        """
        super(OverdampedSimulation, self)._attach_configurations(
            configurations, beta, overdamped=True
        )

        if MASS_KEY in self.initial_data:
            warnings.warn(
                "Masses were provided, but will not be used since "
                "an overdamped Langevin scheme is being used for integration."
            )
        self.expanded_beta = self.beta.repeat_interleave(self.n_atoms)[:, None]
        self.diffusion = 1 / self.expanded_beta / self.friction
        self._dtau = self.diffusion * self.dt

    def timestep(
        self, data: AtomicData, forces: torch.Tensor
    ) -> Tuple[AtomicData, torch.Tensor, torch.Tensor]:
        """Timestep method for overdamped Langevin dynamics
        Parameters
        ----------
        data:
            atomic structure at t
        forces:
            forces evaluated at t
        Returns
        -------
        data:
            atomic structure at t+1
        forces:
            forces evaluated at t+1
        potential:
            potential evaluated at t+1
        """
        x_old = data[POSITIONS_KEY]
        noise = torch.randn(size=x_old.size(), generator=self.rng).to(
            self.device
        )
        x_new = (
            x_old.detach()
            + forces * self._dtau
            + np.sqrt(2 * self._dtau) * noise
        )
        data[POSITIONS_KEY] = x_new
        potential, forces = self.calculate_potential_and_forces(data)
        return data, potential, forces

    def attach_configurations(
        self, configurations: List[AtomicData], beta: Union[float, List[float]]
    ):
        warnings.warn(
            "using 'attach_configurations' is deprecated, use 'attach_model_and_configurations' instead.",
            DeprecationWarning,
        )
        self._attach_configurations(configurations, beta)


# pipe the doc from the base class into the child class so that it's properly
# displayed by sphinx
OverdampedSimulation.__doc__ += _Simulation.__doc__
