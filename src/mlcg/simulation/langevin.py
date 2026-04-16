from typing import List, Tuple, Any, Dict, Sequence, Union
import torch
from torch.distributions.normal import Normal
import numpy as np
import warnings
from copy import deepcopy
from tqdm import tqdm

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
    r"""Langevin simulation class for trained models.

    The following [BAOAB]_ integration scheme is used, where::

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
        & \\
        \epsilon =& \exp(-\eta \; \Delta t) \\
        & \\
        \alpha =& \sqrt{(1 - \epsilon^2) / \beta m}
    
    Initial velocities are sampled from the Maxwell-Boltzmann distribution for 
    the provided beta.

    A diffusion constant :math:`D` can be back-calculated using
    the Einstein relation:

    .. math::
        D = 1 / (\beta  \eta)
    
    For a larger discussion on Langevin Dynamics integrators, please refer to [MDBook]_.
        
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
            The masses of each atom
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
        # Use pre-allocated buffer and fill in-place to avoid allocation
        self._noise_buffer.normal_(generator=self.rng)
        noise = self.beta_mass_ratio * self._noise_buffer
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

        # Pre-allocate noise buffer to avoid allocation every timestep
        noise_shape = (self.n_sims * self.n_atoms, self.n_dims)
        self._noise_buffer = torch.empty(
            noise_shape, dtype=self.dtype, device=self.device
        )

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
LangevinSimulation.__doc__ += _Simulation.__doc__ + "\n"


class LangevinBenchmark(LangevinSimulation):

    def __init__(self, **kwargs: Any):

        if "timesteps_burnout" in kwargs.keys():
            self.burnout = kwargs["timesteps_burnout"]
            kwargs.pop("timesteps_burnout")
        else:
            self.burnout = 5000
        super().__init__(**kwargs)
        self.started_benchmarking = False
        self.times_ms = []

    def simulate(self, overwrite: bool = False, prof=None) -> np.ndarray:
        """Generates independent simulations.

        Parameters
        ----------
        overwrite :
            Set to True if you wish to overwrite any saved simulation data

        Returns
        -------
        simulated_coords :
            Shape [n_simulations, n_frames, n_atoms, n_dimensions]
            Also an attribute; stores the simulation coordinates at the
            save_interval
        """

        self._set_up_simulation(overwrite)
        data = deepcopy(self.initial_data)
        data = data.to(self.device)
        self.compile_model(data)
        _, forces = self.calculate_potential_and_forces(data)
        if self.export_interval is not None:
            t_init = self.current_timestep * self.export_interval
        else:
            t_init = 0
        if t_init >= self.n_timesteps:
            raise ValueError(
                f"Simulation has already been running for {t_init} steps, which is larger than the target number of steps {self.n_timesteps}"
            )
        torch.cuda.synchronize()

        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)

        for t in tqdm(
            range(t_init, self.n_timesteps),
            desc="Simulation timestep",
            mininterval=self.tqdm_refresh,
            initial=t_init,
            total=self.n_timesteps,
        ):
            # step forward in time
            if not self.started_benchmarking and t > self.burnout:

                self.started_benchmarking = True
                start_ev.record()
            data, potential, forces = self.timestep(data, forces)
            self.sim_t = t
            pbc = getattr(data, "pbc", None)
            cell = getattr(data, "cell", None)
            if all([feat != None for feat in [pbc, cell]]):
                data = wrap_positions(data, self.device)

            # save to arrays if relevant
            if (t + 1) % self.save_interval == 0:
                # save arrays
                self.save(
                    data=data,
                    forces=forces,
                    potential=potential,
                    t=t,
                )

                # write numpys to file if relevant; this can be indented here because
                # it only happens when time points are also recorded
                if self.export_interval is not None:
                    if (t + 1) % self.export_interval == 0:
                        self.write()
                        if self.save_subroutine is not None:
                            self.save_subroutine(
                                data, (t + 1) // self.save_interval
                            )
                        if self.started_benchmarking:
                            end_ev.record()
                            torch.cuda.synchronize()
                            ms = start_ev.elapsed_time(end_ev)  # milliseconds
                            self.times_ms.append(ms)

                # log if relevant; this can be indented here because
                # it only happens when time when time points are also recorded
                if self.log_interval is not None:
                    if int((t + 1) % self.log_interval) == 0:
                        self.log((t + 1) // self.save_interval)

            if self.sim_subroutine != None:
                if (t + 1) % self.sim_subroutine_interval == 0:
                    data = self.sim_subroutine(data)

            # reset data outputs to collect the new forces/energies
            data.out = {}
            if prof:
                prof.step()

        # if relevant, save the remainder of the simulation
        if self.export_interval is not None:
            if int(t + 1) % self.export_interval > 0:
                self.write()

        # if relevant, log that simulation has been completed
        if self.log_interval is not None:
            self.summary()

        self.reshape_output()

        self._simulated = True

        return

    def write(self):
        super().write()
        if self.started_benchmarking:
            aux_np = np.array(self.times_ms)
            np.save(f"{self.filename}_ms_per_times.npy", aux_np)


class OverdampedSimulation(_Simulation):
    r"""Overdamped Langevin simulation class for trained models.

    The following Brownian motion scheme is used:

    .. math::
        dX_t = - \nabla U( X_t )   D  \Delta t + \sqrt{( 2  D \Delta t / \beta )} dW_t

    for coordinates :math:`X_t` at time :math:`t`, potential energy :math:`U`,
    diffusion :math:`D`, thermodynamic inverse temperature :math:`\beta`,
    time step :math:`\Delta t`, and stochastic Weiner process :math:`W`.

    Due to the nature of Overdamped Langevin dynamics, the masses and velocities
    are not used.

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
        self.n_sims = len(configurations)
        self.n_atoms = len(configurations[0].atom_types)
        self.n_dims = configurations[0].pos.shape[1]
        self.expanded_beta = self.beta.repeat_interleave(self.n_atoms)[:, None]
        self.diffusion = 1 / self.expanded_beta / self.friction
        self._dtau = self.diffusion * self.dt
        # Pre-allocate noise buffer to avoid allocation every timestep
        noise_shape = (self.n_sims * self.n_atoms, self.n_dims)
        self._noise_buffer = torch.empty(
            noise_shape, dtype=self.dtype, device=self.device
        )

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

        self._noise_buffer.normal_(generator=self.rng)

        x_new = (
            x_old.detach()
            + forces * self._dtau
            + torch.sqrt(2 * self._dtau) * self._noise_buffer
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
