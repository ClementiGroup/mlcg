import json
import os
import time
from typing import List, Tuple, Any, Dict, Optional, Sequence, Union
import torch
from torch.distributions.normal import Normal
import numpy as np
import warnings

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

    def write(self, data: Optional[AtomicData] = None):
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

        super().write(data=data)

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
    r"""Langevin simulation that measures its own speed and memory,
    to use for benchmarks.

    It runs exactly as :class:`LangevinSimulation` (same loop, same
    outputs) and times the steps after a warm-up:

    - the model preparation (set-up, ``optimize`` with its compilation,
      first force evaluation) is reported as ``prepare_s``;
    - the first ``timesteps_burnout`` steps are a warm-up, reported as
      ``warmup_s`` and not part of the step time;
    - the following steps are timed in chunks of ``chunk_steps``, with the
      device synchronized at both ends of a chunk. The clock stops while a
      frame is saved, exported or logged, so the result is the cost of the
      steps alone. Each save still waits for the device, which costs the
      steps around it some time on small systems: for the pure step time
      keep ``save_interval`` at ``chunk_steps`` or above. A last, incomplete
      chunk is dropped;
    - peak memory is reset when the timing starts, so that the memory of
      the steps is separate from that of the setup.

    After :meth:`simulate`, ``results`` holds (and
    ``{filename}_benchmark.json`` stores):

    ``ms_per_step``, ``ms_per_step_min``, ``ms_per_step_max``
        median, fastest and slowest chunk, in ms per step. The median and
        not the mean, because what disturbs a chunk (another process on
        the device, a garbage collection) only makes it slower: one such
        chunk shifts the mean but not the median, and still shows in the
        maximum. A difference between two runs below this spread is noise.
    ``chunk_ms_per_step``, ``chunk_steps``, ``steps_timed``
        the time of every chunk and how many steps were timed. The mean of
        ``chunk_ms_per_step`` is the mean time per step of the timed part.
    ``steps_per_s``, ``structure_steps_per_s``
        steps per second, from the median ``ms_per_step``, and the same
        times the number of structures simulated in parallel.
    ``ns_per_day``, ``ns_per_day_per_structure``
        simulated time per day, from the median ``ms_per_step``, summed
        over the structures and for one of them. Only with
        ``timestep_fs``, since ``dt`` has no fixed unit.
    ``prepare_s``, ``warmup_s``
        seconds from the start of :meth:`simulate` to the first step, and
        seconds spent in the warm-up steps.
    ``recompiles``
        graphs ``torch.compile`` compiled while timing. Anything but 0
        means the warm-up was too short: the timing includes compilation.
    ``setup_peak_mem_mb``, ``step_peak_mem_mb``, ``step_peak_reserved_mb``
        peak memory of the torch allocator before the timing starts
        (preparation and warm-up) and while timing,
        and the peak it reserved from the device while timing (GPU only).
    ``process_mem_mb``
        GPU memory of the whole process at the end of the run, as
        ``nvidia-smi`` shows it: it adds the CUDA context and what other
        libraries allocate. ``None`` without ``pynvml`` or on CPU.
    ``n_structures``, ``n_beads``, ``dt``, ``optimize``, ``device``,
    ``dtype``, ``mlcg``, ``torch``, ``cuda``, ``matmul_precision``
        what was run, and on what.

    Parameters
    ----------
    timesteps_burnout :
        Number of warm-up steps before the timing starts.
    chunk_steps :
        Number of steps in a timed chunk. ``n_timesteps`` must cover the
        warm-up and at least one chunk; three or more give a spread.
    timestep_fs :
        Length of ``dt`` in femtoseconds, to report ns/day.

    Example
    -------
    .. code:: python

        sim = LangevinBenchmark(
            friction=1.0, dt=0.004, n_timesteps=1100, save_interval=100,
            timesteps_burnout=100, chunk_steps=200, timestep_fs=4.0,
            device="cuda", optimize="auto", filename="benchmark/run",
        )
        sim.attach_model_and_configurations(model, configurations, beta)
        sim.simulate()
        print(sim.results["ms_per_step"], sim.results["step_peak_mem_mb"])
    """

    def __init__(
        self,
        *,
        timesteps_burnout: int = 100,
        chunk_steps: int = 100,
        timestep_fs: Optional[float] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if self.n_timesteps < timesteps_burnout + chunk_steps:
            raise ValueError(
                f"n_timesteps ({self.n_timesteps}) must cover the warm-up "
                f"({timesteps_burnout}) and one chunk ({chunk_steps})"
            )
        self.burnout = timesteps_burnout
        self.chunk_steps = chunk_steps
        self.timestep_fs = timestep_fs
        self.results = None
        self._reset()

    def _reset(self):
        self._steps_done, self._chunk_steps_done = 0, 0
        self._chunk_s, self._chunk_ms, self._clock_start = 0.0, [], None
        self._peak_memory_mb()
        self._simulate_start = self._now()

    def _now(self) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def _stop_clock(self):
        if self._clock_start is not None:
            self._chunk_s += self._now() - self._clock_start
            self._clock_start = None

    def _peak_memory_mb(self) -> Tuple[Optional[float], Optional[float]]:
        """Peak allocated and reserved memory since the last reset; resets."""
        if self.device.type != "cuda":
            return None, None
        peaks = (
            torch.cuda.max_memory_allocated(self.device) / 1024**2,
            torch.cuda.max_memory_reserved(self.device) / 1024**2,
        )
        torch.cuda.reset_peak_memory_stats(self.device)
        return peaks

    def _process_memory_mb(self) -> Optional[float]:
        if self.device.type != "cuda":
            return None
        try:
            import pynvml
        except ImportError:
            return None
        pynvml.nvmlInit()
        # NVML numbers the physical GPUs, torch the visible ones
        uuid = torch.cuda.get_device_properties(self.device).uuid
        handle = pynvml.nvmlDeviceGetHandleByUUID(f"GPU-{uuid}")
        for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            if process.pid == os.getpid() and process.usedGpuMemory:
                return process.usedGpuMemory / 1024**2
        return None

    def timestep(
        self, data: AtomicData, forces: torch.Tensor
    ) -> Tuple[AtomicData, torch.Tensor, torch.Tensor]:
        """A :meth:`LangevinSimulation.timestep`, timed after the warm-up."""
        timed = self._steps_done >= self.burnout
        if self._steps_done == 0:
            self._warmup_start = self._now()
            self._prepare_s = self._warmup_start - self._simulate_start
        if self._steps_done == self.burnout:
            self._setup_peak_mb = self._peak_memory_mb()[0]
            self._graphs_before = _compiled_graphs()
            self._warmup_s = self._now() - self._warmup_start
        if timed and self._clock_start is None:
            self._clock_start = self._now()
        out = super().timestep(data, forces)
        self._steps_done += 1
        if timed:
            self._chunk_steps_done += 1
            if self._chunk_steps_done == self.chunk_steps:
                self._stop_clock()
                self._chunk_ms.append(1e3 * self._chunk_s / self.chunk_steps)
                self._chunk_s, self._chunk_steps_done = 0.0, 0
        return out

    def save(self, *args, **kwargs):
        """Stops the clock: saving, and the export and log that follow it
        in the same iteration, are not part of the step."""
        self._stop_clock()
        super().save(*args, **kwargs)

    def simulate(self, overwrite: bool = False, prof=None) -> None:
        """Runs the simulation as :meth:`LangevinSimulation.simulate` and
        fills ``results`` (see the class documentation)."""
        self._reset()
        super().simulate(overwrite=overwrite, prof=prof)
        if not self._chunk_ms:
            raise RuntimeError(
                "No chunk was timed: the simulation restarted from a "
                "checkpoint too close to n_timesteps"
            )
        self.results = self._collect_results()
        with open(f"{self.filename}_benchmark.json", "w") as file:
            json.dump(self.results, file, indent=2)
        if self.log_interval is not None:
            self._write_log(self._summary_line())

    def _collect_results(self) -> Dict[str, Any]:
        from .. import __version__
        from ..nn.flash_models import describe_optimization

        step_peak_mb, step_reserved_mb = self._peak_memory_mb()
        s_per_step = float(np.median(self._chunk_ms)) / 1e3
        results = {
            "ms_per_step": 1e3 * s_per_step,
            "ms_per_step_min": min(self._chunk_ms),
            "ms_per_step_max": max(self._chunk_ms),
            "chunk_ms_per_step": self._chunk_ms,
            "chunk_steps": self.chunk_steps,
            "steps_timed": len(self._chunk_ms) * self.chunk_steps,
            "steps_per_s": 1.0 / s_per_step,
            "structure_steps_per_s": self.n_sims / s_per_step,
            "ns_per_day": None,
            "ns_per_day_per_structure": None,
            "prepare_s": self._prepare_s,
            "warmup_s": self._warmup_s,
            "recompiles": _compiled_graphs() - self._graphs_before,
            "setup_peak_mem_mb": self._setup_peak_mb,
            "step_peak_mem_mb": step_peak_mb,
            "step_peak_reserved_mb": step_reserved_mb,
            "process_mem_mb": self._process_memory_mb(),
            "n_structures": self.n_sims,
            "n_beads": self.initial_data[POSITIONS_KEY].shape[0],
            "dt": self.dt,
            "optimize": describe_optimization(
                self.optimize, self.optimize_report, self._graphed_model
            ),
            "device": (
                torch.cuda.get_device_name(self.device)
                if self.device.type == "cuda"
                else "cpu"
            ),
            "dtype": str(self.dtype),
            "mlcg": __version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "matmul_precision": torch.get_float32_matmul_precision(),
        }
        if self.timestep_fs is not None:
            per_structure = self.timestep_fs * 1e-6 * 86400 / s_per_step
            results["ns_per_day_per_structure"] = per_structure
            results["ns_per_day"] = self.n_sims * per_structure
        return results

    def _summary_line(self) -> str:
        r = self.results
        line = (
            f"Benchmark: {r['ms_per_step']:.3f} ms/step "
            f"({r['ms_per_step_min']:.3f}-{r['ms_per_step_max']:.3f} over "
            f"{len(r['chunk_ms_per_step'])} chunks of {r['chunk_steps']} "
            f"steps), preparation {r['prepare_s']:.1f} s, "
            f"warm-up {r['warmup_s']:.1f} s, "
            f"{r['recompiles']} recompiles"
        )
        if r["ns_per_day"] is not None:
            line += f", {r['ns_per_day']:.1f} ns/day"
        if r["step_peak_mem_mb"] is not None:
            line += f", peak memory {r['step_peak_mem_mb']:.0f} MB"
        return line


def _compiled_graphs() -> int:
    import torch._dynamo.utils as dynamo_utils

    return int(dynamo_utils.counters["stats"]["unique_graphs"])


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

    def input_option_checks(self):
        super(OverdampedSimulation, self).input_option_checks()
        if self.save_velocities:
            raise ValueError(
                "save_velocities=True is not supported for OverdampedSimulation "
                "because overdamped dynamics has no velocities. Use "
                "LangevinSimulation if velocity output is needed."
            )


# pipe the doc from the base class into the child class so that it's properly
# displayed by sphinx
OverdampedSimulation.__doc__ += _Simulation.__doc__
