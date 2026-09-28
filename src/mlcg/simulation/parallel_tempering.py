# Authors: Nick Charron, Felix Musil, Clark Templeton
# Based on code from Yaoyi Chen and Andreas Kramer: https://github.com/noegroup/reform

from typing import List, Tuple, Any, Dict, Sequence
import time
import torch
import numpy as np
import warnings
from copy import deepcopy

from ..data.atomic_data import AtomicData
from ..data._keys import (
    MASS_KEY,
    VELOCITY_KEY,
    POSITIONS_KEY,
    ENERGY_KEY,
    ATOM_TYPE_KEY,
)
from .base import _Simulation
from .langevin import LangevinSimulation
from .specialize_prior import condense_all_priors_for_simulation


class PTSimulation(LangevinSimulation):
    r"""Parallel tempering simulation using a Langevin update scheme.
    For theoretical details on replica exchange/parallel tempering, see
    https://github.com/noegroup/reform.

    Briefly, a pair exchange is proposed, and the associated potential energies are used
    to compute a Metroplis-Hastings Boltzmann ratio. This ratio defines an acceptance threshold
        aginst which approved exchanges are sampled according to a unit uniform distribution.
        For a pair of configurations :math:`A` and :math:`B`, characterized by the respective
        potential energies :math:`U_A` and :math:`U_B` the the inverse thermodynamic temperatures
        :math:`\beta_A` and :math:`\beta_B`, the acceptance rate for exchanging configurations is:

        .. math::

                Acc = \exp{\left( (U_A - U_B) \times (\beta_A - \beta_B) \right)}

        Pairs of candidate configurations undergo exchange if :math:`\rho \sim U(0,1) < Acc`. Note
        that the exchanged velocities for each configuration must further be rescaled according to the
        square root of their inverse beta ratios. See _perform_exchange for more details.

    Currently we only implement parallel tempering for Langevin dynamics.
    Be aware that the output will contain information (e.g., coordinates)
    for all replicas.

    In addition to the typical outputs of `LangevinSimulation`, `PTSimulation` also
    ouputs information about exchange acceptance between replicas at each export.
    This information takes the form of an `acceptance_matrix`, which has shape
    `(n_betas, n_betas)`. The upper triangular portion of the matrix counts
    accepted exchanges between adjacent tempeartures, while the lower triangular
    portion of the matrix counts rejected exchanges between adjacent temperatures. For
    example, the entry `[1,2]` contains the number of accepted exchanges between replicas
    running at the temperatures associated with `beta_1` and `beta_2`, while the entry
    `[2,1]` counts the number of rejected echanges at those two temperatures. The sum of
    such complementary entries across the matrix diagonal always sum to the total number
    of exhchanges proposed between each export interval.

    A per-frame log of the accepted exchanges is exported alongside it, with shape
    `(n_sims, n_frames)`. Column `k` holds the exchanges that were performed *after*
    frame `k` was recorded, so replaying the columns in order reconstructs which
    replica each configuration occupied at every frame. The two sides of an accepted
    exchange are marked with `+p` and `-p`, where the parity `p` is 2 for the
    even pairs (0, 1), (2, 3), ... and 1 for the odd pairs (1, 2), (3, 4), ...


    Note: This implementation only allows for replica exchanges between directly
    adjecent temperatures implied by the user-supplied list of beta values.

    Parameters
    ----------
    friction:
        Scalar friction to use for Langevin updates
    exchange_interval:
        Specifies the number of simulation steps to take before attempting
        replica exchange. Should be a multiple of `save_interval`, otherwise
        the exchange log cannot hold all the attempted exchanges.
    """

    def __init__(
        self,
        friction: float = 1e-3,
        exchange_interval: int = 100,
        **kwargs,
    ):
        super(PTSimulation, self).__init__(
            friction=friction,
            sim_subroutine=self.detect_and_exchange_replicas,
            sim_subroutine_interval=exchange_interval,
            save_subroutine=self.save_exchanges,
            **kwargs,
        )

        self.exchange_interval = exchange_interval
        # Each exchange is logged in the column of the last saved frame. If
        # exchanges are more frequent than saves, several of them share a
        # column and only the last one is kept in the exchange log. The
        # dynamics are unaffected, but the log can no longer be used to
        # demultiplex the replicas, so warn about it.
        if exchange_interval % self.save_interval != 0:
            warnings.warn(
                "exchange_interval ({}) is not a multiple of save_interval ({}): "
                "the exchange log holds at most one exchange per saved frame, so "
                "some exchanges will be missing from it.".format(
                    exchange_interval, self.save_interval
                )
            )

        self._replica_exchange_approved = 0
        self._replica_exchange_attempts = 0
        if self.read_checkpoint_file is None:
            self._old_save_step = 0
        else:
            self._old_save_step = (
                self.current_timestep
                * self.export_interval
                // self.save_interval
            )

    def _reset_exchange_stats(self):
        """Setup function that resets exchange statistics before running a simulation"""
        self._replica_exchange_attempts = 0
        self._replica_exchange_approved = 0

    def _reset_acceptance_matrix(self):
        """(Re)allocates the matrix accumulating exchange statistics"""
        self.acceptance_matrix = torch.zeros(
            self.n_replicas, self.n_replicas, device=self.device
        )

    def _set_up_simulation(self, overwrite: bool = False):
        super(PTSimulation, self)._set_up_simulation(overwrite=overwrite)
        self._reset_exchange_stats()
        self.exchange_arr = torch.zeros(
            (self.n_sims, self.n_timesteps // self.save_interval),
            dtype=torch.int8,
        )
        self._reset_acceptance_matrix()

    def attach_model_and_configurations(
        self,
        model: torch.nn.Module,
        configurations: List[AtomicData],
        betas: List[float],
    ):
        if self.specialize_priors:
            # `_attach_configurations` consumes `checkpointed_data`; keep a
            # handle on it so that the re-attachment below can still restore
            # the checkpointed positions and velocities instead of silently
            # restarting from the initial configurations.
            checkpointed_data = self.checkpointed_data
            new_configurations = self._attach_configurations(
                configurations, betas=betas
            )
            (
                model,
                condensed_configurations,
            ) = condense_all_priors_for_simulation(model, new_configurations)
            # Repeat attachment, this time with the condensed configurations
            self.checkpointed_data = checkpointed_data
            self._manually_reattach_configurations(condensed_configurations)
            self._attach_model(model)
            print("Prior models have been specialized for the simulation.")
            # the condensed configurations are the ones actually simulated, so
            # they (and not the inputs) are what makes the run reproducible
            saved_configurations = condensed_configurations
        else:
            self._attach_configurations(configurations, betas=betas)
            self._attach_model(model)
            saved_configurations = configurations

        if self.filename is not None:
            torch.save(
                (deepcopy(model), deepcopy(saved_configurations)),
                f"{self.filename}_specialized_model_and_config.pt",
            )

    def _manually_reattach_configurations(
        self, configurations: List[AtomicData]
    ):
        """Helper method that allows for re-attachment of prior-condensed configurations"""
        self.validate_data_list(configurations)
        self.initial_data = self.collate(configurations).to(device=self.device)

        if self.checkpointed_data is not None:
            # Load in checkpointed data values and then wipe to conserve space
            self.initial_data[VELOCITY_KEY] = self.checkpointed_data[
                VELOCITY_KEY
            ]
            self.initial_data[POSITIONS_KEY] = self.checkpointed_data[
                POSITIONS_KEY
            ]
            self.checkpointed_data = None

        else:
            # Initialize velocities according to Maxwell-Boltzmann distribution
            self.initial_data[VELOCITY_KEY] = (
                LangevinSimulation.sample_maxwell_boltzmann(
                    self.beta.repeat_interleave(self.n_atoms),
                    self.initial_data[MASS_KEY],
                ).to(self.dtype)
            )
        self.initial_data[MASS_KEY] = self.initial_data[MASS_KEY].to(self.dtype)
        self.initial_data[POSITIONS_KEY] = self.initial_data[POSITIONS_KEY].to(
            self.dtype
        )

    def _attach_configurations(
        self, configurations: List[AtomicData], betas: List[float]
    ) -> List[AtomicData]:
        r"""Attaches the configurations at each of the temperatures defined for
        parallel tempering simulations. If the initial configurations do not contain
        specified velocities, all velocities will be initialized to zero. Unlike
        other `_attach_configurations` methods of the other simulation classes,
        this method also returns the extended configurations across all temperatures
        for use in condensing the priors if specified for the simulation.

        Parameters
        ----------
        configurations:
            List of `AtomicData` instances representing the initial configurations.
        betas:
            List of floats, from largest to smallest, representing beta values
            for each subset of replicas

        Returns
        -------
        new_configurations:
            List of extended initial configurations across all temperatures
        """

        # beta checks
        if not isinstance(betas, list):
            raise ValueError(
                "Parallel tempering requires multiple temperatures, but only {} was supplied.".format(
                    betas
                )
            )
        if not all([b >= 0 for b in betas]):
            raise ValueError(
                "All betas must be positive, but {} contains an illegal value.".format(
                    betas
                )
            )
        assert all([np.isfinite(b) for b in betas])
        if not (np.array(betas[::-1]) == np.sort(betas[::-1])).all():
            raise ValueError(
                "Betas must be in order of increasing temperature."
            )

        self.n_indep_sims = len(configurations)
        self.n_replicas = len(betas)
        if self.n_replicas < 2:
            raise ValueError(
                "Parallel tempering requires at least two temperatures, but "
                "{} was supplied.".format(betas)
            )
        # copy the configurations across each beta/temperature
        new_configurations = []
        extended_betas = []
        for beta in betas:
            extended_betas += self.n_indep_sims * [beta]
            for configuration in configurations:
                config = deepcopy(configuration)
                new_configurations.append(config)
        self.beta = torch.tensor(extended_betas).to(self.device).to(self.dtype)

        self.validate_data_list(new_configurations)
        self.initial_data = self.collate(new_configurations).to(
            device=self.device
        )
        self.n_sims = len(new_configurations)
        self.n_atoms = len(new_configurations[0].atom_types)
        self.n_dims = new_configurations[0].pos.shape[1]

        # Initialize velocities according to Maxwell-Boltzmann distribution

        if self.checkpointed_data is not None:
            # Load in checkpointed data values and then wipe to conserve space
            self.initial_data[VELOCITY_KEY] = self.checkpointed_data[
                VELOCITY_KEY
            ]
            self.initial_data[POSITIONS_KEY] = self.checkpointed_data[
                POSITIONS_KEY
            ]
            self.checkpointed_data = None

        else:
            self.initial_data[POSITIONS_KEY] = self.initial_data[
                POSITIONS_KEY
            ].to(self.dtype)
            # Initialize velocities according to Maxwell-Boltzmann distribution
            self.initial_data[VELOCITY_KEY] = (
                LangevinSimulation.sample_maxwell_boltzmann(
                    self.beta.repeat_interleave(self.n_atoms),
                    self.initial_data[MASS_KEY],
                ).to(self.dtype)
            )

        self.initial_data[MASS_KEY] = self.initial_data[MASS_KEY].to(self.dtype)

        self.beta_mass_ratio = torch.sqrt(
            1.0
            / self.beta.repeat_interleave(self.n_atoms)
            / self.initial_data[MASS_KEY]
        )[:, None]

        # Setup alternating even/odd pair exchanges
        self._propose_even_pairs = True

        # (0, 1), (2, 3), ...
        even_pairs = [(i, i + 1) for i in range(self.n_replicas)[:-1:2]]
        # (1, 2), (3, 4), ...
        odd_pairs = [(i, i + 1) for i in range(self.n_replicas)[1:-1:2]]
        if len(odd_pairs) == 0:
            odd_pairs = even_pairs
        # the proposed pairs never change during the simulation, so the
        # tensors describing them are built once here
        self._even_pairs = self._build_pair_cache(even_pairs)
        self._odd_pairs = self._build_pair_cache(odd_pairs)
        self._reset_acceptance_matrix()

        self.initial_pos_spread = (
            torch.cat([data.pos.std(dim=1) for data in new_configurations])
            .max()
            .detach()
            .cpu()
        )
        return new_configurations

    def attach_model(self, model: torch.nn.Module):
        warnings.warn(
            "using 'attach_model' is deprecated, use 'attach_model_and_configurations' instead.",
            DeprecationWarning,
        )
        self._attach_model(model)

    def attach_configurations(
        self, configurations: List[AtomicData], betas: List[float]
    ):
        warnings.warn(
            "using 'attach_configurations' is deprecated, use 'attach_model_and_configurations' instead.",
            DeprecationWarning,
        )
        self._attach_configurations(configurations, betas)

    def get_replica_info(self, replica_num: int = 0) -> Dict:
        """Returns information for the specified replica after the
        parallel tempering simulation has completed

        Parameters
        ----------
        replica_num:
            integer specifying which replica to interrogate

        Returns
        -------
        dict:
            dictionary with replica exchange information about the
            desired replica
        """

        if (
            type(replica_num) is not int
            or replica_num < 0
            or replica_num >= self.n_replicas
        ):
            raise ValueError("Please provide a valid replica number.")
        indices = torch.arange(
            replica_num * self.n_indep_sims,
            (replica_num + 1) * self.n_indep_sims,
        )
        return {
            # `self.beta` holds one entry per simulation, so index the first
            # simulation of the requested replica
            "beta": self.beta[replica_num * self.n_indep_sims].item(),
            "indices_in_the_output": indices,
        }

    def _build_pair_cache(
        self, replica_pairs: List[Tuple[int, int]]
    ) -> Dict[str, Any]:
        """Precomputes the quantities that are constant for a given set of
        proposed replica pairs: the simulation indices on both sides of every
        pair, their beta values, and the replica indices used to accumulate the
        acceptance matrix. They are built once at attachment time and kept on
        the simulation device, so that no index tensor has to be created or
        transferred to the device during the simulation.

        Parameters
        ----------
        replica_pairs:
            List of (replica, replica) index tuples proposed for exchange

        Returns
        -------
        dict:
            Cached tensors describing the proposed exchanges
        """
        sim_idx = torch.arange(self.n_indep_sims, device=self.device)
        pair_a, pair_b = [], []
        replica_a, replica_b = [], []
        for rep_a, rep_b in replica_pairs:
            pair_a.append(sim_idx + rep_a * self.n_indep_sims)
            pair_b.append(sim_idx + rep_b * self.n_indep_sims)
            replica_a.append(rep_a)
            replica_b.append(rep_b)
        pair_a = torch.cat(pair_a)
        pair_b = torch.cat(pair_b)
        return {
            "a": pair_a,
            "b": pair_b,
            "beta_a": self.beta[pair_a],
            "beta_b": self.beta[pair_b],
            # replica index of each proposed pair, in proposal order
            "replica_a": torch.tensor(replica_a, device=self.device),
            "replica_b": torch.tensor(replica_b, device=self.device),
            "n_pairs": len(replica_pairs),
        }

    def _get_proposed_pairs(self) -> Dict[str, Any]:
        """Proposes the even and odd exchange pairs alternatively each time the
        _detect_exchange method is called. Exchanges can only happen between direcly adjacent
        temperatures defined by the user supplied beta series.

        Returns
        -------
        dict:
            The cache built by `_build_pair_cache` for the current parity,
            holding the simulation indices of both sides of every proposed
            pair ("a"/"b") together with their beta and replica indices.
        """
        if self._propose_even_pairs:
            return self._even_pairs
        else:
            return self._odd_pairs

    def _detect_exchange(self, data: AtomicData) -> Dict:
        """Proposes and checks pairs to be exchanged for parallel tempering, tracking
        the associated acceptance/rejection statistics for each temperature pair at
        each detection query.

        Parameters
        ----------
        data:
            Collated AtomicData instance containing the beta values and current potential
            energies for each replica.

        Returns
        -------
        dict:
            Dictionary containing the approved exchanges
        """
        pairs = self._get_proposed_pairs()
        pair_a, pair_b = pairs["a"], pairs["b"]
        # detached: the model energies still carry their autograd graph, which
        # would be kept alive by the tensors derived from them below
        energies = data.out[ENERGY_KEY].detach()
        u_a, u_b = energies[pair_a], energies[pair_b]
        p_pair = (u_a - u_b) * (pairs["beta_a"] - pairs["beta_b"])
        # drawn from the simulation generator, so that `random_seed` makes the
        # exchanges reproducible too, and on the simulation device, so that no
        # host/device synchronisation is needed at every exchange
        uniform = torch.rand(
            p_pair.shape,
            dtype=p_pair.dtype,
            device=p_pair.device,
            generator=self.rng,
        )
        approved = torch.log(uniform) < p_pair
        self._replica_exchange_approved += torch.sum(approved)
        self._replica_exchange_attempts += len(pair_a)
        pairs_for_exchange = {"a": pair_a[approved], "b": pair_b[approved]}

        # Count the number of approved exchanges for each proposed pair. The
        # proposals are ordered pair-first, so one row per proposed pair.
        approved_per_pair = torch.sum(
            approved.view(pairs["n_pairs"], self.n_indep_sims), dim=1
        )
        # accumulate the symmetric acceptance/rejection matrices
        self.acceptance_matrix[
            pairs["replica_a"], pairs["replica_b"]
        ] += approved_per_pair
        self.acceptance_matrix[pairs["replica_b"], pairs["replica_a"]] += (
            self.n_indep_sims - approved_per_pair
        )
        return pairs_for_exchange

    def _perform_exchange(
        self,
        data: AtomicData,
        pairs_for_exchange: Dict,
        forces: torch.Tensor,
    ) -> Tuple[AtomicData, torch.Tensor]:
        r"""Exchanges the coordinates, velcities and forces for those pairs marked for exchange.
        Exchanged velocities are rescaled based on ratios of beta values from the two configurations.
        A configuration leaving the replica at :math:`\beta_{old}` and entering the replica at
        :math:`\beta_{new}` carries velocities equilibrated at :math:`\beta_{old}`, so they are
        rescaled by:

        .. math::

            vscale = \sqrt{\frac{\beta_{old}}{\beta_{new}}}

        Parameters
        ----------
        data:
            Collated AtomicData instance containing the current Cartesian coordinates and
            velocities for each simulation/replica
        pairs_for_exchange:
            Dictionary that denotes which pairs have been accepted for exchange
        forces:
            Forces of the current positions. All replicas share the same molecule and
            potential, so exchanging configurations only permutes the forces; passing
            them here keeps them consistent with the new positions without an extra
            model evaluation.

        Returns
        -------
        AtomicData:
            The updated collated atomic data where the coordinates and (rescaled) velocities
            have been exchanged according to the appropriate supplied exchange pairs
        torch.Tensor:
            The forces, permuted in the same way as the coordinates
        """
        pair_a, pair_b = pairs_for_exchange["a"], pairs_for_exchange["b"]
        if len(pair_a) == 0 and len(pair_b) == 0:
            return data, forces

        # Column of the exchange log. With an exchange_interval that is a
        # multiple of save_interval (see the warning in __init__) this
        # subroutine runs right after the frame of the current step was saved,
        # and `sim_t // save_interval` is the index of that frame. Unlike
        # `(sim_t + 1) // save_interval - 1` it is never negative, which would
        # otherwise wrap the record around to the end of the array whenever an
        # exchange happens before the first frame is saved.
        save_t_idx = self.sim_t // self.save_interval
        exchange_parity = 2 if self._propose_even_pairs else 1
        self.exchange_arr[pair_a.cpu(), save_t_idx] = exchange_parity
        self.exchange_arr[pair_b.cpu(), save_t_idx] = -exchange_parity

        # Coordinates, velocities and forces are stored atom-by-atom for all
        # replicas at once. Grouping them per simulation lets every accepted
        # pair be swapped by a single fancy-index assignment, instead of
        # building one boolean mask over all atoms per pair.
        per_sim = (self.n_sims, self.n_atoms, self.n_dims)

        # exchange the coordinates
        pos = data[POSITIONS_KEY].detach().clone().reshape(per_sim)
        pos_a, pos_b = pos[pair_a], pos[pair_b]
        pos[pair_a], pos[pair_b] = pos_b, pos_a
        data[POSITIONS_KEY] = pos.reshape(-1, self.n_dims)

        # scale and exchange the velocities: the velocities entering replica a
        # come from replica b, hence they are rescaled by sqrt(beta_b/beta_a)
        beta_a = self.beta[pair_a][:, None, None]
        beta_b = self.beta[pair_b][:, None, None]
        vscale_into_a = torch.sqrt(beta_b / beta_a)
        vscale_into_b = torch.sqrt(beta_a / beta_b)
        vel = data[VELOCITY_KEY].detach().clone().reshape(per_sim)
        vel_a, vel_b = vel[pair_a], vel[pair_b]
        vel[pair_a] = vel_b * vscale_into_a
        vel[pair_b] = vel_a * vscale_into_b
        data[VELOCITY_KEY] = vel.reshape(-1, self.n_dims)

        # exchange the forces: the integrator reuses the forces of the current
        # positions at the next step, so they must follow the configurations
        swapped_forces = forces.clone().reshape(per_sim)
        forces_a, forces_b = swapped_forces[pair_a], swapped_forces[pair_b]
        swapped_forces[pair_a], swapped_forces[pair_b] = forces_b, forces_a
        forces = swapped_forces.reshape(-1, self.n_dims)

        return data, forces

    def detect_and_exchange_replicas(
        self, data: AtomicData, forces: torch.Tensor
    ) -> Tuple[AtomicData, torch.Tensor]:
        """Subroutine for replica exchange: Modifies the internal coordinates and velocities
        according to the algorithm specified by `reform`:

        https://github.com/noegroup/reform

        Parameters
        ----------
        data:
            Current `AtomicData` instance containing all replicas, their coordinates, velocities,
            potential energies, and beta values
        forces:
            Forces of the current positions, permuted along with the exchanged replicas

        Returns
        -------
        data:
            Updated `AtomicData` instance containing potentially exchanged replicas.
        forces:
            Forces belonging to the updated positions.
        """
        pairs_for_exchange = self._detect_exchange(data)
        data, forces = self._perform_exchange(
            data, pairs_for_exchange, forces=forces
        )
        self._propose_even_pairs = not self._propose_even_pairs
        return data, forces

    def save_exchanges(self, data: AtomicData, save_step: int) -> None:
        """Save routine to record the ratio of acceptances/attempts for each temperature during the simulation.
        After saving to file, the acceptances/attempts are reset. For this particular method, the AtomicData
        and save_step are not used, though they are included as arguments for the sake of saving
        """
        # `write` has already advanced the numpy file counter by the time this
        # runs, so step back by one to label these files like the trajectory
        # chunk they belong to
        key = "{:04d}".format(self._npy_file_index - 1)
        np.save(
            "{}_acceptance_{}.npy".format(self.filename, key),
            self.acceptance_matrix.detach().cpu().numpy(),
        )
        exchanges = self.exchange_arr[:, self._old_save_step : save_step]
        # trajectory chunks are always exported with `_save_size` frames, zero
        # padded for the last, partial one; pad here as well so that the
        # exchange log stays aligned with the exported frames column by column
        if exchanges.shape[1] < self._save_size:
            exchanges = torch.nn.functional.pad(
                exchanges, (0, self._save_size - exchanges.shape[1])
            )
        np.save(
            "{}_exchanges_{}.npy".format(self.filename, key),
            exchanges,
        )
        # Reset
        self._old_save_step = save_step
        self._reset_acceptance_matrix()

    def summary(self):
        attempted = int(self._replica_exchange_attempts)
        exchanged = int(self._replica_exchange_approved)
        printstring = "Done simulating ({})".format(time.asctime())
        # a simulation shorter than one exchange interval has no statistics
        if attempted == 0:
            printstring += "\nNo replica exchange was attempted."
        else:
            printstring += "\nReplica-exchange rate: %.2f%% (%d/%d)" % (
                exchanged / attempted * 100.0,
                exchanged,
                attempted,
            )
        printstring += (
            "\nNote that you can call .get_replica_info"
            "(#replica) to query the inverse temperature"
            " and trajectory indices for a given replica."
        )
        if self.log_type == "print":
            print(printstring)
        elif self.log_type == "write":
            printstring += "\n"
            with open(self._log_file, "a") as lfile:
                lfile.write(printstring)


# pipe the doc from the base class into the child class so that it's properly
# displayed by sphinx
PTSimulation.__doc__ += _Simulation.__doc__
