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


    Note: This implementation only allows for replica exchanges between directly
    adjecent temperatures implied by the user-supplied list of beta values.

    Parameters
    ----------
    friction:
        Scalar friction to use for Langevin updates
    exchange_interval:
        Specifies the number of simulation steps to take before attempting
        replica exchange.
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

        self._replica_exchange_approved = 0
        self._replica_exchange_attempts = 0

    def _reset_exchange_stats(self):
        """Setup function that resets exchange statistics before running a simulation"""
        self._replica_exchange_attempts = 0
        self._replica_exchange_approved = 0

    def _set_up_simulation(self, overwrite: bool = False):
        super(PTSimulation, self)._set_up_simulation(overwrite=overwrite)
        self._reset_exchange_stats()
        self.acceptance_matrix = torch.zeros(
            self.n_replicas, self.n_replicas
        ).to(self.device)

    def attach_model_and_configurations(
        self,
        model: torch.nn.Module,
        configurations: List[AtomicData],
        betas: List[float],
    ):
        if self.specialize_priors:
            new_configurations = self._attach_configurations(
                configurations, betas=betas
            )
            (
                model,
                condensed_configurations,
            ) = condense_all_priors_for_simulation(model, new_configurations)
            # Repeat attachment, this time with the condensed configurations
            self._manually_reattach_configurations(condensed_configurations)
            self._attach_model(model)
            print("Prior models have been specialized for the simulation.")
        else:
            self._attach_configurations(configurations, betas=betas)
            self._attach_model(model)

        if self.filename is not None:
            torch.save(
                (deepcopy(model), deepcopy(configurations)),
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
            self.initial_data[
                VELOCITY_KEY
            ] = LangevinSimulation.sample_maxwell_boltzmann(
                self.beta.repeat_interleave(self.n_atoms),
                self.initial_data[MASS_KEY],
            ).to(
                self.dtype
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
                    beta
                )
            )
        assert all([np.isfinite(b) for b in betas])
        if not (np.array(betas[::-1]) == np.sort(betas[::-1])).all():
            raise ValueError(
                "Betas must be in order of increasing temperature."
            )

        self.n_indep_sims = len(configurations)
        self.n_replicas = len(betas)
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
        self.initial_data[
            VELOCITY_KEY
        ] = LangevinSimulation.sample_maxwell_boltzmann(
            self.beta.repeat_interleave(self.n_atoms),
            self.initial_data[MASS_KEY],
        ).to(
            self.dtype
        )

        self.initial_data[MASS_KEY] = self.initial_data[MASS_KEY].to(self.dtype)
        self.initial_data[POSITIONS_KEY] = self.initial_data[POSITIONS_KEY].to(
            self.dtype
        )
        self.beta_mass_ratio = torch.sqrt(
            1.0
            / self.beta.repeat_interleave(self.n_atoms)
            / self.initial_data[MASS_KEY]
        )[:, None]

        # Setup alternating even/odd pair exchanges
        self._propose_even_pairs = True

        # (0, 1), (2, 3), ...
        even_pairs = [(i, i + 1) for i in torch.arange(self.n_replicas)[:-1:2]]
        # (1, 2), (3, 4), ...
        odd_pairs = [(i, i + 1) for i in torch.arange(self.n_replicas)[1:-1:2]]
        if len(odd_pairs) == 0:
            odd_pairs = even_pairs
        pair_a = []
        pair_b = []
        for pair in even_pairs:
            pair_a.append(
                torch.arange(self.n_indep_sims) + pair[0] * self.n_indep_sims
            )
            pair_b.append(
                torch.arange(self.n_indep_sims) + pair[1] * self.n_indep_sims
            )
        self._even_pairs = [torch.cat(pair_a), torch.cat(pair_b)]
        pair_a = []
        pair_b = []
        for pair in odd_pairs:
            pair_a.append(
                torch.arange(self.n_indep_sims) + pair[0] * self.n_indep_sims
            )
            pair_b.append(
                torch.arange(self.n_indep_sims) + pair[1] * self.n_indep_sims
            )
        self._odd_pairs = [torch.cat(pair_a), torch.cat(pair_b)]

        # maps pairs to beta idx for acceptance matrix updates
        self.pair_to_beta_idx = torch.arange(self.n_replicas).repeat_interleave(
            self.n_indep_sims
        )
        self.acceptance_matrix = torch.zeros(
            self.n_replicas, self.n_replicas
        ).to(self.device)

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
            "beta": self.betas[replica_num],
            "indices_in_the_output": indices,
        }

    def _get_proposed_pairs(self) -> List[torch.Tensor]:
        """Proposes the even and odd exchange pairs alternatively each time the
        _detect_exchange method is called. Exchanges can only happen between direcly adjacent
        temperatures defined by the user supplied beta series.

        Returns
        -------
        torch.Tensor:
            pair_a, all possible even/odd pairs across all of the simulations/replicas
        torch.Tensor:
            pair_b, all possible
        """
        if self._propose_even_pairs:
            self._propose_even_pairs = False
            return self._even_pairs
        else:
            self._propose_even_pairs = True
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
        pair_a, pair_b = self._get_proposed_pairs()
        u_a, u_b = data.out[ENERGY_KEY][pair_a], data.out[ENERGY_KEY][pair_b]
        betas_a, betas_b = self.beta[pair_a], self.beta[pair_b]
        beta_idx_a, beta_idx_b = (
            self.pair_to_beta_idx[pair_a],
            self.pair_to_beta_idx[pair_b],
        )
        # Assemble pairs of beta undergoing exchange - useful for indexing below
        beta_pairs = torch.unique(torch.stack((beta_idx_a, beta_idx_b)), dim=1)
        p_pair = torch.exp((u_a - u_b) * (betas_a - betas_b)).cpu()
        approved = torch.rand(len(p_pair)) < p_pair
        num_approved = torch.sum(approved)
        num_attempted = len(pair_a)
        self._replica_exchange_approved += num_approved
        self._replica_exchange_attempts += num_attempted
        pairs_for_exchange = {"a": pair_a[approved], "b": pair_b[approved]}

        # Count the number of approved for each temperature pair
        approved_per_beta = torch.sum(
            approved.reshape(len(torch.unique(betas_a)), self.n_indep_sims),
            dim=1,
        ).to(self.device)
        # accumulate the symmetric acceptance/rejection matrices
        self.acceptance_matrix[
            beta_pairs[0, :], beta_pairs[1, :]
        ] += approved_per_beta
        self.acceptance_matrix[beta_pairs[1, :], beta_pairs[0, :]] += (
            self.n_indep_sims - approved_per_beta
        )
        return pairs_for_exchange

    def _perform_exchange(
        self, data: AtomicData, pairs_for_exchange: Dict
    ) -> AtomicData:
        """Exchanges the coordinates and velcities for those pairs marked for exchange.
        Exchanged velocities are rescaled based on ratios of beta values from the two configurations.
        For a pair of configurations :math:`A` and :math:`B`, characterized by the respective
        potential energies :math:`U_A` and :math:`U_B` the the inverse thermodynamic temperatures
        :math:`\beta_A` and :math:`\beta_B`, the the velocity exchange rescaling factor is:

        .. math::

            vscale = \sqrt{\frac{\beta_{old}}{\beta_{new}}}

        Parameters
        ----------
        data:
            Collated AtomicData instance containing the current Cartesian coordinates and
            velocities for each simulation/replica
        pairs_for_exchange:
            Dictionary that denotes which pairs have been accepted for exchange

        Returns
        -------
        AtomicData:
            The updated collated atomic data where the coordinates and (rescaled) velocities
            have been exchanged according to the appropriate supplied exchange pairs
        """
        pair_a, pair_b = pairs_for_exchange["a"], pairs_for_exchange["b"]
        # exchange the coordinates
        # Here we must make swaps in the coordinates and velocities
        # according to to the collated batch attribute
        if len(pair_a) == 0 and len(pair_b) == 0:
            return data
        else:
            batch_pair_a_cond = False
            batch_pair_b_cond = False
            for idx_a, idx_b in zip(pair_a, pair_b):
                # cumulative bitwise OR to grab corresponding pair batch index
                batch_pair_a_cond |= data.batch == idx_a
                batch_pair_b_cond |= data.batch == idx_b

            # exchange coordinates
            x_changed = data[POSITIONS_KEY].detach().clone()
            x_changed[batch_pair_a_cond] = data[POSITIONS_KEY][
                batch_pair_b_cond
            ]
            x_changed[batch_pair_b_cond] = data[POSITIONS_KEY][
                batch_pair_a_cond
            ]

            # scale and exchange the velocities
            # reshape the betas for simpler elementwise multiplication
            betas_a = self.beta[pair_a].repeat_interleave(self.n_atoms)[:, None]
            betas_b = self.beta[pair_b].repeat_interleave(self.n_atoms)[:, None]
            vscale_a_to_b = torch.sqrt(betas_a / betas_b)
            vscale_b_to_a = torch.sqrt(betas_b / betas_a)
            v_changed = data[VELOCITY_KEY].detach().clone()
            v_changed[batch_pair_a_cond] = (
                data[VELOCITY_KEY][batch_pair_b_cond] * vscale_a_to_b
            )
            v_changed[batch_pair_b_cond] = (
                data[VELOCITY_KEY][batch_pair_a_cond] * vscale_b_to_a
            )

            data[POSITIONS_KEY] = x_changed
            data[VELOCITY_KEY] = v_changed
            return data

    def detect_and_exchange_replicas(self, data: AtomicData) -> AtomicData:
        """Subroutine for replica exchange: Modifies the internal coordinates and velocities
        according to the algorithm specified by `reform`:

        https://github.com/noegroup/reform

        Parameters
        ----------
        data:
            Current `AtomicData` instance containing all replicas, their coordinates, velocities,
            potential energies, and beta values

        Returns
        -------
        data:
            Updated `AtomicData` instance containing potentially exchanged replicas.
        """
        pairs_for_exchange = self._detect_exchange(data)
        data = self._perform_exchange(data, pairs_for_exchange)
        return data

    def save_exchanges(self, data: AtomicData, save_step: int) -> None:
        """Save routine to record the ratio of acceptances/attempts for each temperature during the simulation.
        After saving to file, the acceptances/attempts are reset. For this particular method, the AtomicData
        and save_step are not used, though they are included as arguments for the sake of saving
        """
        key = self._get_numpy_count()
        np.save(
            "{}_acceptance_{}.npy".format(self.filename, key),
            self.acceptance_matrix.detach().cpu().numpy(),
        )
        # Reset
        self.acceptance_matrix = torch.zeros(
            self.n_replicas, self.n_replicas
        ).to(self.device)

    def summary(self):
        attempted = self._replica_exchange_attempts
        exchanged = self._replica_exchange_approved
        printstring = "Done simulating ({})".format(time.asctime())
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
