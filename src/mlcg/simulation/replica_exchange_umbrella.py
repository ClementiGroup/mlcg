"""Replica exchange between umbrella windows (Hamiltonian replica exchange,
REUS) for the native-contact Q restraint."""

from typing import List, Optional, Union
import time
import numpy as np
import torch

from ..data.atomic_data import AtomicData
from ..data._keys import POSITIONS_KEY, VELOCITY_KEY
from ..nn.prior.q_restraint import NativeContactQRestraint
from .base import _Simulation
from .langevin import LangevinSimulation


class REUSSimulation(LangevinSimulation):
    r"""Langevin umbrella sampling with replica exchange between windows.

    Every structure of the batch is one umbrella window of a
    :class:`NativeContactQRestraint` (per-structure ``Q0`` and ``k``), all at
    the same temperature. Every ``exchange_interval`` steps, windows that are
    neighbours in ``Q0`` try to swap configurations, alternating the pairs
    (0,1), (2,3), ... and (1,2), (3,4), ... (in ``Q0`` order). The unbiased
    model is the same in every window and cancels, so for windows :math:`i, j`
    holding configurations with :math:`Q_i, Q_j` and bias
    :math:`U_w(Q) = k_w (Q - Q_{0,w})^2`:

    .. math::

        Acc = \min\left(1, \exp\left(-\beta \left[U_i(Q_j) + U_j(Q_i)
              - U_i(Q_i) - U_j(Q_j)\right]\right)\right)

    An accepted swap exchanges positions and velocities; at a single
    temperature no velocity rescaling is needed. Forces are recomputed before
    the next step, since each configuration now feels the other window's bias.

    Saved frames stay ordered by window (structure ``w`` of the output is
    always window ``w``), so the umbrella/MBAR analysis is unchanged. In
    addition, at every export:

    - ``{filename}_replicas_{key}.npy``, shape ``(n_windows, n_frames)``: the
      replica (index of the starting structure) in each window at each saved
      frame, to follow a configuration through Q and count round trips.
    - ``{filename}_exchanges_{key}.npz``: ``window_a``, ``window_b`` (the
      neighbouring pairs, in ``Q0`` order), and ``accepted``, ``attempted``
      swaps of each pair during the export interval.

    The replica order and the even/odd parity are stored in the checkpoint,
    so a restarted run continues the same exchange sequence.

    Parameters
    ----------
    friction:
        Scalar friction to use for Langevin updates
    exchange_interval:
        Number of simulation steps between exchange attempts.
    restraint_name:
        Name of the :class:`NativeContactQRestraint` in the model (its key in
        ``data.neighbor_list``).
    """

    def __init__(
        self,
        friction: float = 1e-3,
        exchange_interval: int = 100,
        restraint_name: str = "native_contacts_q",
        **kwargs,
    ):
        super(REUSSimulation, self).__init__(
            friction=friction,
            sim_subroutine=self.detect_and_exchange_windows,
            sim_subroutine_interval=exchange_interval,
            **kwargs,
        )
        self.restraint_name = restraint_name
        self._forces_stale = False

    def _attach_model(self, model: torch.nn.Module):
        super()._attach_model(model)
        found = [
            m
            for m in self.model.modules()
            if isinstance(m, NativeContactQRestraint)
            and m.name == self.restraint_name
        ]
        if len(found) != 1:
            raise ValueError(
                f"REUSSimulation needs exactly one NativeContactQRestraint "
                f"named '{self.restraint_name}' in the model, found {len(found)}."
            )
        self.restraint = found[0]

    def _attach_configurations(
        self,
        configurations: List[AtomicData],
        beta: Union[float, List[float]],
    ):
        # the base class wipes the checkpoint once it has used it
        checkpoint = self.checkpointed_data
        super()._attach_configurations(configurations, beta)
        if not torch.all(self.beta == self.beta[0]):
            raise ValueError(
                "Replica exchange between umbrella windows needs a single "
                f"temperature, got betas {self.beta.tolist()}."
            )
        Q0 = self.restraint.Q0
        if Q0.numel() != self.n_sims:
            raise ValueError(
                f"The restraint has {Q0.numel()} windows for {self.n_sims} "
                "structures; REUS needs one structure per window."
            )
        if torch.unique(Q0).numel() != self.n_sims:
            raise ValueError("REUS needs a different Q0 in every window.")

        # windows sorted by Q0: pair p is (order[p], order[p + 1])
        order = torch.argsort(Q0).to(self.device)
        self._window_a, self._window_b = order[:-1], order[1:]
        if checkpoint is not None and "replica_of_window" in checkpoint:
            self.replica_of_window = checkpoint["replica_of_window"].to(
                self.device
            )
            self._propose_even_pairs = checkpoint["propose_even_pairs"]
        else:
            self.replica_of_window = torch.arange(
                self.n_sims, device=self.device
            )
            self._propose_even_pairs = True

    def _set_up_simulation(self, overwrite: bool = False):
        super()._set_up_simulation(overwrite)
        self.simulated_replicas = torch.zeros(
            (self._save_size, self.n_sims), dtype=torch.long
        )
        self._reset_exchange_stats()
        self._total_accepted = 0
        self._total_attempted = 0

    def _reset_exchange_stats(self):
        self._accepted = torch.zeros(
            self.n_sims - 1, dtype=torch.long, device=self.device
        )
        self._attempted = torch.zeros_like(self._accepted)

    def timestep(self, data: AtomicData, forces: torch.Tensor):
        # after a swap the forces of the previous step belong to the other window
        if self._forces_stale:
            _, forces = self.calculate_potential_and_forces(data)
            self._forces_stale = False
        return super().timestep(data, forces)

    @torch.no_grad()
    def window_bias(
        self, windows: torch.Tensor, q: torch.Tensor
    ) -> torch.Tensor:
        """Bias of `windows` evaluated at the Q values `q`, in kcal/mol."""
        k = self.restraint.k.to(q.dtype)[windows]
        Q0 = self.restraint.Q0.to(q.dtype)[windows]
        return k * (q - Q0) ** 2

    @torch.no_grad()
    def detect_and_exchange_windows(self, data: AtomicData) -> AtomicData:
        """Attempts swaps between the even or odd neighbouring windows and
        exchanges positions and velocities of the accepted ones."""
        pairs = torch.arange(
            0 if self._propose_even_pairs else 1,
            self.n_sims - 1,
            2,
            device=self.device,
        )
        self._propose_even_pairs = not self._propose_even_pairs
        a, b = self._window_a[pairs], self._window_b[pairs]
        q = self.restraint.batch_q(data).detach().to(torch.float64)
        delta = self.beta[a].to(torch.float64) * (
            self.window_bias(a, q[b])
            + self.window_bias(b, q[a])
            - self.window_bias(a, q[a])
            - self.window_bias(b, q[b])
        )
        u = torch.rand(
            len(pairs),
            generator=self.rng,
            device=self.device,
            dtype=torch.float64,
        )
        accepted = torch.log(u) < -delta
        self._accepted[pairs] += accepted
        self._attempted[pairs] += 1
        if not accepted.any():
            return data

        a, b = a[accepted], b[accepted]
        perm = torch.arange(self.n_sims, device=self.device)
        perm[a], perm[b] = b, a
        for key in (POSITIONS_KEY, VELOCITY_KEY):
            x = data[key].detach().view(self.n_sims, self.n_atoms, self.n_dims)
            data[key] = x[perm].reshape(-1, self.n_dims)
        self.replica_of_window = self.replica_of_window[perm]
        self._forces_stale = True
        return data

    def save(
        self,
        data: AtomicData,
        forces: torch.Tensor,
        potential: torch.Tensor,
        t: int,
    ):
        super().save(data, forces, potential, t)
        save_ind = (
            t // self.save_interval
        ) - self._npy_file_index * self._save_size
        self.simulated_replicas[save_ind] = self.replica_of_window.cpu()

    def write(self, data: Optional[AtomicData] = None):
        key = self._get_numpy_count()
        np.save(
            "{}_replicas_{}.npy".format(self.filename, key),
            self.simulated_replicas.t().numpy(),
        )
        np.savez(
            "{}_exchanges_{}.npz".format(self.filename, key),
            window_a=self._window_a.cpu().numpy(),
            window_b=self._window_b.cpu().numpy(),
            accepted=self._accepted.cpu().numpy(),
            attempted=self._attempted.cpu().numpy(),
        )
        self._total_accepted += int(self._accepted.sum())
        self._total_attempted += int(self._attempted.sum())
        self._reset_exchange_stats()
        self.simulated_replicas = torch.zeros(
            (self._save_size, self.n_sims), dtype=torch.long
        )
        super().write(data=data)
        # the base class builds and saves the checkpoint in write() (after this step's exchange): add the replica state
        # to it and save it again
        if self.create_checkpoints:
            self.checkpoint["replica_of_window"] = (
                self.replica_of_window.detach().clone()
            )
            self.checkpoint["propose_even_pairs"] = self._propose_even_pairs
            torch.save(
                self.checkpoint,
                "{}_checkpoint.pt".format(self.filename),
            )

    def summary(self):
        super().summary()
        printstring = "Window exchange rate: %.2f%% (%d/%d)" % (
            100.0 * self._total_accepted / max(self._total_attempted, 1),
            self._total_accepted,
            self._total_attempted,
        )
        if self.log_type == "print":
            print(printstring)
        elif self.log_type == "write":
            with open(self._log_file, "a") as lfile:
                lfile.write(printstring + "\n")


# pipe the doc from the base class into the child class so that it's properly
# displayed by sphinx
REUSSimulation.__doc__ += _Simulation.__doc__
