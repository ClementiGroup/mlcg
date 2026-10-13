from typing import Dict
import torch
from torch_geometric.utils import scatter

from .base import _Prior
from ...data.atomic_data import AtomicData
from ...neighbor_list.neighbor_list import make_neighbor_list
from ...geometry.internal_coordinates import compute_distances


def _as_real(x) -> torch.Tensor:
    """Tensor-ise without downcasting: a hard ``.float()`` here would
    permanently lose precision for a ``dtype: double`` simulation."""
    t = torch.as_tensor(x)
    return t if t.is_floating_point() else t.float()


def _check_per_structure(buf: torch.Tensor, num_graphs: int, name: str) -> None:
    """A buffer of the wrong length would broadcast over the batch instead of
    erroring, silently collapsing a whole ladder onto one window."""
    if buf.numel() != num_graphs:
        raise ValueError(
            f"{name} has {buf.numel()} entries but the batch holds "
            f"{num_graphs} structures; it needs one value per structure."
        )


class NativeContactQRestraint(_Prior):
    r"""Harmonic restraint on the fraction of native contacts :math:`Q`.

    Each native contact pair :math:`(i,j)` contributes a smooth indicator
    (Best-Hummer-Eaton switching function):

    .. math::

        q_{ij} = \frac{1}{1 + \exp\left(\beta_Q (r_{ij} - \lambda_Q r^0_{ij})\right)}

    and the restrained collective variable is the per-structure mean over all
    native contacts:

    .. math::

        Q = \frac{1}{N_c}\sum_{ij} q_{ij}

    biased with :math:`U(Q) = k (Q - Q_0)^2`. Only the aggregate :math:`Q` is
    restrained; individual contacts remain free to break/form.

    Note that :math:`Q = 1` is unreachable, since :math:`q_{ij} < 1` for any
    finite :math:`r_{ij}`, so a ladder should stop around :math:`Q_0 = 0.95`.

    :math:`Q_0` and :math:`k` are per-structure buffers, so a batch of
    parallel replicas (e.g. an umbrella-sampling Q ladder) can each carry
    their own target and spring constant.

    Parameters
    ----------
    r0:
        Native distance of each contact pair for a single structure, shape
        ``(n_contacts,)``, in the same length units as ``data.pos``
        (Angstrom for the standard mlcg CG models). Every replica in the
        batch shares the same topology/native contacts, so this is tiled
        across replicas internally at forward time.
    Q0:
        Target Q for each structure of the collated batch, shape
        ``(n_structures,)`` -- for ``n_chains`` copies of an ``n_windows``
        ladder that is all ``n_chains * n_windows`` values, not just the
        windows.
    k:
        Spring constant(s) in kcal/mol; a single value (used for every
        structure) or shape ``(n_structures,)``. Q then fluctuates with width
        ``sqrt(kT / 2k)``, which is what has to be matched to the window
        spacing for MBAR to have overlap between neighbours.
    beta_q:
        Steepness of the switching function, in inverse units of
        ``data.pos``. The default 1.0 Angstrom^-1 is the same switching
        function as ``beta = 10`` nm^-1 in ``traj_rmsd_q_analysis.py``, so
        the biased Q is the analysed Q. Stored as a buffer, so it survives a
        ``state_dict`` round-trip.
    lambda_q:
        Tolerance factor multiplying the native distance (default 1.5, as
        in ``traj_rmsd_q_analysis.py``). Also a buffer.
    name:
        Key under which this term is registered in ``data.neighbor_list`` and
        ``data.out``; must match the name used to build the neighbor list via
        :meth:`neighbor_list`.
    """

    def __init__(
        self,
        r0: torch.Tensor,
        Q0: torch.Tensor,
        k: torch.Tensor,
        beta_q: float = 1.0,
        lambda_q: float = 1.5,
        name: str = "native_contacts_q",
    ) -> None:
        super().__init__()
        self.name = name

        Q0 = _as_real(Q0)
        k = _as_real(k)
        # a single k means "the same spring everywhere"; anything else has to be
        # one per structure, checked against the batch at forward time
        if k.numel() == 1:
            k = k.reshape(()).expand_as(Q0).clone()

        self.register_buffer("r0", _as_real(r0))
        self.register_buffer("Q0", Q0)
        self.register_buffer("k", k)
        # buffers, not plain floats: they define the collective variable, so a
        # state_dict has to carry them
        self.register_buffer("beta_q", _as_real(beta_q))
        self.register_buffer("lambda_q", _as_real(lambda_q))

    def data2features(self, data: AtomicData) -> torch.Tensor:
        mapping = data.neighbor_list[self.name]["index_mapping"]
        pbc = getattr(data, "pbc", None)
        cell = getattr(data, "cell", None)
        cell_shifts = self._get_cell_shifts(
            data.pos, mapping, pbc, cell, data.batch
        )
        return compute_distances(data.pos, mapping, cell_shifts)

    def forward(self, data: AtomicData) -> AtomicData:
        Q = self.batch_q(data)
        energy = self.k.to(Q.dtype) * (Q - self.Q0.to(Q.dtype)) ** 2
        data.out[self.name] = {"energy": energy}
        return data

    def batch_q(self, data: AtomicData) -> torch.Tensor:
        """The restrained :math:`Q` of every structure of a collated batch,
        shape ``(n_structures,)``; used by :meth:`forward` and by replica
        exchange between windows."""
        mapping_batch = data.neighbor_list[self.name]["mapping_batch"]
        r_ij = self.data2features(data)
        # an un-collated single structure carries no ptr, and is one structure
        num_graphs = data.ptr.numel() - 1 if hasattr(data, "ptr") else 1
        _check_per_structure(self.Q0, num_graphs, "Q0")
        _check_per_structure(self.k, num_graphs, "k")
        # every replica shares the same native contacts; r0 covers one
        # structure, so tile it to match the collated batch of replicas
        n_contacts = self.r0.shape[0]
        if r_ij.shape[0] != num_graphs * n_contacts:
            raise ValueError(
                f"got {r_ij.shape[0]} contact distances for {num_graphs} "
                f"structures and {n_contacts} native contacts: every structure "
                "in the batch must carry the same native contact list."
            )
        r0 = self.r0.to(r_ij.dtype).repeat(num_graphs)
        # sigmoid(-x) == 1 / (1 + exp(x)): q_ij -> 1 when r_ij << lambda_q * r0 (contact formed)
        q_ij = torch.sigmoid(-self.beta_q * (r_ij - self.lambda_q * r0))
        return scatter(
            q_ij, mapping_batch, dim=0, reduce="mean", dim_size=num_graphs
        )

    def compute_q(
        self, pos: torch.Tensor, mapping: torch.Tensor
    ) -> torch.Tensor:
        """The restrained :math:`Q` evaluated on plain trajectory frames.

        Lets an analysis recompute the biased Q with the simulation's own
        ``r0``/``beta_q``/``lambda_q``. Distances are taken directly, with no
        minimum-image convention, so unlike :meth:`forward` this is only valid
        without periodic boundaries.

        Parameters
        ----------
        pos:
            Positions of one structure per leading index, shape
            ``(..., n_beads, 3)``, in the same length units as ``self.r0``.
        mapping:
            Contact pairs for a *single* structure, shape ``(2, n_contacts)``
            -- the same mapping passed to :meth:`neighbor_list`.

        Returns
        -------
        torch.Tensor:
            Q per structure, shape ``pos.shape[:-2]``.
        """
        i, j = mapping
        d = torch.linalg.norm(pos[..., i, :] - pos[..., j, :], dim=-1)
        q_ij = torch.sigmoid(
            -self.beta_q * (d - self.lambda_q * self.r0.to(d.dtype))
        )
        return q_ij.mean(dim=-1)

    @staticmethod
    def compute_native_distances(
        pos_ref: torch.Tensor, mapping: torch.Tensor
    ) -> torch.Tensor:
        """Native contact distances from a folded reference structure."""
        return compute_distances(pos_ref, mapping, cell_shifts=None)

    @staticmethod
    def neighbor_list(
        mapping: torch.Tensor, name: str = "native_contacts_q"
    ) -> Dict:
        """Build the ``data.neighbor_list`` entry for a fixed set of contact pairs.

        Parameters
        ----------
        mapping:
            Contact pairs, shape ``(2, n_contacts)``, dtype long.
        name:
            Key to store the neighbor list under; must match the ``name``
            passed to :class:`NativeContactQRestraint`.
        """
        nl = make_neighbor_list(
            tag=name, order=2, index_mapping=mapping, self_interaction=False
        )
        return {name: nl}
