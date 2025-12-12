from typing import Dict, Mapping, Optional
import torch
from .torch_impl import torch_neighbor_list


def atomic_data2neighbor_list(
    data,
    rcut: float,
    self_interaction: bool = False,
    max_num_neighbors: int = 1000,
) -> Dict:
    """Build a neighborlist from a :ref:`mlcg.data.atomic_data.AtomicData` by
    searching for neighboring atom within a maximum radius `rcut`.

    Parameters
    ----------
    data: :ref:`mlcg.data.atomic_data.AtomicData`
        define an atomic structure
    rcut:
        cutoff radius used to compute the connectivity
    self_interaction:
        whether the mapping includes self referring mappings, e.g. mappings where `i` == `j`.
    max_num_neighbors:
        The maximum number of neighbors to return for each atom in :obj:`data`.
        If the number of actual neighbors is greater than
        :obj:`max_num_neighbors`, returned neighbors are picked randomly.

    Returns
    -------
    Dict:
        Neighborlist dictionary
    """
    rcut = float(rcut)
    idx_i, idx_j, cell_shifts, _ = torch_neighbor_list(
        data,
        rcut,
        self_interaction=self_interaction,
        max_num_neighbors=max_num_neighbors,
    )

    mapping = torch.cat([idx_i.unsqueeze(0), idx_j.unsqueeze(0)], dim=0)
    order = mapping.shape[0]
    if "batch" not in data:
        batch = torch.zeros(
            data.pos.shape[0], dtype=torch.long, device=data.pos.device
        )
    else:
        batch = data.batch

    assert torch.all(
        batch[mapping[0]] == batch[mapping[1]]
    ), "torch_neighbor_list() returned pairs of indices in different structures"

    mapping_batch = batch[mapping[0]]
    return make_neighbor_list(
        tag=f"nonbounded rc:{rcut} order:{order}",
        order=order,
        index_mapping=mapping,
        mapping_batch=mapping_batch,
        cell_shifts=cell_shifts,
        rcut=rcut,
        self_interaction=self_interaction,
    )


def make_neighbor_list(
    tag: str,
    order: int,
    index_mapping: torch.Tensor,
    mapping_batch: Optional[torch.Tensor] = None,
    cell_shifts: Optional[torch.Tensor] = None,
    rcut: Optional[float] = None,
    self_interaction: Optional[bool] = None,
) -> Dict:
    """Build a neighbor_list dictionary.

    Parameters
    ----------
    tag:
        quick identifier for compatibility checking
    order:
        an int providing the order of the neighborlist, e.g. order == 2 means that
        central atoms `i` have 1 neighbor `j` so distances can be computed,
        order == 3 means that central atoms `i` have 2 neighbors `j` and `k` so
        angles can be computed
    index_mapping:
        The [order, n_edge] index tensor giving center -> neighbor relations. 1st column
        refers to the central atom index and the 2nd column to the neighbor atom
        index in the list of atoms (so it has to be shifted by a cell_shift to get
        the actual position of the neighboring atoms)
    mapping_batch:
        [n_edge] map for the neighbor -> structure index relation
    cell_shifts:
        A [n_edge, 3] tensor giving the periodic cell shift
    rcut:
        cutoff radius used to compute the connectivity
    self_interaction:
        whether the mapping includes self referring mappings, e.g. mappings where `i` == `j`.

    Returns
    -------
    Dict:
        Neighborlist dictionary
    """
    if index_mapping.shape[0] != order:
        raise RuntimeError(
            f"index_mapping shape does not match the order:{index_mapping.shape[0]} != {order}"
        )
    if mapping_batch is None:
        if len(index_mapping) > 0:
            mapping_batch = torch.zeros(
                (index_mapping.shape[1]), dtype=torch.long
            )
        else:
            mapping_batch = torch.zeros((0,), dtype=torch.long)
    return dict(
        tag=tag,
        order=order,
        index_mapping=torch.as_tensor(index_mapping, dtype=torch.long),
        cell_shifts=cell_shifts,
        rcut=rcut,
        self_interaction=self_interaction,
        mapping_batch=mapping_batch,
    )


class _EmptyField:
    pass


def validate_neighborlist(inp: Dict) -> bool:
    """Tool to validate that the neighborlist dictionary has the required fields

    Parameters
    ----------
    inp:
        Input neighborlist to be validated

    Returns
    -------
    bool:
        True if the supplied neighborlist is valid, false otherwise
    """
    validator = {
        "tag": [str],
        "order": [int],
        "index_mapping": [torch.Tensor],
        "cell_shifts": [torch.Tensor],
        "rcut": [float],
        "self_interaction": [bool],
    }
    is_validated = False
    if isinstance(inp, Mapping):
        vals = []
        # check that entries of validator exists and that the value had one of
        #  the expected type
        for k, types in validator.items():
            v = inp.get(k, _EmptyField())
            vals.append(any([isinstance(v, t) for t in types]))

        if all(vals):
            is_validated = True
    return is_validated
