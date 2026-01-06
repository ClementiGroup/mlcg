import torch
from ase.geometry.analysis import Analysis
from ase.data import atomic_numbers

# dictionary of atomic numbers to element name
ase_z2name = {value: key for key, value in atomic_numbers.items()}


def ase_bonds2tensor(analysis: Analysis, unique=True) -> torch.Tensor:
    """converts ASE single neighborlist analysis bond list to tensor

    Parameters
    ----------
    analysis:
        ASE Analysis instance
    unique:
        If True, onlye unqiue bonds will be considered. If False,
        the resulting tensor will also contain redundant (backwards)
        bond pairs

    .. warning::
        The minimum image convention is applied in `Analysis`.

    Returns
    -------
    edge_tensor:
        Tensor of edges defining the bonds, of shape (2, n_bonds)
    """

    if unique:
        bond_list = analysis.unique_bonds
    else:
        bond_list = analysis.all_bonds
    edge_tensor = torch.tensor([[], []])
    for atom, neighbors in enumerate(bond_list[0]):
        for bonded_neighbor in neighbors:
            edge_tensor = torch.cat(
                (edge_tensor, torch.tensor([[atom], [bonded_neighbor]])), dim=1
            )
    return edge_tensor.long()


def ase_angles2tensor(analysis: Analysis, unique=True) -> torch.Tensor:
    """converts ASE single neighborlist analysis angle list to tensor

    Parameters
    ----------
    analysis:
        ASE Analysis instance
    unique:
        If True, onlye unqiue bonds will be considered. If False,
        the resulting tensor will also contain redundant (backwards)
        bond pairs

    .. warning::
        The minimum image convention is applied in `Analysis`.

    Returns
    -------
    edge_tensor:
        Tensor of edges definiing the bonds, of shape (2, n_bonds)
    """

    if unique:
        angle_list = analysis.unique_angles
    else:
        angle_list = analysis.all_angles
    edge_tensor = torch.tensor([[], [], []])
    for atom, end_point_list in enumerate(angle_list[0]):
        for end_points in end_point_list:
            edge_tensor = torch.cat(
                (
                    edge_tensor,
                    torch.tensor([[end_points[0]], [atom], [end_points[1]]]),
                ),
                dim=1,
            )
    return edge_tensor.long()
