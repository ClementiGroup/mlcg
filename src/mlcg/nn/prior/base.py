import torch
from ...data import AtomicData


def compute_cell_shifts(
    pos: torch.Tensor,
    mapping: torch.Tensor,
    pbc: torch.Tensor,
    cell: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the minimum vector using index 0 as reference
        Scale vectors based on box size and shift if greater than half the box size
        Initial implementation written by Clark Templeton
        Adopted from ase.geometry naive_find_mic
            https://gitlab.com/ase/ase/
    Inputs:
        pos: (n_coords_over_frames x 3(x,y,z))
            positions from AtomicData object
        mapping: (order_map x n_mapping)
            index mapping from AtomicData object
            order_map = 2,3,4,etc. for harmonic, angle, dihedral, etc.
        pbc: (frames x 3)
            whether to apply cell shift in this dimension
        cell: (frames x 3 x 3)
            unit cell
        batch: (n_mapping)
            which frame corresponds to each mapping
    Returns:
        cell_shifts: (n_mapping x 3(x,y,z) x order_map)
            Integer values of how many unit cells to shift for minimum image convention
                based on the first index in mapping
            First column is all zeros by convention as shift to self
    """

    # Must wrap with no grad in order to avoid error when passing through forward
    with torch.no_grad():
        atom_groups, mapping_order = mapping.T.shape[:2]
        cell_shifts = torch.zeros(
            atom_groups, 3, mapping_order, dtype=pos.dtype
        ).to(pos.device)
        if batch == None:
            batch = torch.zeros(pos.shape[0], dtype=int)
        batch_ids = batch[mapping[0]]
        cell_inv = torch.linalg.inv(cell[batch_ids])
        for ii in range(1, cell_shifts.shape[-1]):
            drs = pos[mapping[0]] - pos[mapping[ii]]
            # convert to fractional displacement
            frac_dr = torch.einsum(
                "bij,bj->bi",
                cell_inv.to(drs.dtype),
                drs,
            )
            # compute unit number of unit cell shifts
            cell_shifts[:, :, ii] = torch.floor(frac_dr + 0.5)
            # convert back to cartesian displacement
            cell_shifts[:, :, ii] = pbc[batch_ids] * torch.einsum(
                "bij,bj->bi",
                cell[batch_ids].to(drs.dtype),
                cell_shifts[:, :, ii],
            )
    return cell_shifts


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

    @staticmethod
    def _get_cell_shifts(
        pos: torch.Tensor,
        mapping: torch.Tensor,
        pbc: torch.Tensor,
        cell: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Wrapper method to compute cell shifts using periodic boundary conditions.

        This is a convenience wrapper around the compute_cell_shifts function.

        """
        if all([feat != None for feat in [pbc, cell]]):
            cell_shifts = compute_cell_shifts(pos, mapping, pbc, cell, batch)
        else:
            cell_shifts = None
        return cell_shifts
