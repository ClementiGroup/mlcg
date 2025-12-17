import torch
from torch_geometric.data import Data
from typing import Optional, Dict, Any

from ._keys import (
    validate_keys,
    POSITIONS_KEY,
    N_ATOMS_KEY,
    CELL_KEY,
    PBC_KEY,
    ATOM_TYPE_KEY,
    NEIGHBOR_LIST_KEY,
    TAG_KEY,
    ENERGY_KEY,
    FORCE_KEY,
    MASS_KEY,
    VELOCITY_KEY,
)


class AtomicData(Data):
    """A data object holding atomic structures.
    The attribute names are defined in :py:data:`mlcg.data._keys`

    Attributes
    ----------

    pos: [n_atoms * n_structures, 3]
        set of atomic positions in each structures
    atom_types: [n_atoms * n_structures]
        if atoms then it's the atomic number, if it's a CG bead then it's a number defined by the CG mapping
    masses: [n_atoms * n_structures]
        Masses of each atom
    pbc: [n_structures, 3] (Optional)
        periodic boundary conditions
    cell: [n_structures, 3, 3] (Optional)
        unit cell of the atomic structure. Lattice vectors are defined row wise
    tag: [n_structures] (Optional)
        metadata about each structure
    energy: [n_structures] (Optional)
        reference energy associated with each structures
    forces: [n_atoms * n_structures, 3] (Optional)
        reference forces associated with each structures
    velocities: [n_atoms * n_structures, 3] (optional)
        velocities associated with each structure
    neighborlist: Dict[str, Dict[str, Any]] (Optional)
        contains information about the connectivity formatted according to
        :ref:`mlcg.neighbor_list.neighbor_list.make_neighbor_list`.
    batch: [n_atoms]
        maps the atoms to their structure index (from 0 to n_structures-1)
    """

    def __init__(
        self,
        **kwargs,
    ):
        """kwargs:
        Allowed fields are defined in :ref:`ALLOWED_KEYS`
        """
        validate_keys(kwargs.keys())

        super(AtomicData, self).__init__(**kwargs)

        self.out = {}

        # check the sanity of the inputs
        if N_ATOMS_KEY in self and POSITIONS_KEY in self:
            assert (
                torch.sum(self.n_atoms) == self.pos.shape[0]
            ), f"number of atoms {torch.sum(self.n_atoms)} and number of positions {self.pos.shape[0]}"

            assert self.pos.shape[1] == 3
        if MASS_KEY in self and POSITIONS_KEY in self:
            assert len(self.masses) == self.pos.shape[0]

        if CELL_KEY in self and self.cell is not None:
            assert self.cell.dim() == 3
            assert self.cell.shape[1:] == torch.Size((3, 3))
            assert self.cell.dtype == self.pos.dtype
        if FORCE_KEY in self and self[FORCE_KEY] is not None:
            assert self[FORCE_KEY].shape == self[POSITIONS_KEY].shape
            assert self[FORCE_KEY].dtype == self[POSITIONS_KEY].dtype
        if VELOCITY_KEY in self and self[VELOCITY_KEY] is not None:
            assert self[VELOCITY_KEY].shape == self[POSITIONS_KEY].shape
            assert self[VELOCITY_KEY].dtype == self[POSITIONS_KEY].dtype
        if ENERGY_KEY in self and self[ENERGY_KEY] is not None:
            assert self[ENERGY_KEY].shape == self[N_ATOMS_KEY].shape
            assert self[ENERGY_KEY].dtype == self[POSITIONS_KEY].dtype
        if PBC_KEY in self and self.pbc is not None:
            assert self.pbc.dim() == 2, f"dim {self.pbc.dim()}"
            assert self.pbc.shape[1:] == torch.Size(
                [3]
            ), f"shape {self.pbc.shape[1:]}"
            assert self.pbc.dtype == torch.bool

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        """Method for incrementing nested values during collation"""
        if "index" in key or "face" in key:
            return self.num_nodes
        elif "mapping_batch" == key:
            return 1
        else:
            return 0

    @staticmethod
    def from_ase(
        frame,
        energy_tag: str = ENERGY_KEY,
        force_tag: str = FORCE_KEY,
    ):
        r"""Method for constructing an AtomicData instance from
        an ASE Atoms instance. Atom types are taken from atomic
        numbers, while periodic flags, masses, cell vectors, and
        configuration tags are taken from the ASE Atoms attributes.

        Parameters
        ----------
        frame:
            Input ase.atom.Atoms instance
        energy_tag:
            String value used to access the ASE energy
        force_tag:
            String value used to access the ASE forces

        Returns
        -------
        data:
            Populated AtomicData instance
        """

        z = torch.from_numpy(frame.get_atomic_numbers())
        pos = torch.from_numpy(frame.get_positions())
        masses = torch.from_numpy(frame.get_masses())
        pbc = torch.from_numpy(frame.get_pbc())
        cell = torch.tensor(frame.get_cell().tolist(), dtype=pos.dtype)

        tag = frame.info.get("tag")

        energy = frame.info.get(energy_tag)
        forces = frame.arrays.get(force_tag)

        return AtomicData.from_points(
            pos=pos,
            atom_types=z,
            masses=masses,
            pbc=pbc,
            cell=cell,
            tag=tag,
            energy=energy,
            forces=forces,
        )

    @staticmethod
    def from_points(
        pos: torch.Tensor,
        atom_types: torch.Tensor,
        masses: Optional[torch.Tensor] = None,
        pbc: Optional[torch.Tensor] = None,
        cell: Optional[torch.Tensor] = None,
        tag: Optional[str] = None,
        energy: Optional[torch.Tensor] = None,
        forces: Optional[torch.Tensor] = None,
        velocities: Optional[torch.Tensor] = None,
        neighborlist: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs,
    ):
        r"""Method for constructing an AtomicData instance directly
        from input tensors

        Parameters
        ----------
        pos:
            Cartesian coordinates of a single configuration, of shape
            (n_atoms, 3)
        atom_types:
            Atom type labels of shape (n_atoms)
        masses:
            Atom masses of shape (n_atoms)
        pbc:
            Period boundary condition flags for the frame
        cell:
            Box vectors for the configuration, for computing periodic
            images, of shape (3,3)
        tag:
            String tag for the configuration
        energy:
            Reference scalar energy for the configuration
        forces:
            Reference cartesian forces for the configuration, of
            shape (n_atoms, 3)
        velocities:
            Atomic velocities, of shape (n_atoms, 3)
        neighborlist:
            Neighborlist dictionary for tagged features. Eg,

        Returns
        -------
        data:
            Populated AtomicData instance
        """

        data = {}
        data.update(**kwargs)
        data[ATOM_TYPE_KEY] = torch.as_tensor(atom_types)
        data[POSITIONS_KEY] = torch.as_tensor(pos)
        data[N_ATOMS_KEY] = torch.tensor(
            [data[ATOM_TYPE_KEY].shape[0]], dtype=torch.long
        )

        if masses is not None:
            assert all([m > 0 for m in masses])
            data[MASS_KEY] = torch.as_tensor(masses)
        if energy is not None:
            data[ENERGY_KEY] = torch.as_tensor(
                energy, dtype=data[POSITIONS_KEY].dtype
            )
        if forces is not None:
            data[FORCE_KEY] = torch.as_tensor(
                forces, dtype=data[POSITIONS_KEY].dtype
            )
        if velocities is not None:
            data[VELOCITY_KEY] = torch.as_tensor(
                velocities, dtype=data[POSITIONS_KEY].dtype
            )

        data[TAG_KEY] = tag
        if neighborlist is None:
            data[NEIGHBOR_LIST_KEY] = {}
        else:
            data[NEIGHBOR_LIST_KEY] = neighborlist

        if pbc is not None:
            data[PBC_KEY] = torch.as_tensor(pbc).view(-1, 3)
        if cell is not None:
            data[CELL_KEY] = torch.as_tensor(cell).view(-1, 3, 3)

        return AtomicData(**data)
