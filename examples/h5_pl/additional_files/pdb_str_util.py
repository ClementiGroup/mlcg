import io
import mdtraj as md

__all__ = ["read_from_pdb", "load_mdtraj_from_str"]


def read_from_pdb(filepath):
    """Read the whole pdbfile to a string."""
    with open(filepath) as f:
        pdbdata = f.read()
    return pdbdata


# the code below is modified from mdtraj
from mdtraj.utils import cast_indices, in_units_of


def load_mdtraj_from_str(
    pdbstr,
    stride=None,
    atom_indices=None,
    frame=None,
    no_boxchk=False,
    standard_names=True,
    top=None,
):
    """Load a RCSB Protein Data Bank file from a string in memory.
    Parameters
    ----------
    pdbstr : string-like
        String that contains the whole PDB file
    stride : int, default=None
        Only read every stride-th model from the file
    atom_indices : array_like, default=None
        If not None, then read only a subset of the atoms coordinates from the
        file. These indices are zero-based (not 1 based, as used by the PDB
        format). So if you want to load only the first atom in the file, you
        would supply ``atom_indices = np.array([0])``.
    frame : int, default=None
        Use this option to load only a single frame from a trajectory on disk.
        If frame is None, the default, the entire trajectory will be loaded.
        If supplied, ``stride`` will be ignored.
    no_boxchk : bool, default=False
        By default, a heuristic check based on the particle density will be
        performed to determine if the unit cell dimensions are absurd. If the
        particle density is >1000 atoms per nm^3, the unit cell will be
        discarded. This is done because all PDB files from RCSB contain a CRYST1
        record, even if there are no periodic boundaries, and dummy values are
        filled in instead. This check will filter out those false unit cells and
        avoid potential errors in geometry calculations. Set this variable to
        ``True`` in order to skip this heuristic check.
    standard_names : bool, default=True
        If True, non-standard atomnames and residuenames are standardized to conform
        with the current PDB format version. If set to false, this step is skipped.
    top : mdtraj.core.Topology, default=None
        if you give a topology as input the topology won't be parsed from the pdb file
        it saves time if you have to parse a big number of files
    Returns
    -------
    trajectory : md.Trajectory
        The resulting trajectory, as an md.Trajectory object.
    Examples
    --------
    >>> import mdtraj as md
    >>> pdb = md.load_pdb('2EQQ.pdb')
    >>> print(pdb)
    <mdtraj.Trajectory with 20 frames, 423 atoms at 0x110740a90>
    See Also
    --------
    mdtraj.PDBTrajectoryFile : Low level interface to PDB files
    """
    from mdtraj import Trajectory

    atom_indices = cast_indices(atom_indices)

    with PDBTrajFromString(pdbstr, standard_names=standard_names, top=top) as f:
        atom_slice = slice(None) if atom_indices is None else atom_indices
        if frame is not None:
            coords = f.positions[[frame], atom_slice, :]
        else:
            coords = f.positions[::stride, atom_slice, :]
        assert coords.ndim == 3, "internal shape error"
        n_frames = len(coords)

        topology = f.topology
        if atom_indices is not None:
            # The input topology shouldn't be modified because
            # subset makes a copy inside the function
            topology = topology.subset(atom_indices)

        if f.unitcell_angles is not None and f.unitcell_lengths is not None:
            unitcell_lengths = np.array([f.unitcell_lengths] * n_frames)
            unitcell_angles = np.array([f.unitcell_angles] * n_frames)
        else:
            unitcell_lengths = None
            unitcell_angles = None

        in_units_of(
            coords, f.distance_unit, Trajectory._distance_unit, inplace=True
        )
        in_units_of(
            unitcell_lengths,
            f.distance_unit,
            Trajectory._distance_unit,
            inplace=True,
        )

    time = np.arange(len(coords))
    if frame is not None:
        time *= frame
    elif stride is not None:
        time *= stride

    traj = Trajectory(
        xyz=coords,
        time=time,
        topology=topology,
        unitcell_lengths=unitcell_lengths,
        unitcell_angles=unitcell_angles,
    )

    if not no_boxchk and traj.unitcell_lengths is not None:
        # Only one CRYST1 record is allowed, so only do this check for the first
        # frame. Some RCSB PDB files do not *really* have a unit cell, but still
        # have a CRYST1 record with a dummy definition. These boxes are usually
        # tiny (e.g., 1 A^3), so check that the particle density in the unit
        # cell is not absurdly high. Standard water density is ~55 M, which
        # yields a particle density ~100 atoms per cubic nm. It should be safe
        # to say that no particle density should exceed 10x that.
        particle_density = traj.top.n_atoms / traj.unitcell_volumes[0]
        if particle_density > 1000:
            warnings.warn(
                "Unlikely unit cell vectors detected in PDB file likely "
                "resulting from a dummy CRYST1 record. Discarding unit "
                "cell vectors.",
                category=UserWarning,
            )
            traj._unitcell_lengths = traj._unitcell_angles = None

    return traj


class PDBTrajFromString(md.formats.PDBTrajectoryFile):
    def __init__(self, pdbstring, standard_names=True, top=None):
        self._open = False
        self._file = None
        self._topology = top
        self._positions = None
        self._mode = "r"
        self._last_topology = None
        self._standard_names = standard_names

        # only implement for mode "r"
        md.formats.PDBTrajectoryFile._loadNameReplacementTables()
        self._file = io.StringIO(pdbstring)
        self._read_models()

        self._open = True
