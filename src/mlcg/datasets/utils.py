import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset
from typing import Dict, List, Sequence, Union, TextIO, Optional, Tuple
from mlcg.data.atomic_data import AtomicData
import mdtraj as md
import numpy as np

from ..data._keys import FORCE_KEY


def chunker(seq: Sequence, size: int) -> Sequence:
    """Helper function for working with chunks of a specified size
    for a given sequence"""
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def remove_baseline_forces(
    data_list: List[AtomicData], models: Dict[str, torch.nn.Module]
) -> List[AtomicData]:
    """Compute the forces on the input :obj:`data_list` with the :obj:`models`
    and remove them from the reference forces contained in :obj:`data_list`.
    The computation of the forces is done on the whole :obj:`data_list` at once
    so it should not be too large.

    Parameters
    ----------
    data_list:
        Uncollated list of AtomicData instances that contain the full
        reference forces
    models:
        Pytorch ModuleDict containing models that compute prior/baseline
        forces

    Returns
    -------
    List[AtomicData]:
        Uncollated list of AtomicData instances, where the value of the
        'forces' field is now the delta forces (original forces minus
        the baseline/prior forces). An additional field 'baseline' forces
        is added, whose value is equal to the baseline/prior forces
    """

    n_frame = len(data_list)
    dataloader = DataLoader(data_list, batch_size=n_frame)
    baseline_forces = []
    for data in dataloader:
        for k in models.keys():
            models[k].eval()

            data = models[k](data)
            baseline_forces.append(data.out[k][FORCE_KEY].flatten())
            # make sure predicted properties don't require gradient anymore
            for key, v in data.out[k].items():
                data.out[k][key] = v.detach()
    baseline_forces = torch.sum(torch.vstack(baseline_forces), dim=0).view(
        -1, 3
    )

    for i_frame in range(n_frame):
        mask = data.batch == i_frame
        data_list[i_frame].forces -= baseline_forces[mask]
        data_list[i_frame].baseline_forces = baseline_forces[mask]

    return data_list


def write_PDB(
    dataset: InMemoryDataset,
    frame: int = 0,
    fout: str = "cg.pdb",
):
    """
    Given a mlcg dataset containing both data and topologies attributes,
    write out trajectory in PDB for a particular frame More or less a copy
    of mdtraj PDBReporter but explicit writing of bond connection

    Parameters
    ----------
    dataset:
        mlcg InMemory dataset instance
    frame:
        Specifies the frame with which a structure should be saved
    fout:
        Name of the saved PDB file.
    """

    mol_id = dataset.data.name[frame]
    topology = dataset.topologies[mol_id].to_mdtraj()
    n_atoms = topology.n_atoms
    cg_traj = md.Trajectory(
        dataset.data.pos[
            int(frame * n_atoms) : int((frame + 1) * n_atoms)
        ].numpy(),
        topology,
    )
    chains = [chain for chain in topology.chains]
    bfactors = ["{0:5.2f}".format(0.0)] * cg_traj.xyz.shape[1]
    # file = open(fout, "w")
    with open(fout, "w") as file:
        _write_header(file)
        atomIndex = 1
        posIndex = 0
        modelIndex = 0
        file.write("MODEL     %4d \n" % modelIndex)

        for chainIndex, chain in enumerate(topology.chains):
            chainName = chains[chainIndex].index
            # converts int to alphabet (0->a,1->b, etc.) to match convention
            chainName = chr(chainName + 97).upper()
            residues = list(chain.residues)
            for resIndex, res in enumerate(residues):
                if len(res.name) > 3:
                    resName = res.name[:3]
                else:
                    resName = res.name
                for atom in res.atoms:
                    if (
                        len(atom.name) < 4
                        and atom.name[:1].isalpha()
                        and (
                            atom.element is None or len(atom.element.symbol) < 2
                        )
                    ):
                        atomName = " " + atom.name
                    elif len(atom.name) > 4:
                        atomName = atom.name[:4]
                    else:
                        atomName = atom.name
                    coords = cg_traj.xyz[0, posIndex]
                    if atom.element is not None:
                        symbol = atom.element.symbol
                    else:
                        symbol = " "
                    if atom.serial is not None and len(topology._chains) < 2:
                        # We can't do this for more than 1 chain
                        # to prevent issue 1611
                        atomSerial = atom.serial
                    else:
                        atomSerial = atomIndex
                    line = (
                        "ATOM  %5d %-4s %3s %1s%4d    %s%s%s  1.00 %5s      %-4s%2s  "
                        % (  # Right-justify atom symbol
                            atomSerial % 100000,
                            atomName,
                            resName,
                            chainName,
                            (res.resSeq) % 10000,
                            _format_83(coords[0]),
                            _format_83(coords[1]),
                            _format_83(coords[2]),
                            bfactors[posIndex],
                            atom.segment_id[:4],
                            symbol[-2:],
                        )
                    )
                    assert len(line) == 80, "Fixed width overflow detected"
                    file.write(line + "\n")
                    posIndex += 1
                    atomIndex += 1
                if resIndex == len(residues) - 1:
                    file.write(
                        "TER   %5d      %3s %s%4d \n"
                        % (atomSerial + 1, resName, chainName, res.resSeq),
                    )
                    atomIndex += 1

        _write_footer(file, topology)
        return None


def _write_header(
    file: TextIO,
    unitcell_lengths: Optional[Tuple] = None,
    unitcell_angles: Optional[Tuple] = None,
    write_metadata: bool = True,
):
    """Write out the header for a PDB file.

    Parameters
    ----------
    file:
        Python file object
    unitcell_lengths:
        The lengths of the three unitcell vectors, ``a``, ``b``, ``c``
    unitcell_angles:
        The angles between the three unitcell vectors, ``alpha``,
        ``beta``, ``gamma``
    """

    if unitcell_lengths is None and unitcell_angles is None:
        return
    if unitcell_lengths is not None and unitcell_angles is not None:
        if not len(unitcell_lengths) == 3:
            raise ValueError("unitcell_lengths must be length 3")
        if not len(unitcell_angles) == 3:
            raise ValueError("unitcell_angles must be length 3")
    else:
        raise ValueError(
            "either unitcell_lengths and unitcell_angles"
            "should both be spefied, or neither"
        )

    box = list(unitcell_lengths) + list(unitcell_angles)
    assert len(box) == 6

    if write_metadata:
        file.write("REMARK   \n")
    file.write(
        "CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1           1 \n" % tuple(box),
    )


def _write_footer(file: TextIO, topology: Optional[md.Topology] = None):
    """Write out the footer for a PDB file

    Parameters
    ----------
    file:
        Python file object
    topology:
        MDTraj topology instance
    """

    conectBonds = []
    if topology is not None:
        for atom1, atom2 in topology.bonds:
            conectBonds.append((atom1, atom2))
    if len(conectBonds) > 0:
        # Work out the index used in the PDB file for each atom.

        atomIndex = {}
        nextAtomIndex = 0
        prevChain = None
        for chain in topology.chains:
            for atom in chain.atoms:
                if atom.residue.chain != prevChain:
                    nextAtomIndex += 1
                    prevChain = atom.residue.chain
                atomIndex[atom] = nextAtomIndex
                nextAtomIndex += 1

        # Record which other atoms each atom is bonded to.

        atomBonds = {}
        for atom1, atom2 in conectBonds:
            index1 = atomIndex[atom1]
            index2 = atomIndex[atom2]
            if index1 not in atomBonds:
                atomBonds[index1] = []
            if index2 not in atomBonds:
                atomBonds[index2] = []
            atomBonds[index1].append(index2)
            atomBonds[index2].append(index1)

        # Write the CONECT records.

        for index1 in sorted(atomBonds):
            bonded = atomBonds[index1]
            while len(bonded) > 4:
                file.write(
                    "CONECT%5d%5d%5d%5d \n"
                    % (index1, bonded[0], bonded[1], bonded[2]),
                )
                del bonded[:4]
            line = "CONECT%5d" % index1
            for index2 in bonded:
                line = "%s%5d" % (line, index2)
            file.write(line + "\n")
    file.write("END")


def _format_83(f):
    """Format a single float into a string of width 8, with ideally 3 decimal
    places of precision. If the number is a little too large, we can
    gracefully degrade the precision by lopping off some of the decimal
    places. If it's much too large, we throw a ValueError"""
    if -999.999 < f < 9999.999:
        return "%8.3f" % f
    if -9999999 < f < 99999999:
        return ("%8.3f" % f)[:8]
    raise ValueError(
        'coordinate "%s" could not be represnted ' "in a width-8 field" % f
    )


def write_PSF(
    dataset: InMemoryDataset,
    fout: str = "cg.psf",
    charges: Optional[np.ndarray] = None,
):
    """
    Write out charmm format psf file from AtomicData object

    Parameters
    ----------
    dataset:
        Mlcg dataset containing the molecule from which a PSF will be extracted
    fout:
        Output filename
    charges:
        Array of charges
    """
    with open(fout, "w") as file:
        file.write("PSF \n")
        cg_topo = dataset.topologies
        topology = cg_topo.to_mdtraj()
        n_atoms = topology.n_atoms

        # Add in dummy charge
        if charges == None:
            charges = ["{0:2.6f}".format(0.0)] * n_atoms
        seg_names = ["{:2s}".format("CG")] * n_atoms

        if hasattr(dataset.data, "masses"):
            masses = dataset.data.masses
        else:
            masses = ["{0:2.4f}".format(0)] * n_atoms

        # Write out atom information
        atom_index = 1
        pos_index = 0
        file.write("{:>8d} !NATOM \n".format(n_atoms))
        for _, chain in enumerate(topology.chains):
            # converts int to alphabet (0->a,1->b, etc.) to match convention
            residues = list(chain.residues)
            for _, res in enumerate(residues):
                for atom in res.atoms:
                    atom_type = atom.name
                    atom_name = atom.name
                    resid = atom.residue.index
                    resname = atom.residue.name
                    line = "{:>8d} {:<4s} {:<4d} {:<4} {:<4} {:<4} {} {:>2.4} 0 \n".format(
                        atom_index % 100000,
                        seg_names[pos_index],
                        resid,
                        resname,
                        atom_name,
                        atom_type,
                        charges[pos_index],
                        masses[pos_index],
                    )
                    file.write(line)
                    atom_index += 1
                    pos_index += 1

        # Write out bonding information
        properties = ["bonds", "angles", "dihedrals", "impropers"]
        titles = [
            "!NBOND: bonds",
            "!NTHETA: angles",
            "!NPHI: dihedrals",
            "!NIMPHI: impropers",
        ]
        items_per_line = [4, 3, 2, 2]

        for i_p, property in enumerate(properties):
            prop = getattr(cg_topo, property)
            prop = np.asarray(prop).T
            n_prop = len(prop)
            title = "{:>8d} ".format(n_prop) + titles[i_p]
            file.write(title + "\n")
            item_per_line = items_per_line[i_p]
            i_b = 0
            while i_b < n_prop:
                print_list = []
                for _ in range(item_per_line):
                    if i_b > n_prop - 1:
                        continue
                    for atom in prop[i_b]:
                        print_list.append(atom + 1)
                    i_b += 1
                string = ["{:>8d}".format(x) for x in print_list]
                string = "".join(string)
                file.write(string + "\n")


def get_exclusion_pairs_from_neighborlist(
    nl_dict, ignore_terms=("non_bonded", "gamma_1", "gamma_2")
):
    """Extract the exclusion pairs for a molecule given the neighbor list used.
    For term of order bigger than 2, extracting the pairs formed by the first and last bead.

    Input
    -----
    nl_dict: dict of neighbor list dictionaries, each of which contains an `index_mapping`
    ignore_terms: sequence of str, names of terms to be ignored, e.g., "non_bonded"

    Output
    ------
    int Tensor of shape (2, N_pairs) with all pairs extracted from the the neighbor list.
    """
    exc_pairs = []
    for term_name, term in nl_dict.items():
        if term_name not in ignore_terms:
            exc_pairs.append(term["index_mapping"][[0, -1]].T)
            exc_pairs.append(term["index_mapping"][[-1, 0]].T)
    exc_pairs = torch.unique(torch.cat(exc_pairs), dim=1).T
    return exc_pairs
