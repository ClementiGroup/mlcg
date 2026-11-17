cimport cython
cimport numpy as np
from libc.string cimport memcpy
import numpy as np

# init numpy
np.import_array()
ctypedef np.npy_intp intp # this is the actual int type in Python


@cython.boundscheck(False)
def make_batch(n_elements):
    cdef intp i, i_elements, n_element, i_element_offset = 0
    out_batch = np.empty(sum(n_elements), dtype=int)
    cdef intp[::1] batch = out_batch
    out_ptr = np.empty(len(n_elements) + 1, dtype=int)
    cdef intp[::1] ptr = out_ptr
    ptr[0] = 0
    for i, n_element in enumerate(n_elements):
        for i_elements in range(i_element_offset, i_element_offset + n_element):
            batch[i_elements] = i
        ptr[i + 1] = ptr[i] + n_element
        i_element_offset += n_element
    return out_batch, out_ptr


@cython.wraparound(False)   # Deactivate negative indexing.
@cython.boundscheck(False)
def collate_bool_array_2d(list_of_arrays):
    cdef intp i, n_dim = list_of_arrays[0].shape[0]
    m = len(list_of_arrays)
    out_target = np.empty((m, n_dim), dtype=np.bool_)
    cdef np.ndarray[np.npy_bool, ndim=2, mode="c"] target = out_target
    cdef np.ndarray[np.npy_bool, ndim=2, mode="c"] source
    # cdef float[:][:] sources = list_of_arrays
    #for i, source_p in enumerate(list_of_arrays):
    for i in range(m):
        source = list_of_arrays[i].reshape(-1,3)
        #target[i_element_offset:(i_element_offset + n_elements[i])] = source[:]
        #source = source_p
        #memcpy(<float*>&target[0, 0] + i_element_offset * n_dim, <float*>&source[0, 0], n_elements[i] * n_dim * sizeof(float))
        memcpy(<void*>&target[i, 0], <const void*>&source[0, 0],  n_dim * sizeof(np.npy_bool))
    return out_target

@cython.wraparound(False)   # Deactivate negative indexing.
@cython.boundscheck(False)
def collate_float_array_2d(list_of_arrays):
    cdef intp i, i_element_offset = 0, n_dim = list_of_arrays[0].shape[1]
    n_elements = [len(arr) for arr in list_of_arrays]
    out_target = np.empty((sum(n_elements), n_dim), dtype=np.float32)
    cdef np.ndarray[float, ndim=2, mode="c"] target = out_target
    cdef np.ndarray[float, ndim=2, mode="c"] source
    # cdef float[:][:] sources = list_of_arrays
    for i, source_p in enumerate(list_of_arrays):
        source = source_p
        #target[i_element_offset:(i_element_offset + n_elements[i])] = source[:]
        #source = source_p
        #memcpy(<float*>&target[0, 0] + i_element_offset * n_dim, <float*>&source[0, 0], n_elements[i] * n_dim * sizeof(float))
        memcpy(<float*>&target[i_element_offset, 0], <float*>&source[0, 0], n_elements[i] * n_dim * sizeof(float))
        i_element_offset += n_elements[i]
    return out_target



@cython.wraparound(False)   # Deactivate negative indexing.
@cython.boundscheck(False)
def collate_float_array_3d(list_of_arrays):
    """
    Collate m arrays of shape (1,3,3) into one array of shape (m,3,3),
    dtype float32.
    """
    cdef intp i, m
    cdef intp offset = 0  # counts how many (1,3,3) blocks written
    cdef intp n_dim = list_of_arrays[0].shape[1]
    cdef intp n_dim_sq = n_dim * n_dim
    m = len(list_of_arrays)
    out = np.empty((m, n_dim, n_dim), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] target = out
    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] source
    for i in range(m):
        source = list_of_arrays[i].reshape(-1,3,3)

        # Copy 1*3*3 floats = 9 floats into target[i,:,:]
        memcpy(<float*>&target[i, 0, 0],
               <float*>&source[0, 0, 0],
               n_dim_sq * sizeof(np.float32_t))
        offset += 1
    return out


def batch_collate(list_of_atomic_data):
    n_frames = len(list_of_atomic_data)
    n_atoms = np.empty(n_frames, dtype=int)
    all_pos = []
    all_atom_types = []
    all_forces = []
    for i_frame, frame in enumerate(list_of_atomic_data):
        all_pos.append(frame["pos"])
        this_atom_types = frame["atom_types"]
        n_atoms[i_frame] = len(this_atom_types)
        all_atom_types.append(this_atom_types)
        all_forces.append(frame["forces"])
    batch, ptr = make_batch(n_atoms)
    collated = {
        "pos": collate_float_array_2d(all_pos),
        "atom_types": np.concatenate(all_atom_types),
        "n_atoms": n_atoms,
        "forces": collate_float_array_2d(all_forces),
        "batch": batch,
        "ptr": ptr,
    }
    return collated


cdef collate_edge_indices(
    intp[::1] edge_array_ptrs,
    intp[::1] n_edges,
    intp[::1] n_atoms,
    intp order,
    intp[:, ::1] out_edge_array,
    intp[::1] out_map_batch,
    intp n_frames,
):
    """Cython cdef function for collate edge (or hyperedges) index lists (e.g.,
    index_mapping under each neighbor list term). The major task of this function
    is to build a joint list of all edges in the frames of a batch, with the graph
    node (atom) indices properly offsetted according to the order of frame in the
     batch.
    
    Arguments
    ---------
    edge_array_ptrs: (pointer to) an array (of length `n_frames`) of pointers, 
        each pointing to the edge index storage of the corresponding frame. Each
        edge index storage should have shape [order, n_edges_i], where `i` is the
        index of frame in the batch
    n_edges: (pointer to) an array (of length `n_frames`), each corresponding to
        the number of edges for that frame
    n_atoms: (pointer to) an array (of length `n_frames`), each corresponding to
        the number of atoms (graph nodes) for that frame
    order: number of nodes in a (hyper-)edge. For normal edges please input `2`
    out_edge_array: (pointer to) a [output] array of type intp, which should have
        shape [order, sum(n_edges)]. To receive the collated edge lists
    out_map_batch: (pointer to) a [output] array of type intp, which should have 
        shape [sum(n_edges)]. To receive the index of frame in the batch for each
        edge in the collated output
    n_frames: number of frames in the batch
    """
    cdef intp id_tuple_offset = 0, atom_offset = 0, i_frame, i_order, i_id
    for i_frame in range(n_frames):
        edge_array_ptr = <intp*>edge_array_ptrs[i_frame]
        for i_order in range(order):
            for i_id in range(n_edges[i_frame]):
                out_edge_array[i_order, id_tuple_offset + i_id] = edge_array_ptr[n_edges[i_frame] * i_order + i_id] + atom_offset
                if i_order == 0:
                    out_map_batch[id_tuple_offset + i_id] = i_frame
        id_tuple_offset += n_edges[i_frame]
        atom_offset += n_atoms[i_frame]

def get_terms_in_nls(nl_example, excludes=("fmap",)):
    term_orders = []
    for term_name, term in nl_example.items():
        if term_name not in excludes:
            term_orders.append((term_name, term["order"]))
    return term_orders

def collate_nl_term(list_of_nls, n_atoms, term_name, term_order):
    n_frames = len(list_of_nls)
    collated_term = {}
    cdef intp i_frame
    cdef intp[::1] index_mapping_ptrs = np.empty(n_frames, dtype=int)
    cdef intp[::1] n_ids = np.empty(n_frames, dtype=int)
    cdef np.ndarray[intp, ndim=2, mode="c"] source
    for i_frame, nl in enumerate(list_of_nls):
        source = nl[term_name]["index_mapping"]
        if source.size > 0:
            index_mapping_ptrs[i_frame] = <intp>&source[0, 0]
        n_ids[i_frame] = source.shape[1]
    n_id_tuples = sum(n_ids)
    out_id_mapping = np.empty((term_order, n_id_tuples), dtype=int)
    out_map_batch = np.empty(n_id_tuples, dtype=int)
    cdef intp[:, ::1] id_mapping = out_id_mapping
    cdef intp[::1] map_batch = out_map_batch
    cdef intp[::1] in_n_atoms = n_atoms
    collate_edge_indices(index_mapping_ptrs, n_ids, in_n_atoms, term_order, id_mapping, map_batch, n_frames)
    collated_term["index_mapping"] = out_id_mapping
    collated_term["mapping_batch"] = out_map_batch
    return collated_term

def collate_nls(list_of_nls, n_atoms, excludes=("fmap",)):
    # assuming all having the same nls
    # figure out which terms there are
    term_orders = get_terms_in_nls(list_of_nls[0], excludes=excludes)
    collated_nls = {}
    for term_name, order in term_orders:
        collated_nls[term_name] = collate_nl_term(list_of_nls, n_atoms, term_name, order)
    return collated_nls

def collate_index_pairs(list_of_pairs, n_atoms):
    """Collate per-frame `(2, n_pairs_i)` node index pairs (e.g., excluded
    pairs) into one `(2, sum_i n_pairs_i)` array, offsetting the indices of
    each frame by the number of atoms in the preceding frames."""
    n_frames = len(list_of_pairs)
    cdef intp i_frame
    cdef intp[::1] pair_ptrs = np.empty(n_frames, dtype=int)
    cdef intp[::1] n_pairs = np.empty(n_frames, dtype=int)
    cdef np.ndarray[intp, ndim=2, mode="c"] source
    # contiguous intp copies (only if needed); the list keeps them alive
    # while their raw pointers are used below
    sources = [np.ascontiguousarray(p, dtype=np.intp) for p in list_of_pairs]
    for i_frame, pairs in enumerate(sources):
        if pairs.ndim != 2 or pairs.shape[0] != 2:
            raise ValueError(
                f"index pairs of frame {i_frame} must have shape (2, n_pairs), "
                f"got {pairs.shape}"
            )
        source = pairs
        if pairs.size > 0:
            pair_ptrs[i_frame] = <intp>&source[0, 0]
        n_pairs[i_frame] = pairs.shape[1]
    sum_n_pairs = sum(n_pairs)
    out_pairs = np.empty((2, sum_n_pairs), dtype=int)
    # the per-pair frame index is not returned, but is required by the helper
    out_map_batch = np.empty(sum_n_pairs, dtype=int)
    cdef intp[:, ::1] out_pairs_c = out_pairs
    cdef intp[::1] map_batch = out_map_batch
    cdef intp[::1] in_n_atoms_c = n_atoms
    collate_edge_indices(pair_ptrs, n_pairs, in_n_atoms_c, 2, out_pairs_c, map_batch, n_frames)
    return out_pairs


def _has_key_in_all_or_none(list_of_atomic_data, key):
    """Whether the optional `key` is present (and not None) in the frames,
    raising if only some of the frames of the batch carry it."""
    present = [frame.get(key) is not None for frame in list_of_atomic_data]
    if any(present) and not all(present):
        raise ValueError(
            f"`{key}` is present in only some frames of the batch "
            f"({sum(present)}/{len(present)})"
        )
    return all(present)


def collate_excluded_pairs(list_of_excluded_pairs, n_atoms):
    """Collate `excluded_pairs` dicts `{excluded_order, index_mapping}` the same
    way PyG collates them for `AtomicData`: `index_mapping` is concatenated with
    node offsets and `excluded_order` becomes one entry per frame."""
    return {
        "excluded_order": np.array(
            [exc["excluded_order"] for exc in list_of_excluded_pairs],
            dtype=np.int64,
        ),
        "index_mapping": collate_index_pairs(
            [exc["index_mapping"] for exc in list_of_excluded_pairs], n_atoms
        ),
    }


def collate_noise_levels(list_of_noise_levels, n_atoms):
    """Concatenate the per-bead noise levels of each frame into float32."""
    out = []
    for i_frame, noise_levels in enumerate(list_of_noise_levels):
        noise_levels = np.asarray(noise_levels, dtype=np.float32).reshape(-1)
        if len(noise_levels) != n_atoms[i_frame]:
            raise ValueError(
                f"frame {i_frame} has {len(noise_levels)} noise levels for "
                f"{n_atoms[i_frame]} atoms"
            )
        out.append(noise_levels)
    return np.concatenate(out)


def batch_collate_w_nls(list_of_atomic_data, transform=None, with_pbc=False):
    """Collate a list of frames (dicts of numpy arrays) into a batch.

    Always collated: `pos`, `forces`, `atom_types`, `n_atoms`,
    `neighbor_list`, `batch` and `ptr`. Optional fields:

    - `cell` and `pbc`: if `with_pbc` is True (required in every frame).
    - `excluded_pairs`: if the frames carry it, collated with node offsets.
    - `noise_levels`: if the frames carry it (after `transform`), as a
      per-bead float32 array.

    Optional fields must be present in all frames of the batch or in none.
    """
    n_frames = len(list_of_atomic_data)
    n_atoms = np.empty(n_frames, dtype=int)
    all_pos = []
    all_atom_types = []
    all_forces = []
    all_nls = []
    for i_frame, frame in enumerate(list_of_atomic_data):
        if transform is not None:
            transform(frame)
        all_pos.append(frame["pos"])
        this_atom_types = frame["atom_types"]
        n_atoms[i_frame] = len(this_atom_types)
        all_atom_types.append(this_atom_types)
        all_forces.append(frame["forces"])
        all_nls.append(frame["neighbor_list"])
    batch, ptr = make_batch(n_atoms)
    collated = {
        "pos": collate_float_array_2d(all_pos),
        "atom_types": np.concatenate(all_atom_types),
        "n_atoms": n_atoms,
        "forces": collate_float_array_2d(all_forces),
        "neighbor_list": collate_nls(all_nls, n_atoms),
        "batch": batch,
        "ptr": ptr,
    }
    if with_pbc:
        for key in ("cell", "pbc"):
            if not _has_key_in_all_or_none(list_of_atomic_data, key):
                raise ValueError(f"`with_pbc` is set but the frames have no `{key}`")
        collated["cell"] = collate_float_array_3d(
            [frame["cell"] for frame in list_of_atomic_data]
        )
        collated["pbc"] = collate_bool_array_2d(
            [frame["pbc"] for frame in list_of_atomic_data]
        )
    if _has_key_in_all_or_none(list_of_atomic_data, "excluded_pairs"):
        collated["excluded_pairs"] = collate_excluded_pairs(
            [frame["excluded_pairs"] for frame in list_of_atomic_data], n_atoms
        )
    if _has_key_in_all_or_none(list_of_atomic_data, "noise_levels"):
        collated["noise_levels"] = collate_noise_levels(
            [frame["noise_levels"] for frame in list_of_atomic_data], n_atoms
        )
    return collated


def batch_collate_w_nls_w_pbc(list_of_atomic_data, transform=None):
    """Same as `batch_collate_w_nls` with `with_pbc=True`."""
    return batch_collate_w_nls(
        list_of_atomic_data, transform=transform, with_pbc=True
    )
