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

# cdef collate_nl_index_mapping(intp[:] index_mapping_ptrs, intp[:] n_, intp n_frames, ):
#     cdef intp offset = 0
#     for i_frame in range(n_frames):
#         # copy over the index mappings

cdef collate_nl_index_mapping(intp[::1] index_mapping_ptrs, intp[::1] n_ids, intp[::1] n_atoms, intp order, intp[:, ::1] out_id_map, intp[::1] out_map_batch, intp n_frames):
    cdef intp id_tuple_offset = 0, atom_offset = 0, i_frame, i_order, i_id
    for i_frame in range(n_frames):
        index_mapping_ptr = <intp*>index_mapping_ptrs[i_frame]
        for i_order in range(order):
            for i_id in range(n_ids[i_frame]):
                out_id_map[i_order, id_tuple_offset + i_id] = index_mapping_ptr[n_ids[i_frame] * i_order + i_id] + atom_offset
                if i_order == 0:
                    out_map_batch[id_tuple_offset + i_id] = i_frame
        id_tuple_offset += n_ids[i_frame]
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
    collate_nl_index_mapping(index_mapping_ptrs, n_ids, in_n_atoms, term_order, id_mapping, map_batch, n_frames)
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

def batch_collate_w_nls(list_of_atomic_data, transform=None):
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
    return collated

        
