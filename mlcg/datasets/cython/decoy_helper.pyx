cimport cython
cimport numpy as np
from libc.string cimport memcpy
import numpy as np
import torch

# init numpy
np.import_array()
ctypedef np.npy_intp intp # this is the actual int type in Python


def check_and_sum_decoy_prob(opts):
    """
    Sanity check: the probabilities are not exceeding 1
    Example of expected input:
    [
        {
            "scale": 0.5,
            "prob": 0.02,
        },
        {
            "scale": 5.0,
            "prob": 0.02,
        }
    ]
    Example output:
    [0.02, 0.02, 0.96]
    """
    summary_decoy_probs = []
    for option in opts:
        if (
            "scale" not in option
            or option["scale"] <= 0
            or "prob" not in option
            or option["prob"] < 0
        ):
            raise ValueError(f"Invalid decoy option: {option}")
        summary_decoy_probs.append(option["prob"])
    if sum(summary_decoy_probs) > 1:
        raise ValueError(
            f"total decoy probability exceeds 1: {summary_decoy_probs}"
        )
    summary_decoy_probs.append(1 - sum(summary_decoy_probs))
    return summary_decoy_probs

def pick_decoy_frames(collated_data_np, summary_decoy_probs):
    """Decide the number of frames picked for each case, the last one corresponds to the
    number of intact samples. Unlike the general version, this cython version works on
    the collated np version (output of `batch_collate(_w_nls)`) to avoid back-and-forth 
    transforms.
    """
    n_frames = len(collated_data_np["n_atoms"])
    n_picks = np.random.multinomial(n_frames, summary_decoy_probs)[:-1]
    shuffled_ids = np.arange(n_frames)
    np.random.shuffle(shuffled_ids)
    shuffled_ids = shuffled_ids.tolist()
    n_picks_sum = np.cumsum(n_picks).tolist()
    id_picks = [
        shuffled_ids[a:b] for a, b in zip([0] + n_picks_sum, n_picks_sum)
    ]
    ptr = collated_data_np["ptr"]
    frame_ranges = [
        [(ptr[id_], ptr[id_ + 1]) for id_ in id_pick] for id_pick in id_picks
    ]
    return id_picks, frame_ranges


def decoy_frame_(data, scale, frame_start, frame_stop):
    """Make decoy frames in place.
    1. add (huge amount of) noises to the corresponding frames in the `pos` tensor
    2. set the corresponding frames in the `forces` tensor to zero
    such that the NN learns to rely on priors instead of very wrong extrapolations
    """
    frame_coords = data.pos[frame_start:frame_stop]
    frame_coords += torch.randn_like(frame_coords) * scale
    data.forces[frame_start:frame_stop] = 0

