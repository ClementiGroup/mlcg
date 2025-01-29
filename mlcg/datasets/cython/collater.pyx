cimport cython
cimport numpy as np
from libc.string cimport memcpy
import numpy as np
import torch
from .batch_helper import batch_collate_w_nls, batch_collate_w_nls_w_exc_pair
from .decoy_helper import check_and_sum_decoy_prob, pick_decoy_frames, decoy_frame_

# init numpy
np.import_array()
ctypedef np.npy_intp intp # this is the actual int type in Python

# MLCG keys
from typing import Final, Union, List, Dict, Callable
FORCE_KEY: Final[str] = "forces"

class MockBatch(dict):
    """Mocking a collated Batch. We are not using the generic pytorch geometric
    `data.batch.Batch` structure for the sake of simplicity and speed.
    (based on https://stackoverflow.com/a/14620633)
    """
    def __init__(self, *args, **kwargs):
        super(MockBatch, self).__init__(*args, **kwargs)
        self.__dict__ = self
    
    def to(self, *args, **kwargs):
        tensor_dict_to_(self, *args, **kwargs)
        return self

    def cuda(self, device=None, *args, **kwargs):
        device = "cuda" if device is None else device
        tensor_dict_to_(self, device=device, *args, **kwargs)
        return self

    def cpu(self, *args, **kwargs):
        tensor_dict_to_(self, device="cpu", *args, **kwargs)
        return self

def tensor_dict_to_(tensor_dict, *args, **kwargs):
    for k, v in tensor_dict.items():
        if isinstance(v, dict):
            tensor_dict[k] = tensor_dict_to_(v, *args, **kwargs)
        else:
            tensor_dict[k] = v.to(*args, **kwargs)
    return tensor_dict

def array_dict_to_torch(arr_dict):
    out_dict = {}
    for k, v in arr_dict.items():
        if isinstance(v, dict):
            out_dict[k] = array_dict_to_torch(v)
        else:
            out_dict[k] = torch.from_numpy(v)
    return out_dict

class CythonCollater:
    def __init__(
        self,
        transform: Union[Callable, None] = None,
        baseline_models: Union[str, Dict[str, torch.Module], None] = None,
        remove_neighbor_list: bool = True,
        decoy_options: Union[List[Dict[str, float]], None] = None,
        exclude_bonded_pairs: bool = False,
        device: str = "cuda",
    ):
        self.transform = transform
        self._remove_neighbor_list = remove_neighbor_list
        self.device = device
        if isinstance(baseline_models, str):
            _baseline_models = torch.load(baseline_models)
        else:
            _baseline_models = baseline_models
        if _baseline_models is not None:
            if hasattr(_baseline_models, "models"):
                # unwrap from a SumOut module
                _baseline_models = _baseline_models.models
            self.baseline_models = _baseline_models.to(self.device)
        else:
            self.baseline_models = None
        if decoy_options is not None:
            self._decoy_opts = decoy_options
            self._summary_decoy_probs = check_and_sum_decoy_prob(decoy_options)
        else:
            self._decoy_opts = None
        self._moved_to_device = False
        self._exclude_bonded_pairs = exclude_bonded_pairs

    def __call__(self, np_data_batch: Iterable[AtomicData]) -> MockBatch:
        if not self._moved_to_device:
            # copy the params over to the current device
            # since we noticed that the self.device can change after init in ddp
            if self.baseline_models:
                self.baseline_models.to(self.device)
            self._moved_to_device = True
        # collation happens here
        if self._exclude_bonded_pairs:
            collated_data_np = batch_collate_w_nls_w_exc_pair(np_data_batch, transform=self.transform)
        else:
            collated_data_np = batch_collate_w_nls(np_data_batch, transform=self.transform)
        # pick the decoy indices here over the numpy data
        if self._decoy_opts:
            _, frame_ranges = pick_decoy_frames(collated_data_np, self._summary_decoy_probs)
        collated_data = MockBatch()
        collated_data.update(array_dict_to_torch(collated_data_np))
        collated_data.out = {}
        collated_data.to(self.device)
        # compute and remove the prior forces
        corrected_data = self._collated_remove_baseline(collated_data)  # type: ignore
        # zero-out the decoy delta forces
        if self._decoy_opts:
            for opt, ranges in zip(self._decoy_opts, frame_ranges):
                for start, stop in ranges:
                    decoy_frame_(corrected_data, opt["scale"], start, stop)
        return corrected_data

    def _collated_remove_baseline(
        self, collated_frames: MockBatch
    ) -> MockBatch:
        """Remove baseline from forces in the collated AtomicData.

        This may modify the input AtomicData instances.
        """
        if self.baseline_models is None:
            return collated_frames
        else:
            # remove baseline (collated)
            collated_forces = collated_frames[FORCE_KEY]
            baseline_forces = torch.zeros_like(collated_forces, requires_grad=False)
            for k, model in self.baseline_models.items():
                model.eval()
                with torch.enable_grad():
                    collated_frames = model(collated_frames)
                baseline_forces += collated_frames.out[k][FORCE_KEY].detach()
                del collated_frames.out[k]
            collated_forces -= baseline_forces
            if self._remove_neighbor_list:
                collated_frames.neighbor_list = {}
            return collated_frames
