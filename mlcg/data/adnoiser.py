"""Noiser."""

from typing import Union, Callable, Tuple, Final, Dict, Iterable, List, Any, Optional
from copy import copy as shallow_copy
from copy import deepcopy
import pickle
import numpy as np
from numpy import ndarray, asarray
from torch_geometric.transforms.base_transform import BaseTransform
from torch_geometric.loader.dataloader import Collater as PyGCollater
from torch import (
    enable_grad,
    Tensor,
    as_tensor,
    tensor,
    load,
    stack,
    float32,
    zeros_like,
)
from torch.nn import Module
import torch
import math
from mlcg.data import AtomicData
from mlcg.datasets.utils import remove_baseline_forces

# these should be imported from MLCG*.keys
POSITIONS_KEY: Final[str] = "pos"
FORCE_KEY: Final[str] = "forces"

_pftransform = Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]


class _AggforceFramewiseWrapper:
    """Wraps map_arrays method from an aggforce.TrajectoryMap instance.

    Call method operates on tensors representing single frames. Returned tensors
    are float32.
    """

    def __init__(
        self, mapping: Callable[[ndarray, ndarray], Tuple[ndarray, ndarray]], /
    ) -> None:
        self.mapping = mapping

    def __call__(
        self, positions: Tensor, forces: Tensor
    ) -> Tuple[Tensor, Tensor]:
        results = self.mapping(
            asarray(positions)[None, ...], asarray(forces)[None, ...]
        )
        positions = as_tensor(results[0][0], dtype=float32)
        forces = as_tensor(results[1][0], dtype=float32)
        return (positions, forces)


class _AggforceWrapper:
    """Wraps map_arrays method from an aggforce.TrajectoryMap instance.

    Call method operates on tensors representing chunks. Returned tensors
    are float32.
    """

    def __init__(
        self, mapping: Callable[[ndarray, ndarray], Tuple[ndarray, ndarray]], /
    ) -> None:
        self.mapping = mapping

    def __call__(
        self, positions: Tensor, forces: Tensor
    ) -> Tuple[Tensor, Tensor]:
        results = self.mapping(asarray(positions), asarray(forces))
        positions = as_tensor(results[0], dtype=float32)
        forces = as_tensor(results[1], dtype=float32)
        return (positions, forces)


def _aggforce_framewise_wrapper(
    c: Callable[[ndarray, ndarray], Tuple[ndarray, ndarray]], /
) -> _pftransform:
    """Wrap function which maps numpy trajectory chunks to one mapping Tensor frames.

    For example, if an array of shape (5,3) is passed to the wrapped function, c
    will see it as (1,5,3). The output will then have its left-most axis removed.
    Input will be cast to ndarray before being passed to c, and output will be cast to
    Tensor.

    Arguments:
    ---------
    c:
        Callable that takes two ndarrays as input and returns two ndarrays. This
        function should take arrays of shape [n,...]. Arrays that are passed to
        the this function after wrapping will have a left-most axis added with a length
        of 1.

    Returns:
    -------
    Callable that calls c after reshaping/casting and returns reshaped/casted output.

    Notes:
    -----
    In the context of aggforce TrajectoryMap isntances, this function should be used to
    wrap the .map_arrays method, not the TrajectoryMap itself. Also note that the
    output of this function may not be pickleable due to being a closure.

    """

    def _wrapped(positions: Tensor, forces: Tensor) -> Tuple[Tensor, Tensor]:
        results = c(asarray(positions)[None, ...], asarray(forces)[None, ...])
        positions = as_tensor(results[0][0])
        forces = as_tensor(results[1][0])
        return (positions, forces)

    return _wrapped


def _aggforce_wrapper(
    c: Callable[[ndarray, ndarray], Tuple[ndarray, ndarray]], /
) -> _pftransform:
    """Wrap function which maps numpy trajectory chunks to one mapping Tensor chunks.

    Arguments:
    ---------
    c:
        Callable that takes two ndarrays as input and returns two ndarrays.

    Returns:
    -------
    Callable that calls c after casting and returns cast output.

    Notes:
    -----
    In the context of aggforce TrajectoryMap isntances, this function should be used to
    wrap the .map_arrays method, not the TrajectoryMap itself. Also note that the
    output of this function may not be pickleable due to being a closure.

    """

    def _wrapped(positions: Tensor, forces: Tensor) -> Tuple[Tensor, Tensor]:
        results = c(asarray(positions), asarray(forces))
        positions = as_tensor(results[0])
        forces = as_tensor(results[1])
        return (positions, forces)

    return _wrapped


class PosForceTransform(BaseTransform):
    """Transforms coordinate and force entries of an AtomicData instance.

    Transform is effectively applied _per frame_ as this maps individual
    AtomicData instances.
    """

    def __init__(
        self,
        transform: Union[_pftransform, str, Callable],
        copy: bool = True,
        baseline_models: Union[str, Dict[str, Module], None] = None,
        aggforce_style: bool = False,
    ) -> None:
        """Initialize.

        Arguments:
        ---------
        transform:
            Callable that powers the
        copy:
            If true, we deepcopy the input AtomicData instance during call.
        baseline_models:
            Used to subtract out forces on the _transformed_ positions. If not
            a string or None, passed to mlcg.datasets.utils.remove_baseline_forces.
            If a string, read using torch.load and then passed. If None, no
            post-transform correction is done.
        aggforce_style:
            If true, we wrap the provided transform using _aggforce_wrapper; this
            allows a TrajectoryMap.map_arrays instance to be used.

        Notes:
        -----
        If transform is a string, we treat it as a file name and attempt to read it
        via pickle; the read object is then used as the transform. If aggforce_style
        is False, the used transform should take two arguments: first the position
        Tensor and Force Tensor of a frame(both with shape (n_particles,3)), and will
        return a 2-tuple of Tensors (first element mapped positions, second element the
        mapped forces).

        If aggforce_style is True, then the transform should be a .map_arrays method of
        a TrajectoryMap (not a TrajectoryMap instance itself!).

        """
        if isinstance(transform, str):
            # maybe we can use torch.load here too
            with open(transform, "rb") as f:  # noqa: PTH123
                _transform: Callable = pickle.load(f)
        else:
            _transform = transform
        if aggforce_style:
            self.transform: _pftransform = _AggforceFramewiseWrapper(_transform)
        else:
            self.transform = _transform
        self.copy = copy
        if isinstance(baseline_models, str):
            _baseline_models = load(baseline_models)
        else:
            _baseline_models = baseline_models
        self.baseline_models = _baseline_models

    def forward(self, data: AtomicData) -> AtomicData:
        """Evaluate transform.

        Arguments:
        ---------
        data:
            AtomicData instance to transform. Should have position and force entries.

        Returns:
        -------
        AtomicData instance representing transformed data. May be a copy; see
        initialization. Entries besides positions and forces will match input instance.

        """
        if self.copy:
            new_data = deepcopy(data)
        else:
            new_data = data
        old_positions = getattr(data, POSITIONS_KEY)
        old_forces = getattr(data, FORCE_KEY)
        new_positions, new_forces = self.transform(old_positions, old_forces)
        setattr(new_data, POSITIONS_KEY, as_tensor(new_positions))
        setattr(new_data, FORCE_KEY, as_tensor(new_forces))
        corrected_data = self._remove_baseline(new_data)
        return corrected_data

    def __call__(self, data: Any) -> Any:
        # Shallow-copy the data so that we prevent in-place data modification.
        # add here since the older version of pytorch geometric has a different
        # __call__ in BaseTransform
        return self.forward(shallow_copy(data))

    def _remove_baseline(self, frame: AtomicData) -> AtomicData:
        """Remove baseline from forces.

        This may modify the input AtomicData.
        """
        if self.baseline_models is None:
            return frame
        else:
            # note the terminal [0]
            with enable_grad():
                frame = remove_baseline_forces(
                    data_list=[frame], models=self.baseline_models
                )[0]
            return frame

    def __repr__(self) -> str:
        """Return string representation of transform."""
        msg = repr(type(self)) + ": " + repr(self.transform)
        return msg


class PosForceBT:
    """Transforms coordinate and force entries of an AtomicData instance.

    Transform is effectively applied _per frame_ as this maps individual
    AtomicData instances.
    """

    def __init__(
        self,
        transform: Union[_pftransform, str, Callable],
        copy: bool = True,
        baseline_models: Union[str, Dict[str, Module], None] = None,
        aggforce_style: bool = False,
    ) -> None:
        """Initialize.

        Arguments:
        ---------
        transform:
            Callable that powers the
        copy:
            If true, we deepcopy the input AtomicData instance during call.
        baseline_models:
            Used to subtract out forces on the _transformed_ positions. If not
            a string or None, passed to mlcg.datasets.utils.remove_baseline_forces.
            If a string, read using torch.load and then passed. If None, no
            post-transform correction is done.
        aggforce_style:
            If true, we wrap the provided transform using _aggforce_wrapper; this
            allows a TrajectoryMap.map_arrays instance to be used.

        Notes:
        -----
        If transform is a string, we treat it as a file name and attempt to read it
        via pickle; the read object is then used as the transform. If aggforce_style
        is False, the used transform should take two arguments: first the position
        Tensor and Force Tensor of a frame(both with shape (n_particles,3)), and will
        return a 2-tuple of Tensors (first element mapped positions, second element the
        mapped forces).

        If aggforce_style is True, then the transform should be a .map_arrays method of
        a TrajectoryMap (not a TrajectoryMap instance itself!).

        """
        if isinstance(transform, str):
            # maybe we can use torch.load here too
            with open(transform, "rb") as f:  # noqa: PTH123
                _transform: Callable = pickle.load(f)
        else:
            _transform = transform
        if aggforce_style:
            self.batch_transform = _aggforce_wrapper(_transform)
        else:
            self.batch_transform = _transform
        self.copy = copy
        if isinstance(baseline_models, str):
            _baseline_models = load(baseline_models)
        else:
            _baseline_models = baseline_models
        self.baseline_models = _baseline_models

    def __call__(self, datas: Iterable[AtomicData]) -> List[AtomicData]:
        """Evaluate transform.

        Arguments:
        ---------
        datas:
            Iterable of AtomicData instances to transform. Should have position and
            force entries.

        Returns:
        -------
        AtomicData instance representing transformed data. May be a copy; see
        initialization. Entries besides positions and forces will match input instance.

        """
        if self.copy:
            new_datas = deepcopy(datas)
        else:
            new_datas = datas
        old_positions = stack([getattr(x, POSITIONS_KEY) for x in datas])
        old_forces = stack([getattr(x, FORCE_KEY) for x in datas])
        new_positions, new_forces = self.batch_transform(
            old_positions, old_forces
        )
        # we are copying via tensor here
        for p, f, d in zip(new_positions, new_forces, new_datas):
            setattr(d, POSITIONS_KEY, tensor(p))
            setattr(d, FORCE_KEY, tensor(f))
        corrected_datas = self._remove_baseline(new_datas)  # type: ignore
        return corrected_datas

    def _remove_baseline(self, frames: List[AtomicData]) -> List[AtomicData]:
        """Remove baseline from forces.

        This may modify the input AtomicData instances.
        """
        if self.baseline_models is None:
            return frames
        else:
            return remove_baseline_forces(
                data_list=frames, models=self.baseline_models
            )

    def __repr__(self) -> str:
        """Return string representation of transform."""
        msg = repr(type(self)) + ": " + repr(self.batch_transform)
        return msg


def simplify_trajectory_map_callable(tmap_array_noiser):
    from aggforce.map.tmap import SeperableTMap, AugmentedTMap
    from aggforce.map.jaxlinearmap import JLinearMap
    from aggforce.trajectory.jaxgausstraj import JCondNormal

    if not hasattr(tmap_array_noiser, "__self__") or not isinstance(
        tmap_array_noiser.__self__, AugmentedTMap
    ):
        raise ValueError("Expecting a AugmentedTMap to be simplified.")

    def convert_combined_tmap_to_numpy_only(tmap):
        assert isinstance(tmap, SeperableTMap)
        if isinstance(tmap.coord_map, JLinearMap):
            tmap.coord_map = tmap.coord_map.to_linearmap()
        if isinstance(tmap.force_map, JLinearMap):
            tmap.force_map = tmap.force_map.to_linearmap()

    convert_combined_tmap_to_numpy_only(tmap_array_noiser.__self__.tmap)
    if isinstance(tmap_array_noiser.__self__.augmenter, JCondNormal):
        tmap_array_noiser.__self__.augmenter = (
            tmap_array_noiser.__self__.augmenter.to_SimpleCondNormal()
        )
    return tmap_array_noiser


class PosForceTransformCollater:
    """Transforms coordinate and force entries of an AtomicData collated from a batch of frames
    that correspond to the same molecule.

    Transform is effectively applied _per frame_ as this maps individual
    AtomicData instances.
    """

    def __init__(
        self,
        transform: Union[_pftransform, str, Callable, None] = None,
        copy: bool = True,
        baseline_models: Union[str, Dict[str, Module], None] = None,
        aggforce_style: bool = False,
        collater_fn: PyGCollater = PyGCollater(None, None),
        remove_neighbor_list: bool = True,
        use_simple_augmenter: bool = True,
    ) -> None:
        """Initialize.

        Arguments:
        ---------
        transform:
            Callable that powers the
        copy:
            If true, we deepcopy the input AtomicData instance during call.
        baseline_models:
            Used to subtract out forces on the _transformed_ positions. If not
            a string or None, passed to mlcg.datasets.utils.remove_baseline_forces.
            If a string, read using torch.load and then passed. If None, no
            post-transform correction is done.
        aggforce_style:
            If true, we wrap the provided transform using _aggforce_wrapper; this
            allows a TrajectoryMap.map_arrays instance to be used.

        Notes:
        -----
        If transform is a string, we treat it as a file name and attempt to read it
        via pickle; the read object is then used as the transform. If aggforce_style
        is False, the used transform should take two arguments: first the position
        Tensor and Force Tensor of a frame(both with shape (n_particles,3)), and will
        return a 2-tuple of Tensors (first element mapped positions, second element the
        mapped forces).

        If aggforce_style is True, then the transform should be a .map_arrays method of
        a TrajectoryMap (not a TrajectoryMap instance itself!).

        """
        if isinstance(transform, str):
            # maybe we can use torch.load here too
            with open(transform, "rb") as f:  # noqa: PTH123
                _transform: Callable = pickle.load(f)
        else:
            _transform = transform
        if _transform is not None and aggforce_style:
            if use_simple_augmenter:
                _transform = simplify_trajectory_map_callable(_transform)
            self.batch_transform = _AggforceWrapper(_transform)
        else:
            self.batch_transform = _transform
        self.copy = copy
        if isinstance(baseline_models, str):
            _baseline_models = load(baseline_models)
        else:
            _baseline_models = baseline_models
        self.baseline_models = _baseline_models
        self._remove_neighbor_list = remove_neighbor_list
        self._collater_fn = collater_fn

    def __call__(self, datas: Iterable[AtomicData]) -> AtomicData:
        """Evaluate transform and collate the output to a single AtomicData

        Arguments:
        ---------
        datas:
            Iterable of AtomicData instances to transform. Should have position and
            force entries.

        Returns:
        -------
        AtomicData instance representing transformed data. May be a copy; see
        initialization. Entries besides positions and forces will match input instance.

        """
        if self.copy:
            new_datas = deepcopy(datas)
        else:
            new_datas = datas
        # we assume the frames are from the same molecule
        old_positions = stack([getattr(x, POSITIONS_KEY) for x in datas])
        old_forces = stack([getattr(x, FORCE_KEY) for x in datas])
        if self.batch_transform is not None:
            new_positions, new_forces = self.batch_transform(
                old_positions, old_forces
            )
        else:
            new_positions, new_forces = old_positions, old_forces
        # collation happens here
        collated_data = self._collater_fn(datas)
        setattr(collated_data, POSITIONS_KEY, new_positions.flatten(0, 1))
        setattr(collated_data, FORCE_KEY, new_forces.flatten(0, 1))
        # remove the baseline forces
        corrected_data = self._collated_remove_baseline(collated_data)  # type: ignore

        return corrected_data

    def _collated_remove_baseline(
        self, collated_frames: AtomicData
    ) -> AtomicData:
        """Remove baseline from forces in the collated AtomicData.

        This may modify the input AtomicData instances.
        """
        if self.baseline_models is None:
            return collated_frames
        else:
            # remove baseline (collated)
            collated_forces = collated_frames[FORCE_KEY]
            baseline_forces = zeros_like(collated_forces, requires_grad=False)
            for k, model in self.baseline_models.items():
                model.eval()
                with enable_grad():
                    collated_frames = model(collated_frames)
                baseline_forces += collated_frames.out[k][FORCE_KEY].detach()
                del collated_frames.out[k]
            collated_forces -= baseline_forces
            if self._remove_neighbor_list:
                collated_frames.neighbor_list = {}
            return collated_frames

    def __repr__(self) -> str:
        """Return string representation of transform."""
        msg = repr(type(self)) + ": " + repr(self.batch_transform)
        return msg


class PosForceTransformCollaterLWP(PosForceTransformCollater):
    """Same as `PosForceTransformCollater` but generates Light-Weight Pickle file
    and PytorchLightning checkpoints.


    Note: pytorch lightning expects this class to be able to pickled and unpickled
    without __init__ run. So there are some hacks to work around this requirement.
    """

    _init_args = {}

    def __init__(self, **kwargs):
        self._init_args = kwargs
        super(PosForceTransformCollaterLWP, self).__init__(**kwargs)

    def __getstate__(self):
        """Determines what to be saved when pickling."""
        return self._init_args

    def __setstate__(self, state):
        """Reconstruct the instance."""
        if state:
            super(PosForceTransformCollaterLWP, self).__init__(**state)

    def __repr__(self) -> str:
        """Return string representation of transform."""
        if self._init_args:
            return super(PosForceTransformCollaterLWP, self).__repr__()
        else:
            return ""


class Noiser(torch.nn.Module):
    ndim = 3

    def __init__(self, sigma, kbt, coord_map, force_map):
        super(Noiser, self).__init__()
        self.sigma = torch.nn.Parameter(
            torch.tensor(sigma), requires_grad=False
        )
        self.kbt = torch.nn.Parameter(torch.tensor(kbt), requires_grad=False)
        self.coord_map = torch.nn.Parameter(coord_map, requires_grad=False)
        self.force_map = torch.nn.Parameter(force_map, requires_grad=False)

    def forward(self, coords, forces):
        with torch.no_grad():
            coords = coords.reshape([-1, len(self.coord_map), self.ndim])
            forces = forces.reshape([-1, len(self.force_map), self.ndim])
            noise = torch.randn_like(coords)
            # aug_coords = self.augmenter.sample(coords)
            aug_coords = coords + self.sigma * noise
            full_coords = torch.cat([coords, aug_coords], axis=-2)
            # aug_forces = self.kbt * aug_lgrad
            aug_forces = -self.kbt * (noise / self.sigma)
            # real_forces_corrected = forces + self.kbt * real_lgrad_correction
            real_forces_corrected = forces - aug_forces
            full_forces = torch.cat(
                [real_forces_corrected, aug_forces], axis=-2
            )
            orig_prec = torch.get_float32_matmul_precision()
            torch.set_float32_matmul_precision("highest")
            new_coords, new_forces = (
                self.coord_map @ full_coords,
                self.force_map @ full_forces,
            )
            torch.set_float32_matmul_precision(orig_prec)
            return new_coords.flatten(0, 1), new_forces.flatten(0, 1)

    @classmethod
    def from_aug_tmap(cls, aug_tmap):
        sigma = math.sqrt(aug_tmap.augmenter._cov)
        kbt = aug_tmap.kbt
        coord_map = torch.tensor(aug_tmap.tmap.coord_map.standard_matrix)
        force_map = torch.tensor(aug_tmap.tmap.force_map.standard_matrix)
        return cls(sigma, kbt, coord_map, force_map)

    def __repr__(self) -> str:
        """Return string representation of Noiser."""
        msg = f"<Noiser: sigma={self.sigma.item()}, kbt={self.kbt.item()} with linear transforms>"
        return msg


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
        if "scale" not in option or option["scale"] <= 0 or "prob" not in option or option["prob"] < 0:
            raise ValueError(f"Invalid decoy option: {option}")
        summary_decoy_probs.append(option["prob"])
    if sum(summary_decoy_probs) > 1:
        raise ValueError(f"total decoy probability exceeds 1: {summary_decoy_probs}")
    summary_decoy_probs.append(1 - sum(summary_decoy_probs))
    return summary_decoy_probs


def pick_decoy_frames(data, summary_decoy_probs):
    """Decide the number of frames picked for each case, the last one corresponds to the number of intact samples
    """
    n_frames = len(data.n_atoms)
    n_picks = np.random.multinomial(n_frames, summary_decoy_probs)[:-1]
    shuffled_ids = np.arange(n_frames)
    np.random.shuffle(shuffled_ids)
    shuffled_ids = shuffled_ids.tolist()
    n_picks_sum = np.cumsum(n_picks).tolist()
    id_picks = [shuffled_ids[a:b] for a, b in zip([0] + n_picks_sum, n_picks_sum)]
    ptr = data.ptr.cpu().numpy()
    frame_ranges = [[(ptr[id_], ptr[id_ + 1]) for id_ in id_pick] for id_pick in id_picks]
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


class PosForceTransformCollaterTorch:
    """Transforms coordinate and force entries of an AtomicData collated from a batch of frames
    that correspond to the same molecule.

    Transform is effectively applied _per frame_ as this maps individual
    AtomicData instances.
    """

    def __init__(
        self,
        transform: Union[_pftransform, str, Callable, None] = None,
        baseline_models: Union[str, Dict[str, Module], None] = None,
        aggforce_style: bool = False,
        collater_fn: PyGCollater = PyGCollater(None, None),
        remove_neighbor_list: bool = True,
        decoy_options: Union[List[Dict[str, float]], None] = None,
        device: str = "cuda",
    ) -> None:
        """Initialize.

        Arguments:
        ---------
        transform:
            Callable that powers the
        copy:
            If true, we deepcopy the input AtomicData instance during call.
        baseline_models:
            Used to subtract out forces on the _transformed_ positions. If not
            a string or None, passed to mlcg.datasets.utils.remove_baseline_forces.
            If a string, read using torch.load and then passed. If None, no
            post-transform correction is done.
        aggforce_style:
            If true, we wrap the provided transform using _aggforce_wrapper; this
            allows a TrajectoryMap.map_arrays instance to be used.

        Notes:
        -----
        If transform is a string, we treat it as a file name and attempt to read it
        via pickle; the read object is then used as the transform. If aggforce_style
        is False, the used transform should take two arguments: first the position
        Tensor and Force Tensor of a frame(both with shape (n_particles,3)), and will
        return a 2-tuple of Tensors (first element mapped positions, second element the
        mapped forces).

        If aggforce_style is True, then the transform should be a .map_arrays method of
        a TrajectoryMap (not a TrajectoryMap instance itself!).

        """
        if isinstance(transform, str):
            # maybe we can use torch.load here too
            print(f"Loading... {transform}")
            with open(transform, "rb") as f:  # noqa: PTH123
                _transform: Callable = pickle.load(f)
        else:
            _transform = transform
        self.device = torch.device(device)
        if _transform is not None and aggforce_style:
            self.noiser = Noiser.from_aug_tmap(_transform.__self__).to(
                self.device
            )
        else:
            self.noiser = _transform
        if isinstance(baseline_models, str):
            _baseline_models = load(baseline_models)
        else:
            _baseline_models = baseline_models
        if _baseline_models is not None:
            self.baseline_models = _baseline_models.to(self.device)
        else:
            self.baseline_models = None
        if decoy_options is not None:
            self._decoy_opts = decoy_options
            self._summary_decoy_probs = check_and_sum_decoy_prob(decoy_options)
        else:
            self._decoy_opts = None
        self._remove_neighbor_list = remove_neighbor_list
        self._collater_fn = collater_fn
        self._moved_to_device = False

    def __call__(self, datas: Iterable[AtomicData]) -> AtomicData:
        """Evaluate transform and collate the output to a single AtomicData

        Arguments:
        ---------
        datas:
            Iterable of AtomicData instances to transform. Should have position and
            force entries.

        Returns:
        -------
        AtomicData instance representing transformed data. May be a copy; see
        initialization. Entries besides positions and forces will match input instance.

        """
        if not self._moved_to_device:
            # copy the params over to the current device
            # since we noticed that the self.device can change after init in ddp
            if self.baseline_models:
                self.baseline_models.to(self.device)
            if self.noiser:
                self.noiser.to(self.device)
            self._moved_to_device = True
        # collation happens here
        collated_data = self._collater_fn(datas).to(self.device)
        if self.noiser:
            new_pos, new_forces = self.noiser(
                collated_data[POSITIONS_KEY], collated_data[FORCE_KEY]
            )
            setattr(collated_data, POSITIONS_KEY, new_pos)
            setattr(collated_data, FORCE_KEY, new_forces)
        # remove the baseline forces
        corrected_data = self._collated_remove_baseline(collated_data)  # type: ignore
        # alek's idea of regularization of neural network predictions
        # 0. randomly choose some frames
        # 1. add huge amount of noises to the coords (creating decoys)
        # 2. set the delta forces to zero (so the NN learns to rely on priors instead of very wrong extrapolations)
        if self._decoy_opts:
            _, frame_ranges = pick_decoy_frames(corrected_data, self._summary_decoy_probs)
            for opt, ranges in zip(self._decoy_opts, frame_ranges):
                for start, stop in ranges:
                    decoy_frame_(corrected_data, opt["scale"], start, stop)
        return corrected_data

    def _collated_remove_baseline(
        self, collated_frames: AtomicData
    ) -> AtomicData:
        """Remove baseline from forces in the collated AtomicData.

        This may modify the input AtomicData instances.
        """
        if self.baseline_models is None:
            return collated_frames
        else:
            # remove baseline (collated)
            collated_forces = collated_frames[FORCE_KEY]
            baseline_forces = zeros_like(collated_forces, requires_grad=False)
            for k, model in self.baseline_models.items():
                model.eval()
                with enable_grad():
                    collated_frames = model(collated_frames)
                baseline_forces += collated_frames.out[k][FORCE_KEY].detach()
                del collated_frames.out[k]
            collated_forces -= baseline_forces
            if self._remove_neighbor_list:
                collated_frames.neighbor_list = {}
            return collated_frames

    def __repr__(self) -> str:
        """Return string representation of transform."""
        msg = repr(type(self)) + ": " + repr(self.noiser)
        return msg


class PosForceTransformCollaterTorchLWP(PosForceTransformCollaterTorch):
    """Same as `PosForceTransformCollaterTorch` but generates Light-Weight Pickle file
    and PytorchLightning checkpoints.


    Note: pytorch lightning expects this class to be able to pickled and unpickled
    without __init__ run. So there are some hacks to work around this requirement.
    """

    _init_args = {}

    def __init__(self, **kwargs):
        self._init_args = kwargs
        super(PosForceTransformCollaterTorchLWP, self).__init__(**kwargs)

    def __getstate__(self):
        """Determines what to be saved when pickling."""
        return self._init_args

    def __setstate__(self, state):
        """Reconstruct the instance."""
        if state:
            super(PosForceTransformCollaterTorchLWP, self).__init__(**state)

    def __repr__(self) -> str:
        """Return string representation of transform."""
        if self._init_args:
            return super(PosForceTransformCollaterTorchLWP, self).__repr__()
        else:
            return ""


class MixingNoiser(torch.nn.Module):

    def __init__(self, sigma, kbt, mixing_matrix=None, mixing_coeffs=None, noising_mask=None):
        """Prioritize `mixing_matrix` if it's not None, otherwise use `mixing_coeffs` if they exist, 
        otherwise generates noise-only signals. `noising_mask` can be used to set which beads to be noised.
        """
        super(MixingNoiser, self).__init__()
        self.sigma = torch.nn.Parameter(
            torch.tensor(sigma), requires_grad=False
        )
        self.kbt = torch.nn.Parameter(torch.tensor(kbt), requires_grad=False)
        self.mixing_coeffs = None
        self.mixing_matrix = None
        if mixing_matrix is not None:
            self.mixing_matrix = torch.nn.Parameter(torch.tensor(mixing_matrix), requires_grad=False)
            self.n_beads = self.mixing_matrix.shape[1]
        elif mixing_coeffs is not None:
            self.mixing_coeffs = torch.nn.Parameter(torch.tensor(mixing_coeffs), requires_grad=False)
        else:
            print("Noise-only forces will be generated.")
        if noising_mask is not None:
            self.noising_mask = torch.nn.Parameter(torch.tensor(noising_mask, dtype=torch.bool), requires_grad=False)
            self.n_noising_beads = self.noising_mask.sum().item()
            self.n_beads = len(self.noising_mask)
            if self.mixing_coeffs is not None:
                assert len(self.mixing_coeffs) == self.n_noising_beads
            print(f"Noising only the following indices {self.noising_mask.argwhere()[:, 0]}")
        else:
            self.noising_mask = None

    def forward(self, coords, forces):
        if self.noising_mask is None:
            # traditional: noising all CG sites
            with torch.no_grad():
                noise = torch.randn_like(coords)
                aug_coords = coords + self.sigma * noise
                aug_forces = -self.kbt * (noise / self.sigma)
                if self.mixing_matrix is not None:
                    real_forces_corrected = forces - aug_forces
                    batch_forces = real_forces_corrected.reshape([-1, self.n_beads, 3])
                    orig_prec = torch.get_float32_matmul_precision()
                    torch.set_float32_matmul_precision("highest")
                    batch_forces = self.mixing_matrix @ batch_forces
                    torch.set_float32_matmul_precision(orig_prec)
                    aug_forces += batch_forces.reshape([-1, 3]) 
                elif self.mixing_coeffs is not None:
                    # real_forces_corrected = forces - aug_forces
                    n_batch = len(forces) // len(self.mixing_coeffs)
                    mixing_coeffs = torch.tile(self.mixing_coeffs, (n_batch,))[:, None]
                    aug_forces = forces * mixing_coeffs + aug_forces * (1 - mixing_coeffs)
                return aug_coords, aug_forces
        else:
            # only noising specified sites:
            with torch.no_grad():
                batch_coords = coords.reshape([-1, self.n_beads, 3])
                batch_forces = forces.reshape([-1, self.n_beads, 3])
                noise = torch.randn(len(batch_coords), self.n_noising_beads, 3, device=coords.device)
                batch_coords[:, self.noising_mask] += self.sigma * noise
                aug_forces = -self.kbt * (noise / self.sigma)
                if self.mixing_matrix is not None:
                    batch_forces[:, self.noising_mask] -= aug_forces
                    orig_prec = torch.get_float32_matmul_precision()
                    torch.set_float32_matmul_precision("highest")
                    batch_forces = self.mixing_matrix @ batch_forces
                    torch.set_float32_matmul_precision(orig_prec)
                    batch_forces[:, self.noising_mask] += aug_forces
                elif self.mixing_coeffs is not None:
                    # real_forces_corrected = batch_forces[:, self.noising_mask] - aug_forces
                    mixing_coeffs = self.mixing_coeffs[:, None]
                    aug_forces = batch_forces[:, self.noising_mask] * mixing_coeffs + aug_forces * (1 - mixing_coeffs)
                    batch_forces[:, self.noising_mask] = aug_forces
                else:
                    # a mixture of noise-only forces on the noised beads and 0-noise original CG force on the rest
                    batch_forces[:, self.noising_mask] = aug_forces
                return batch_coords.reshape([-1, 3]), batch_forces.reshape([-1, 3])

    def __repr__(self) -> str:
        """Return string representation of Noiser."""
        msg = f"<MixingNoiser: sigma={self.sigma.item()}, kbt={self.kbt.item()}"
        if self.mixing_matrix is not None:
            msg += " with mixing matrix"
        elif self.mixing_coeffs is not None:
            msg += " with mixing coefficients"
        else:
            msg += " (noise only)"
        if self.noising_mask is not None:
            msg += f" (noising only beads {self.noising_mask.argwhere()[:, 0]})>"
        else:
            msg += ">"
        return msg


class PosForceNoiseMixingCollaterTorch(PosForceTransformCollaterTorch):
    """Transforms coordinate and force entries of an AtomicData collated from a batch of frames
    that correspond to the same molecule.
    
    When `noise_level` is a positive float, noise the coordinates and forces. Otherwise use
    the original forces from the upstream dataset (i.e., mapped all-atom forces).
    Only supports simple noise mixing, i.e., output force is noise signal + mixing_coeffs * 
    real_forces_corrected. When `mixing_coeffs` is None while `noise_level` is a positive
    float, generates data for noise-only training.
    
    When `baseline_models` is a valid path to a saved torch ModuleDict, the baseline forces
    will be removed from the forces from above.
    """

    def __init__(
        self,
        noise_level: Optional[float] = None,
        kbt: Optional[float] = None,
        mixing_matrix: Optional[str] = None,
        mixing_coeffs: Optional[str] = None,
        noising_mask: Optional[str] = None,
        baseline_models: Union[str, Dict[str, Module], None] = None,
        collater_fn: PyGCollater = PyGCollater(None, None),
        remove_neighbor_list: bool = True,
        decoy_options: Union[List[Dict[str, float]], None] = None,
        device: str = "cuda",
    ) -> None:
        """Initialize.

        Arguments:
        ---------
        noise_level:
            The variance of the white isotropic Guassian gonna be added to the positions
            and have effects on the forces.
        baseline_models:
            Used to subtract out forces on the _transformed_ positions. If not
            a string or None, passed to mlcg.datasets.utils.remove_baseline_forces.
            If a string, read using torch.load and then passed. If None, no
            post-transform correction is done.
        """
        if isinstance(mixing_matrix, str):
            print(f"Loading mixing matrix... {mixing_matrix}")
            mixing_matrix = np.load(mixing_matrix)
        elif isinstance(mixing_coeffs, str):
            print(f"Loading mixing coefficients... {mixing_coeffs}")
            mixing_matrix = None
            mixing_coeffs = np.load(mixing_coeffs)
        else:
            mixing_matrix = None
            mixing_coeffs = None
        
        if isinstance(noising_mask, str):
            print(f"Loading noising mask... {noising_mask}")
            noising_mask = np.load(noising_mask)
        else:
            noising_mask = None
        
        self.device = torch.device(device)
        
        if noise_level:
            sigma = math.sqrt(noise_level)
            self.noiser = MixingNoiser(
                sigma, 
                kbt,
                mixing_matrix=mixing_matrix,
                mixing_coeffs=mixing_coeffs,
                noising_mask=noising_mask,
            )
        else:
            self.noiser = None
        
        if isinstance(baseline_models, str):
            self.baseline_models = load(baseline_models)
        else:
            self.baseline_models = baseline_models
        self._remove_neighbor_list = remove_neighbor_list
        self._collater_fn = collater_fn
        self._moved_to_device = False
        if decoy_options is not None:
            self._decoy_opts = decoy_options
            self._summary_decoy_probs = check_and_sum_decoy_prob(decoy_options)
        else:
            self._decoy_opts = None


class PosForceNoiseMixingCollaterTorchLWP(PosForceNoiseMixingCollaterTorch):
    """Same as `PosForceNoiseMixingCollaterTorch` but generates Light-Weight Pickle file
    and PytorchLightning checkpoints.


    Note: pytorch lightning expects this class to be able to pickled and unpickled
    without __init__ run. So there are some hacks to work around this requirement.
    """

    _init_args = {}

    def __init__(self, **kwargs):
        self._init_args = kwargs
        super(PosForceNoiseMixingCollaterTorchLWP, self).__init__(**kwargs)

    def __getstate__(self):
        """Determines what to be saved when pickling."""
        return self._init_args

    def __setstate__(self, state):
        """Reconstruct the instance."""
        if state:
            super(PosForceNoiseMixingCollaterTorchLWP, self).__init__(**state)

    def __repr__(self) -> str:
        """Return string representation of transform."""
        if self._init_args:
            return super(PosForceNoiseMixingCollaterTorchLWP, self).__repr__()
        else:
            return ""
