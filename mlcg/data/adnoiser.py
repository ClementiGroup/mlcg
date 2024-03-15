"""Noiser."""

from typing import Union, Callable, Tuple, Final, Dict, Iterable, List, Any
from copy import copy as shallow_copy
from copy import deepcopy
import pickle
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
