import pytest

import typing
import inspect
import torch

from ._fix_hparams_saving import yaml


def my_inspect_Parameter():
    name = "func"
    annotations = [
        typing.Optional[
            typing.Tuple[
                typing.List[torch.Tensor],
                typing.Union[int, typing.List[int]],
                typing.Optional[int],
            ]
        ],
        typing.Optional[torch.Tensor],
        typing.Tuple[int, float, str, torch.Tensor, typing.List[float]],
    ]
    kind = 1
    default = None
    for annotation in annotations:
        param = inspect.Parameter(
            name, kind, default=default, annotation=annotation
        )
        yield param


@pytest.mark.parametrize(
    "param",
    [(param) for param in my_inspect_Parameter()],
)
def test_yaml_load_dump(param):
    vv = yaml.dump(param)
    duplicate = yaml.load(vv, yaml.Loader)

    keys = [
        "name",
        "annotation",
        "kind",
        "default",
    ]
    for k in keys:
        assert getattr(duplicate, k) == getattr(param, k)

    assert duplicate == param
