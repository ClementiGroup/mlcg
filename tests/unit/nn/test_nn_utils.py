from copy import deepcopy
import functools
import numpy as np
import pytest
import torch

from mlcg.nn.utils import desparsify_prior_module, sparsify_prior_module
from mlcg.nn.prior import HarmonicAngles, Dihedral

rng = np.random.default_rng()

angle_keys = [tuple(key_comb) for key_comb in rng.integers(20, size=(30, 3))]
angle_dict = {
    key_comb: {"x_0": rng.random(), "k": rng.random()}
    for key_comb in angle_keys
}
dihedral_keys = [tuple(key_comb) for key_comb in rng.integers(60, size=(30, 4))]
dihedral_dict = {
    key_comb: {
        "v_0": rng.random(),
        "k1s": {
            "k1_1": rng.random(),
            "k1_2": rng.random(),
            "k1_3": rng.random(),
        },
        "k2s": {
            "k2_1": rng.random(),
            "k2_2": rng.random(),
            "k2_3": rng.random(),
        },
    }
    for key_comb in angle_keys
}

ang_prior = HarmonicAngles(angle_dict)
dih_prior = Dihedral(dihedral_dict, n_degs=3)


torch_assert_equal = functools.partial(
    torch.testing.assert_close, rtol=0, atol=0
)


@pytest.mark.parametrize("prior_module", [ang_prior, dih_prior])
def test_prior_sparsification(prior_module: torch.nn.Module) -> None:
    original_prior = deepcopy(prior_module)
    sparsify_prior_module(prior_module)
    for name, dense_buf in original_prior.named_buffers():
        sparse_buf = prior_module.get_buffer(name)
        torch_assert_equal(dense_buf, sparse_buf.to_dense())
    desparsify_prior_module(prior_module)
    for name, dense_buf in original_prior.named_buffers():
        desparsified_buf = prior_module.get_buffer(name)
        torch_assert_equal(dense_buf, desparsified_buf)
