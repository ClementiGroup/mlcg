"""
Contuinity test for restarting the training

This test retries to run training from some checkpoints previously created.
This test automatically validates all the training YAMLs in `./input_yamls`
(except those with bonded exclusions) by performing the following steps:

1. **Training**
   - For each training YAML, it loads the corresponding reference data YAML.
   - It rewrites paths (HDF5, partitions) and trainer options to produce a small,
     CPU-only training configuration suitable for pytest.
   - It extracts the exact training command from `input_yamls/README.md` and
     executes it in a subprocess.
   - The test asserts that training completes successfully.


Temporary files and directories (`pytest_runs/`) are created and automatically
cleaned up by the `test_dir` fixture, regardless of whether the test passes
or fails.

**Important:** This test depends on the example commands written in
`input_yamls/README.md`. If those commands or the example YAMLs are modified,
the tests may fail.
"""

import subprocess
from mlcg.utils import load_yaml, dump_yaml
from pathlib import Path
import pytest
import os
from shutil import rmtree

_here = Path(__file__).parent
_example_dir = _here.parent.parent / "examples"

training_yaml_list = sorted((_example_dir / "input_yamls").glob("train*.yaml"))
training_yaml_list = [
    file for file in training_yaml_list if "exclusions" not in str(file)
]

global_data_yaml = (
    _example_dir / "h5_pl/multiple_molecules/train_demo_cuda.yaml"
)
priors = str(_example_dir / "h5_pl/multiple_molecules/prior.pt")
struct = str(
    _example_dir / "h5_pl/multiple_molecules/cln_configurations_demo.pt"
)
sim_yaml = _example_dir / "h5_pl/multiple_molecules/cln_sim_demo.yaml"


# All yield fixtures in pytest are executed untill the
# yield in the order they are provided to the test function and
# after the function is finished the code after the yeld is executed
# for all the fixtures in reverse order


# Create a temporary test directory
@pytest.fixture
def test_dir(tmp_path):
    yield tmp_path
    # Teardown: always run also if the test fails
    if tmp_path.exists():
        rmtree(tmp_path)


@pytest.mark.parametrize("model_yaml", training_yaml_list, ids=lambda p: p.stem)
def test_architecture(model_yaml, test_dir):
    name = model_yaml.stem.replace("train_", "")
    print(f"Testing architecture {name}")
    data_path = Path(global_data_yaml)
    training_yaml = load_yaml(model_yaml)
    data_yaml = load_yaml(global_data_yaml)

    training_yaml["data"] = data_yaml["data"]
    training_yaml["data"]["h5_file_path"] = str(
        data_path.parent / data_yaml["data"]["h5_file_path"]
    )
    training_yaml["data"]["partition_options"] = str(
        data_path.parent / "small_partition_demo.yaml"
    )
    training_yaml["trainer"]["max_epochs"] = 4
    training_yaml["trainer"].pop("devices")
    training_yaml["trainer"]["default_root_dir"] = str(test_dir)
    training_yaml["trainer"]["accelerator"] = "cpu"
    training_yaml["ckpt_path"] = f"{_here}/training_ckpts/{name}_train.ckpt"
    dump_yaml(test_dir / "pytest_training.yaml", training_yaml)

    # Run the training command
    result = subprocess.run(
        [
            "mlcg-train_h5",
            "fit",
            "--config",
            str(test_dir / "pytest_training.yaml"),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)
    assert result.returncode == 0, f"Failed training model with {model_yaml}"
    # Clean after every iteration: always run also if the test fails
    if test_dir.exists():
        for filename in os.listdir(test_dir):
            file_path = os.path.join(test_dir, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                rmtree(file_path)
