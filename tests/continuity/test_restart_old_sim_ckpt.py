import subprocess
import os
import pytest
from shutil import rmtree, copy2
from pathlib import Path
from torch import save as tsave
from mlcg.utils import load_yaml, dump_yaml
from mlcg.nn import load_and_adapt_old_checkpoint

_here = Path(__file__).parent
_ckpt_config_dir = _here / "simulation_ckpts"


# Create a temporary test directory
@pytest.fixture
def test_dir(tmp_path):
    yield tmp_path
    # Teardown: always run also if the test fails
    if tmp_path.exists():
        rmtree(tmp_path)


@pytest.mark.parametrize(
    "model_and_config_ckpt,sim_checkpoint",
    [
        (  ## CLN CA, multiple architectures
            _ckpt_config_dir / f"{arch}_cln_ca_specialized_model_and_config.pt",
            _ckpt_config_dir / f"{arch}_cln_ca_checkpoint.pt",
        )
        for arch in ["schnet", "mace", "so3krates"]
    ]
    + [
        (  ## NTL9 5B Allegro
            _ckpt_config_dir
            / "allegro_ntl9_5b_specialized_model_and_config.pt",
            _ckpt_config_dir / "allegro_ntl9_5b_checkpoint.pt",
        ),
        (  ## PaiNN from Klara old PyG
            _ckpt_config_dir / "painn_cln_5b_specialized_model_and_config.pt",
            _ckpt_config_dir / "painn_cln_5b_checkpoint.pt",
        ),
    ],
)
def test_restart_simulation(model_and_config_ckpt, sim_checkpoint, test_dir):
    r"""
    test reproducing:
     - model+prior combination
     - simulation config creation
     - NVT Langevin simulation
    """

    ## load and expand model and configurations

    old_simulated_model, old_configurations = load_and_adapt_old_checkpoint(
        model_and_config_ckpt
    )
    tsave(old_simulated_model, test_dir / "sim_extracted_model.pt")
    tsave(old_configurations, test_dir / "old_configurations.pt")
    old_simulation_ckpt = load_and_adapt_old_checkpoint(sim_checkpoint)
    tsave(old_simulation_ckpt, test_dir / "test_sim_checkpoint.pt")

    ## Prepare simulation config
    sim_config = load_yaml(_here / "base_sim_config.yaml")
    sim_config["structure_file"] = "old_configurations.pt"
    sim_config["simulation"]["read_checkpoint_file"] = True
    new_final_time = (
        sim_config["simulation"]["n_timesteps"]
        + sim_config["simulation"]["export_interval"] * 5
    )
    sim_config["simulation"]["n_timesteps"] = int(new_final_time)

    # priors can't be specialized twice
    sim_config["simulation"]["specialize_priors"] = False
    dump_yaml(test_dir / "sim_config.yaml", sim_config)

    _cmd = [
        "mlcg-nvt_langevin",
        "--config",
        "sim_config.yaml",
    ]

    result = subprocess.run(
        _cmd,
        text=True,
        encoding="utf-8",
        check=True,
        capture_output=True,
        cwd=test_dir,  # Run from test_dir
    )

    print("\n--- STDOUT ---\n", result.stdout)
    print("\n--- STDERR ---\n", result.stderr)

    result.check_returncode()

    # Clean after every iteration: always run also if the test fails
    if test_dir.exists():
        for filename in os.listdir(test_dir):
            file_path = os.path.join(test_dir, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                rmtree(file_path)
