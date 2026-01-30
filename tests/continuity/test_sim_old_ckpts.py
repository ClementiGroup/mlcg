import subprocess
import os
import pytest
from shutil import rmtree
from pathlib import Path
from mlcg.utils import load_yaml, dump_yaml


_here = Path(__file__).parent
_ckpt_config_dir = _here / "model_ckpts"


# Create a temporary test directory
@pytest.fixture
def test_dir(tmp_path):
    yield tmp_path
    # Teardown: always run also if the test fails
    if tmp_path.exists():
        rmtree(tmp_path)


@pytest.mark.parametrize(
    "model_ckpt,prior,structures",
    [
        (  ## CLN CA
            _ckpt_config_dir / "schnet_cln_ca.ckpt",
            _ckpt_config_dir / "sparse_prior_cln_ca.pt",
            _ckpt_config_dir / "structures_cln_ca.pt",
        ),
        (  ##NTL9 5B Schnet
            _ckpt_config_dir / "schnet_ntl9_5b.ckpt",
            _ckpt_config_dir / "prior_ntl9_5b.pt",
            _ckpt_config_dir / "structures_ntl9_5b.pt",
        ),
        (  ## NTL9 5B Allegro
            _ckpt_config_dir / "allegro_ntl9_5b.ckpt",
            _ckpt_config_dir / "prior_ntl9_5b.pt",
            _ckpt_config_dir / "structures_ntl9_5b.pt",
        ),
        (  ## PaiNN from Klara old PyG
            _ckpt_config_dir / "painn_cln_5b.ckpt",
            _ckpt_config_dir / "sparse_prior_cln_5b.pt",
            _ckpt_config_dir / "structures_cln_5b.pt",
        ),
    ],
)
def test_simulation_pipeline(model_ckpt, prior, structures, test_dir):
    """
    Integration test reproducing:
      - model+prior combination
      - simulation config creation
      - NVT Langevin simulation
    """

    ## Combine model with priors
    _cmd = [
        "mlcg-combine_model",
        "--ckpt",
        str(model_ckpt),
        "--prior",
        str(prior),
        "--out",
        str(test_dir / "sim_extracted_model.pt"),
    ]

    result = subprocess.run(
        _cmd, encoding="utf-8", capture_output=True, text=True, cwd=_here
    )

    print("\n--- STDOUT ---\n", result.stdout)
    print("\n--- STDERR ---\n", result.stderr)

    result.check_returncode()

    ## Prepare simulation config
    sim_config = load_yaml(_here / "base_sim_config.yaml")
    sim_config["structure_file"] = str(structures)
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
