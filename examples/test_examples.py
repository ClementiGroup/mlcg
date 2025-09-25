import subprocess
from mlcg.utils import load_yaml, dump_yaml
from pathlib import Path
from glob import glob
import pytest
import os
from shutil import rmtree

training_yaml_list = sorted(glob("./input_yamls/train*.yaml"))
# removing the yaml with bonded exclusions at it requires a custom kernel
training_yaml_list = [file for file in training_yaml_list if "exclusions" not in file]
data_yaml_list = sorted(glob("./h5_pl/*molecule*/train*.yaml"))

@pytest.mark.parametrize(
    "model_yaml,reference_yaml",
    [
        (training_yaml_list[0],data_yaml_list[0])
    ]
)
def test_training_and_simulation_examples(model_yaml,reference_yaml,tmp_path):
    training_yaml = load_yaml(model_yaml)
    data_yaml = load_yaml(reference_yaml)
    training_yaml['data'] = data_yaml['data']
    #training_yaml['trainer']['default_root_dir'] = tmp_path
    training_yaml['trainer']['max_epochs'] = 2
    training_yaml['trainer'].pop("devices")
    reference_yaml_root = Path(reference_yaml).parent
    os.chdir(reference_yaml_root)
    if not os.path.isdir("sims"):
        os.mkdir("sims")
    dump_yaml("./pytest_training.yaml",training_yaml)
    with open("README.md") as f:
        sub_processes = [line.strip() for line in f.readlines() if line[0:5]=="mlcg-" and ".py" in line]
    for process in sub_processes:
        arg_list = process.split(" ")
        for idx,arg in enumerate(arg_list):
            if "train" in arg and ".yaml" in arg:
                arg_list[idx] = "pytest_training.yaml"
        result = subprocess.run(arg_list)
        assert result.returncode == 0
    rmtree("./ckpt")
    rmtree("./tensorboard")
    rmtree("./sims")
    
    