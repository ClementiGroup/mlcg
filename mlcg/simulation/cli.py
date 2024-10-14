from typing import Any, List, Dict, Tuple, Sequence
import torch
from jsonargparse import (
    ActionConfigFile,
    ArgumentParser,
    class_from_function,
)
from jsonargparse.typing import Path_fr

from . import (
    _Simulation,
    LangevinSimulation,
    PTSimulation,
    OverdampedSimulation,
)
from ..data import AtomicData
from ..nn import load_and_adapt_old_checkpoint
from ..utils import dump_yaml


def parse_simulation_config(
    simulation_class,
    description: str = "Simulation command line tool",
    parser_kwargs: Dict[str, Any] = None,
    subclass_mode: bool = False,
) -> Tuple[torch.nn.Module, List[AtomicData], _Simulation]:
    """Utility to parse a configuration file for run MD simulations with the
    classes defined in library.

    Parameters
    ----------
    simulation_class :
        a child class of _Simulation
    description : str, optional
        cli description, by default "Simulation command line tool"
    parser_kwargs : Dict[str, Any], optional
        more arguments to the parser, by default None
    subclass_mode: bool, optional
        Whether allow any subclass of the given class. So if true,
        one could provide `_Simulation` as input here but define `LangevinSimulation`
        in the input file, e.g.
        `{"simulation":"class_path": "mlcg.simulation.LangevinSimulation", "init_args": {.....}}`.
    Returns
    -------
    model, atomic_data_list, simulation_obj
    """
    parser_kwargs = {} if parser_kwargs is None else parser_kwargs
    parser_kwargs.update({"description": description})
    parser = SimulationParser(**parser_kwargs)
    parser.add_simulation_args(
        simulation_class, "simulation", subclass_mode=subclass_mode
    )

    parser.add_argument(
        "-tm",
        "--betas",
        metavar="FN",
        type=list,
        help="inverse temperature(s) (1/kBT) at which the simulation will run",
    )

    parser.add_argument(
        "-mf",
        "--model_file",
        metavar="FN",
        type=Path_fr,
        help="path to the pytorch model file (including the priors) in pytorch format",
    )

    parser.add_argument(
        "-sf",
        "--structure_file",
        metavar="FN",
        type=Path_fr,
        help="path to the starting configurations in pytorch format",
    )

    parser.add_argument(
        "-p",
        "--profile",
        type=str,
        default="",
        help="filename for the output profiling file",
    )

    config = parser.parse_args()
    # save config
    exported_config = {}
    for k, v in config.items():
        # Path_fr must be converted to string otherwise they can't be saved
        if isinstance(v, Path_fr):
            exported_config[k] = str(v)
        # redundant to save the path to the original config
        elif k == "config":
            continue
        else:
            exported_config[k] = v
    out_name = exported_config["simulation"]["filename"]
    dump_yaml(f"{out_name}_config.yaml", exported_config)
    # Sanitize PTSimulation kwargs
    if simulation_class == PTSimulation:
        config["simulation"].pop("sim_subroutine", None)
        config["simulation"].pop("sim_subroutine_interval", None)
        config["simulation"].pop("save_subroutine", None)

    model_fn = config.pop("model_file")
    model = load_and_adapt_old_checkpoint(
        (model_fn if isinstance(model_fn, str) else model_fn())
    )

    structures_fn = config.pop("structure_file")
    initial_data_list = torch.load(
        (structures_fn if isinstance(structures_fn, str) else structures_fn())
    )

    config_init = parser.instantiate_classes(config)
    simulation = config_init.get("simulation")
    betas = config.pop("betas")
    if len(betas) == 1:
        betas = float(betas[0])
    profile = config.pop("profile")

    return model, initial_data_list, betas, simulation, profile


class ConfigurationException(Exception):
    """
    Exception used to inform users
    """


class SimulationParser(ArgumentParser):
    """Extension of jsonargparse's ArgumentParser to parse simulation arguments."""

    def __init__(
        self,
        *args: Any,
        parse_as_dict: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        For full details of accepted arguments see `ArgumentParser.__init__
        <https://jsonargparse.readthedocs.io/en/stable/#jsonargparse.core.ArgumentParser.__init__>`_.

        """

        super().__init__(*args, parse_as_dict=parse_as_dict, **kwargs)

        self.add_argument(
            "--config",
            action=ActionConfigFile,
            help="Path to a configuration file in json or yaml format.",
        )

    def add_simulation_args(
        self,
        simulation_class: _Simulation,
        nested_key: str,
        subclass_mode: bool = False,
        required: bool = True,
    ) -> List[str]:
        """
        Adds arguments from a lightning class to a nested key of the parser

        Parameters
        ----------

        simulation_class:
            A callable or any subclass of {_Simulation}.
                nested_key:
            Name of the nested namespace to store arguments.
        subclass_mode:
            Whether allow any subclass of the given class. So if true,
            one could provide `_Simulation` as input here but define `LangevinSimulation` in the input file, e.g. `{"simulation":"class_path": "mlcg.simulation.LangevinSimulation", "init_args": {.....}}`.
        """
        if callable(simulation_class) and not isinstance(
            simulation_class, type
        ):
            simulation_class = class_from_function(simulation_class)

        if isinstance(simulation_class, type) and issubclass(
            simulation_class, (_Simulation)
        ):
            if subclass_mode:
                return self.add_subclass_arguments(
                    simulation_class,
                    nested_key,
                    fail_untyped=False,
                    required=required,
                )
            return self.add_class_arguments(
                simulation_class,
                nested_key,
                fail_untyped=False,
                instantiate=True,
                sub_configs=True,
            )
        raise ConfigurationException(
            f"Cannot add arguments from: {simulation_class}. You should provide either a callable or a subclass of: "
            "Trainer, LightningModule, LightningDataModule, or Callback."
        )
