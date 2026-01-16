from typing import List, Union, Any
import torch
import warnings
from copy import deepcopy
from torch_geometric.data.collate import collate

from ..data.atomic_data import AtomicData
from .langevin import LangevinSimulation
from ..nn import Constraint, GradientsOut
from ..neighbor_list.neighbor_list import make_neighbor_list


class LangevinConstraint(LangevinSimulation):
    r"""Langevin simulation class with constraint priors.
    
    This class extends LangevinSimulation by automatically adding a constraint
    prior to the model and the necessary neighbor lists to configurations.
    
    The constraint prior restricts certain atom types to remain within a 
    specified radius (r_s) during the simulation.
    
    Parameters
    ----------
    r_s : float
        Constraint radius in Angstroms. Atoms of the specified types will be
        constrained to stay within this radius.
    k_s : float
        Scaling factor for the constraint potential.
    constraint_atom_types : List[int]
        List of atom type indices that should be constrained.
    friction : float, default=1e-3
        Friction value for Langevin scheme, in units of inverse time.
    **kwargs
        Additional arguments passed to LangevinSimulation parent class.
        
    """
    
    def __init__(
        self,
        r_s: float,
        k_s: float,
        constraint_atom_types: List[int],
        friction: float = 1e-3,
        **kwargs: Any
    ):
        super(LangevinConstraint, self).__init__(friction=friction, **kwargs)
        
        if r_s <= 0:
            raise ValueError(f"r_s must be positive, got {r_s}")
        if k_s <= 0:
            raise ValueError(f"k_s must be positive, got {k_s}")
        if not constraint_atom_types:
            raise ValueError("constraint_atom_types cannot be empty")
        
        self.r_s = r_s
        self.k_s = k_s
        self.constraint_atom_types = constraint_atom_types
        self._constraint_added = False
        
    def attach_model_and_configurations(
        self,
        model: torch.nn.Module,
        configurations: List[AtomicData],
        beta: Union[float, List[float]],
    ):
        """Attach model and configurations with automatic constraint setup.
        
        This method:
        1. Adds constraint neighbor lists to all configurations
        2. Creates a constraint prior
        3. Adds the constraint prior to the model
        4. Calls the parent class method to complete setup
        
        Parameters
        ----------
        model : torch.nn.Module
            Trained model used to generate simulation data
        configurations : List[AtomicData]
            List of AtomicData instances representing initial structures
        beta : Union[float, List[float]]
            Desired temperature(s) of the simulation
        """
        # First, add constraint neighbor lists to configurations
        configurations = self._add_constraint_neighbor_lists(configurations)
        
        # Create and add constraint prior to model
        model = self._add_constraint_prior(model)
        
        # Call parent method with modified model and configurations
        super(LangevinConstraint, self).attach_model_and_configurations(
            model, configurations, beta
        )
        
        self._constraint_added = True
        
        if self.log_interval is not None:
            msg = (
                f"Constraint prior added: r_s={self.r_s} Å, "
                f"with a scaling factor k_s={self.k_s}, "
                f"atom_types={self.constraint_atom_types}"
            )
            if self.log_type == "print":
                print(msg)
            elif self.log_type == "write" and hasattr(self, '_log_file'):
                with open(self._log_file, 'a') as f:
                    f.write(msg + "\n")
    
    def _add_constraint_neighbor_lists(
        self, 
        configurations: List[AtomicData]
    ) -> List[AtomicData]:
        """Add constraint neighbor lists to all configurations.
        
        Parameters
        ----------
        configurations : List[AtomicData]
            Original configurations without constraint neighbor lists
            
        Returns
        -------
        List[AtomicData]
            Configurations with constraint neighbor lists added
        """
        modified_configs = []
        n_configs = len(configurations)
        
        for i, config in enumerate(configurations):
            config_copy = deepcopy(config)
            N = config_copy.n_atoms.item() if hasattr(config_copy.n_atoms, 'item') else config_copy.n_atoms
            
            # Create mapping for this single configuration
            mapping = torch.arange(N)
            mapping_batch = torch.zeros(N, dtype=torch.long)
            
            # Create neighbor list
            nl = make_neighbor_list(
                tag="constraint",
                order=N,
                index_mapping=mapping,
                mapping_batch=mapping_batch,
                self_interaction=False,
            )
            
            # Add to configuration's neighbor list dictionary
            if not hasattr(config_copy, 'neighbor_list'):
                config_copy.neighbor_list = {}
            config_copy.neighbor_list["constraint"] = nl
            
            modified_configs.append(config_copy)
        
        return modified_configs
    
    def _add_constraint_prior(self, model: torch.nn.Module) -> torch.nn.Module:
        """Add constraint prior to the model.
        
        Parameters
        ----------
        model : torch.nn.Module
            Original model without constraint prior
            
        Returns
        -------
        torch.nn.Module
            Model with constraint prior added
        """
        # Create constraint prior
        constraint_prior = GradientsOut(
            Constraint(
                constraint_atom_types=self.constraint_atom_types, 
                r_s=self.r_s,
                k_s=self.k_s
            ),
            targets=["energy", "forces"]
        )
        
        # Check if model has a ModuleDict structure
        if hasattr(model, 'models') and isinstance(model.models, torch.nn.ModuleDict):
            # Add constraint to existing ModuleDict
            model_copy = deepcopy(model)
            model_copy.models["constraint"] = constraint_prior
        else:
            # Create new ModuleDict structure
            module_dict = torch.nn.ModuleDict()
            
            # If model has models attribute, copy them
            if hasattr(model, 'models'):
                for key in model.models:
                    module_dict[key] = model.models[key]
            else:
                # Wrap the entire model
                module_dict["main"] = model
            
            # Add constraint prior
            module_dict["constraint"] = constraint_prior
            
            # Create a simple wrapper model
            class ConstrainedModel(torch.nn.Module):
                def __init__(self, models):
                    super().__init__()
                    self.models = models
                    
                def forward(self, data):
                    for model in self.models.values():
                        data = model(data)
                    return data
                    
                def eval(self):
                    for model in self.models.values():
                        model.eval()
                    return self
            
            model_copy = ConstrainedModel(module_dict)
        
        return model_copy
    
    def simulate(self, overwrite: bool = False, prof=None) -> None:
        """Run the simulation with constraints.
        
        Parameters
        ----------
        overwrite : bool, default=False
            Set to True if you wish to overwrite any saved simulation data
        prof : optional
            Profiler object for performance monitoring
            
        Returns
        -------
        None
            Results are stored in class attributes (simulated_coords, etc.)
        """
        if not self._constraint_added:
            raise RuntimeError(
                "Constraint has not been added. Call "
                "attach_model_and_configurations() first."
            )
        
        # Call parent simulate method
        return super(LangevinConstraint, self).simulate(overwrite=overwrite, prof=prof)