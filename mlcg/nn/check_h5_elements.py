import h5py
import numpy as np

def check_1L2Y_embeds():
    h5_file_path = "/srv/data/kamenrur95/mlcg/examples/h5_pl/single_molecule/1L2Y_prior_tag.h5"
    
    with h5py.File(h5_file_path, 'r') as f:
        print("=== Complete File Structure ===")
        
        def print_everything(name, obj):
            print(f"{name}: {type(obj).__name__}")
            if hasattr(obj, 'shape'):
                print(f"  Shape: {obj.shape}")
            if hasattr(obj, 'dtype'):
                print(f"  Dtype: {obj.dtype}")
            
            # Check attributes for this object
            if hasattr(obj, 'attrs') and len(obj.attrs) > 0:
                print(f"  Attributes: {list(obj.attrs.keys())}")
                for attr_name, attr_value in obj.attrs.items():
                    print(f"    {attr_name}: {attr_value}")
                    if 'embed' in attr_name.lower():
                        unique_vals = np.unique(attr_value)
                        print(f"      Unique values: {unique_vals}")
                        ELEMENT_MAP = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S", 15: "P"}
                        if np.max(unique_vals) < 120:
                            elements = [ELEMENT_MAP.get(int(x), f"El_{int(x)}") for x in unique_vals]
                            print(f"      As elements: {elements}")
        
        f.visititems(print_everything)
        
        print("\n=== Specific Check for Expected Structure ===")
        
        # Check the expected structure: /1L2Y/1L2Y/attrs/cg_embeds
        if '1L2Y' in f:
            metaset = f['1L2Y']
            print(f"Metaset '1L2Y' found")
            
            if '1L2Y' in metaset:
                mol_group = metaset['1L2Y']
                print(f"Molecule '1L2Y' found")
                print(f"Datasets: {list(mol_group.keys())}")
                print(f"Attributes: {list(mol_group.attrs.keys())}")
                
                # Look for embeds in attributes
                for attr_name in mol_group.attrs.keys():
                    attr_value = mol_group.attrs[attr_name]
                    print(f"\nAttribute '{attr_name}':")
                    print(f"  Type: {type(attr_value)}")
                    print(f"  Value: {attr_value}")
                    
                    if hasattr(attr_value, 'shape'):
                        print(f"  Shape: {attr_value.shape}")
                        
                    if isinstance(attr_value, np.ndarray) and len(attr_value.shape) == 1:
                        unique_vals = np.unique(attr_value)
                        print(f"  Unique values: {unique_vals}")
                        
                        if len(unique_vals) < 20 and 'embed' in attr_name.lower():
                            ELEMENT_MAP = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S", 15: "P", 9: "F", 17: "Cl"}
                            elements = [ELEMENT_MAP.get(int(x), f"Element_{int(x)}") for x in unique_vals]
                            print(f"  🎉 POTENTIAL EMBEDS! As elements: {elements}")
                            return elements
        
        return None

if __name__ == "__main__":
    elements = check_1L2Y_embeds()
    if elements:
        print(f"\n✅ Found embeds! Update your config with:")
        print(f"type_names: {elements}")
    else:
        print(f"\n❌ No embeds found in expected locations")