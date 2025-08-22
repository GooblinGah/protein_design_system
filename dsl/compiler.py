import json
import numpy as np
from typing import Dict, List, Tuple, Any
from pathlib import Path
import torch
from constraints.fsa import create_dfa_table


class DSLCompiler:
    """Compiler for Protein Design DSL to constraint objects."""
    
    def __init__(self):
        self.aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"
    
    def parse_dsl(self, dsl_json: str) -> Dict[str, Any]:
        """
        Parse DSL JSON string into structured object.
        
        Args:
            dsl_json: JSON string with DSL specification
            
        Returns:
            Parsed DSL object
        """
        try:
            dsl_obj = json.loads(dsl_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid DSL JSON: {e}")
        
        # Validate required fields
        required_fields = ["length", "motifs"]
        for field in required_fields:
            if field not in dsl_obj:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate length specification
        if not isinstance(dsl_obj["length"], list) or len(dsl_obj["length"]) != 2:
            raise ValueError("Length must be [min, max] list")
        
        # Validate motifs
        if not isinstance(dsl_obj["motifs"], list):
            raise ValueError("Motifs must be a list")
        
        for motif in dsl_obj["motifs"]:
            if not isinstance(motif, dict):
                raise ValueError("Each motif must be a dictionary")
            if "name" not in motif or "dfa" not in motif or "window" not in motif:
                raise ValueError("Motif must have name, dfa, and window fields")
            if not isinstance(motif["window"], list) or len(motif["window"]) != 2:
                raise ValueError("Motif window must be [start, end] list")
        
        return dsl_obj
    
    def compile_to_constraints(self, dsl_obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compile DSL object to constraint objects.
        
        Args:
            dsl_obj: Parsed DSL object
            
        Returns:
            Compiled constraints
        """
        # Extract length constraints
        length_min, length_max = dsl_obj["length"]
        
        # Process motifs
        dfa_tables = []
        windows = []
        
        for motif in dsl_obj["motifs"]:
            # Create DFA table from pattern
            dfa_table = create_dfa_table(motif["dfa"], self.aa_alphabet)
            dfa_tables.append(dfa_table)
            
            # Store window bounds
            windows.append(motif["window"])
        
        # Convert to numpy arrays
        dfa_tables_np = [table.numpy() for table in dfa_tables]
        windows_np = np.array(windows, dtype=np.int32)
        
        # Create compiled output
        compiled = {
            "dfa_tables": dfa_tables_np,
            "windows": windows_np,
            "length": np.array([length_min, length_max], dtype=np.int32),
            "negatives": dsl_obj.get("negatives", []),
            "tags": dsl_obj.get("tags", [])
        }
        
        return compiled
    
    def save_compiled(self, compiled: Dict[str, Any], output_path: str):
        """
        Save compiled constraints to NPZ file.
        
        Args:
            compiled: Compiled constraints
            output_path: Path to save NPZ file
        """
        np.savez_compressed(
            output_path,
            dfa_tables=compiled["dfa_tables"],
            windows=compiled["windows"],
            length=compiled["length"],
            negatives=compiled["negatives"],
            tags=compiled["tags"]
        )
    
    def load_compiled(self, npz_path: str) -> Dict[str, Any]:
        """
        Load compiled constraints from NPZ file.
        
        Args:
            npz_path: Path to NPZ file
            
        Returns:
            Loaded constraints
        """
        data = np.load(npz_path, allow_pickle=True)
        
        return {
            "dfa_tables": data["dfa_tables"].tolist(),
            "windows": data["windows"],
            "length": data["length"],
            "negatives": data["negatives"].tolist() if data["negatives"].size > 0 else [],
            "tags": data["tags"].tolist() if data["tags"].size > 0 else []
        }


def compile_dsl_file(dsl_file: str, output_file: str):
    """
    Convenience function to compile DSL file to NPZ.
    
    Args:
        dsl_file: Path to DSL JSON file
        output_file: Path to output NPZ file
    """
    compiler = DSLCompiler()
    
    with open(dsl_file, 'r') as f:
        dsl_content = f.read()
    
    dsl_obj = compiler.parse_dsl(dsl_content)
    compiled = compiler.compile_to_constraints(dsl_obj)
    compiler.save_compiled(compiled, output_file)
    
    print(f"Compiled {dsl_file} to {output_file}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python compiler.py <dsl_file> <output_file>")
        sys.exit(1)
    
    compile_dsl_file(sys.argv[1], sys.argv[2])
