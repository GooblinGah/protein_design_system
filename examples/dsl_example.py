#!/usr/bin/env python3
"""
Example usage of the DSL compiler for protein design specifications.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dsl.compiler import DSLCompiler


def main():
    """Demonstrate DSL compilation with example specifications."""
    
    # Example 1: Lipase with GXSXG motif
    lipase_dsl = {
        "length": [230, 330],
        "motifs": [
            {
                "name": "lipase_gxsxg",
                "dfa": "G X S X G",
                "dfa": "G X S X G",
                "window": [50, 90]
            }
        ],
        "negatives": ["signal_peptide", "low_complexity"],
        "tags": ["pH~7", "secreted"]
    }
    
    # Example 2: Esterase with multiple motifs
    esterase_dsl = {
        "length": [280, 380],
        "motifs": [
            {
                "name": "esterase_gxsxg",
                "dfa": "G X S X G",
                "window": [60, 100]
            },
            {
                "name": "catalytic_triad",
                "dfa": "S D H",
                "window": [150, 200]
            }
        ],
        "negatives": ["transmembrane", "nuclear_localization"],
        "tags": ["cytoplasmic", "pH~6.5"]
    }
    
    # Initialize compiler
    compiler = DSLCompiler()
    
    print("Compiling lipase specification...")
    try:
        lipase_compiled = compiler.compile_to_constraints(lipase_dsl)
        print("✅ Lipase compiled successfully")
        print(f"   - Length range: {lipase_compiled['length']}")
        print(f"   - Number of motifs: {len(lipase_compiled['dfa_tables'])}")
        print(f"   - Tags: {lipase_compiled['tags']}")
    except Exception as e:
        print(f"❌ Lipase compilation failed: {e}")
    
    print("\nCompiling esterase specification...")
    try:
        esterase_compiled = compiler.compile_to_constraints(esterase_dsl)
        print("✅ Esterase compiled successfully")
        print(f"   - Number of motifs: {len(esterase_compiled['dfa_tables'])}")
        print(f"   - Tags: {esterase_compiled['tags']}")
    except Exception as e:
        print(f"❌ Esterase compilation failed: {e}")
    
    print("\nDSL compilation examples completed!")


if __name__ == "__main__":
    main()
