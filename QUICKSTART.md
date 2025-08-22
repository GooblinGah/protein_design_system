# Protein Design System - Quick Start Guide

## Prerequisites

- Python 3.10+
- PyTorch 2.3+ (with CUDA support recommended)
- 8GB+ GPU memory (for training)

## Installation

1. **Clone and setup**:
```bash
cd protein_design_system
pip install -r requirements.txt
```

2. **Verify installation**:
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import constraints.fsa; print('FSA engine loaded')"
```

## Quick Examples

### 1. DSL Compilation
```python
from dsl.compiler import DSLCompiler

# Define protein specification
dsl_spec = {
    "length": [230, 330],
    "motifs": [
        {
            "name": "esterase_gxsxg",
            "dfa": "G X S X G",
            "window": [50, 90]
        }
    ],
    "tags": ["pH~7", "secreted"]
}

# Compile to constraints
compiler = DSLCompiler()
constraints = compiler.compile_to_constraints(dsl_spec)
print(f"Compiled {len(constraints['dfa_tables'])} motifs")
```

### 2. FSA Constraint Checking
```python
import torch
from constraints.fsa import FSAConstraintEngine, create_dfa_table

# Create constraint engine
engine = FSAConstraintEngine()

# Create DFA table for GXSXG motif
dfa_table = create_dfa_table("G X S X G")
windows = torch.tensor([[[50, 90]]])  # [B, M, 2]

# Check constraints at position 60
pos1 = torch.tensor([60])
allowed = engine.allowed_tokens(0, windows, [dfa_table], pos1)
print(f"Allowed AAs at pos 60: {allowed.sum().item()}/20")
```

### 3. Tokenization
```python
from models.tokenizer import ProteinTokenizer

# Initialize tokenizer
tokenizer = ProteinTokenizer()

# Encode protein sequence
sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
encoded = tokenizer.encode(sequence)
print(f"Encoded length: {len(encoded)}")

# Decode back
decoded = tokenizer.decode(encoded)
print(f"Decoded matches: {sequence == decoded}")
```

## Configuration

Edit `config.yaml` to customize:
- Model architecture (layers, dimensions)
- Training parameters (learning rates, batch sizes)
- Novelty thresholds (identity caps)
- Controller settings (pacing thresholds)

## Project Structure

```
protein_design_system/
├── constraints/          # FSA constraint engine
├── dsl/                 # DSL parser/compiler
├── models/              # Neural network models
├── examples/            # Usage examples
├── tests/               # Test scripts
└── config.yaml         # Configuration
```

## Next Steps

1. **Run examples**: `python examples/dsl_example.py`
2. **Run tests**: `python tests/test_fsa.py`
3. **Customize config**: Edit `config.yaml`
4. **Extend functionality**: Add new motifs, constraints

## Support

- Check `IMPLEMENTATION_STATUS.md` for progress
- Review `README.md` for detailed documentation
- Refer to original specification for requirements
