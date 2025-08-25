# Protein Design System

A complete, production-ready system for designing proteins using deep learning with constraint satisfaction, exemplar retrieval, and provenance tracking.

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Create conda environment
conda create -n protein_design python=3.9 -y
conda activate protein_design

# Install dependencies
pip install -r requirements.txt

# Install bioinformatics tools
conda install -c bioconda hmmer muscle cd-hit
```

### 2. Run End-to-End Training (Alpha/Beta Hydrolases)
```bash
# Make scripts executable
chmod +x scripts/*.sh run_complete_ab_hydrolase_training.sh

# Run complete pipeline
./run_complete_ab_hydrolase_training.sh
```

### 3. Manual Step-by-Step
```bash
# Collect data
python scripts/collect_ab_hydrolase_data.py

# Prepare training data with clustering
python scripts/prepare_ab_hydrolase_training.py

# Build retrieval index
python scripts/build_retrieval_index.py --data data/processed/train.parquet --output data/processed/retrieval_index

# Start training
python train_ab_hydrolase_model.py --config config_ab_hydrolase.yaml
```

## What This System Does

### Core Capabilities
- **Protein Sequence Generation**: Generate novel protein sequences from natural language prompts
- **Constraint Satisfaction**: Enforce structural and functional constraints using FSA-based validation
- **Exemplar Retrieval**: Use FAISS to find similar sequences for copy mechanisms
- **Identity Governance**: Hard ≤0.70 identity cap during decoding to ensure novelty
- **Provenance Tracking**: Hash-chained ledger for all design decisions

### Specialized for Alpha/Beta Hydrolases
- **Domain-Specific Data**: 30k-50k curated sequences from UniProt
- **Motif Detection**: GXSXG nucleophile elbow, catalytic triad, oxyanion hole
- **Structural Validation**: PDB integration for high-resolution structures
- **Curriculum Learning**: Progressive introduction of motifs and exemplars

## Expected Performance

```python
expected_performance = {
    "dataset_size": "30,000-50,000 sequences",
    "training_time": "48-72 hours on single A100",
    "expected_metrics": {
        "gxsxg_motif_accuracy": ">95%",
        "constraint_satisfaction": ">90%",
        "novelty_rate": ">80% (<70% identity)",
        "valid_fold_prediction": ">75%"
    },
    "convergence": "10-12 epochs",
    "final_loss": "<1.5"
}
```

## System Architecture

### Data Pipeline
```
Raw Sequences (UniProt) → Motif Extraction → Clustering (CD-HIT) → Leakage-Aware Splits
```

### Model Architecture
```
Prompt + DSL → FSA Constraints → Pointer Generator → Identity Validation → Output
```

### Key Components
- **`data/`**: Dataset handling, preprocessing, retrieval, alignment
- **`models/`**: Tokenizer, decoder, FSA constraints, provenance
- **`evaluation/`**: Metrics, validation, constraint checking
- **`training/`**: Trainer with curriculum learning
- **`scripts/`**: Data collection, clustering, validation

## Configuration

### Main Config (`config.yaml`)
- Model architecture parameters
- Training hyperparameters
- Evaluation thresholds

### AB Hydrolase Config (`config_ab_hydrolase.yaml`)
- Domain-specific settings
- Motif definitions
- Curriculum stages

### Validation
```bash
# Validate configuration
python scripts/validate_config.py --config config_ab_hydrolase.yaml --generate-manifest
```

## 📁 File Structure

```
protein_design_system/
├── data/                          # Data handling modules
│   ├── dataset.py                # ProteinDesignDataset
│   ├── retrieval.py              # ExemplarRetriever
│   ├── alignment.py              # AlignmentProcessor
│   └── preprocessing.py          # Data preprocessing
├── models/                        # Model components
│   ├── decoder/                  # Decoder implementations
│   ├── tokenizer.py              # Protein tokenizer
│   └── provenance/               # Provenance tracking
├── evaluation/                    # Evaluation and validation
│   ├── metrics.py                # ProteinDesignEvaluator
│   └── validator.py              # ConstraintValidator
├── training/                      # Training infrastructure
│   └── trainer.py                # Main trainer
├── scripts/                       # Utility scripts
│   ├── collect_ab_hydrolase_data.py
│   ├── cluster_splits.py         # CD-HIT clustering
│   └── validate_config.py        # Configuration validation
├── config_ab_hydrolase.yaml      # AB Hydrolase config
├── requirements.txt               # Dependencies
└── README.md                     # This file
```

## Testing & CI

### Run Tests
```bash
# Test imports
python -c "from data import ProteinDesignDataset; print('Data modules work')"

# Test evaluation
python -c "from evaluation import ProteinDesignEvaluator; print('Evaluation works')"

# Test configuration
python scripts/validate_config.py --config config_ab_hydrolase.yaml
```

### GitHub Actions
- Automatic testing on push/PR
- Unit tests for critical components
- Configuration validation
- Model initialization tests

## 🚨 Critical Features

### Hard Identity Cap During Decoding
```python
# Fast identity estimation during beam expansion
max_id = fast_identity_estimate(new_prefix, exemplars_tokens)
if max_id >= cfg.eval.max_identity_threshold:
    score = -inf  # prune beam

# Full alignment validation at finalization
final_identity = compute_identity(sequence, exemplars)
if final_identity > threshold:
    sequence_rejected = True
```

### Leakage-Aware Dataset Splits
```bash
# Use CD-HIT clustering at 30% identity
python scripts/cluster_splits.py --sequences data/raw/sequences.csv --identity 0.30

# Creates train/val/test with no sequence leakage
# Saves cluster map for reproducibility
```

### Provenance Tracking
```python
# Log all generation decisions
ledger.log_generation_decision(
    prompt=prompt,
    dsl_constraints=constraints,
    exemplars=exemplars,
    generated_sequences=sequences,
    beam_pruning_info=pruning_info,
    identity_constraints=identity_config
)
```

## Training Workflow

### 1. Data Collection
- UniProt API integration
- Motif detection and validation
- Structural data from PDB

### 2. Data Preparation
- CD-HIT clustering at 30% identity
- Leakage-aware train/val/test splits
- Quality filtering and validation

### 3. Model Training
- Curriculum learning stages
- Motif placement emphasis
- Exemplar integration

### 4. Evaluation
- Constraint satisfaction metrics
- Motif accuracy tracking
- Novelty validation

## Monitoring & Debugging

### Training Logs
```bash
# Monitor training progress
tail -f runs/experiment_1/training.log

# Check metrics
cat runs/experiment_1/metrics.json
```

### Provenance Ledger
```bash
# View design decisions
cat ledger.jsonl | jq '.entry_type'

# Check identity violations
cat ledger.jsonl | jq 'select(.entry_type == "identity_violation")'
```

### Configuration Validation
```bash
# Preflight check
python scripts/validate_config.py --config config_ab_hydrolase.yaml

# Generate run manifest
python scripts/validate_config.py --config config_ab_hydrolase.yaml --generate-manifest
```

## 🚀 Performance Optimization

### Memory Management
```yaml
# Reduce batch size for large models
training:
  batch_size: 16  # Reduce from 32
  batch_tokens: 32768  # Reduce from 65536
```

### Embedding Models
```yaml
# Use smaller model for indexing
retrieval:
  embedding_model: "esm2_t12_35M_UR50D"  # 35M vs 650M parameters
  embedding_dim: 480  # Reduced dimension
```

### GPU Optimization
```yaml
# Enable mixed precision
training:
  amp: true  # Automatic Mixed Precision
  fp16: true # 16-bit training
```

## 🆘 Troubleshooting

### Common Issues

#### 1. UniProt API Limits
```bash
# Error: HTTP 429 (Too Many Requests)
# Solution: Increase sleep time in collect_ab_hydrolase_data.py
time.sleep(1.0)  # Increase from 0.5s to 1.0s
```

#### 2. Memory Issues
```yaml
# Reduce batch size in config
training:
  batch_size: 16  # Reduce from 32
```

#### 3. Missing External Tools
```bash
# Install bioinformatics tools
conda install -c bioconda hmmer muscle cd-hit

# Or use pyhmmer alternative
pip install pyhmmer
```

### Getting Help
1. Check the logs: `tail -f runs/experiment_1/training.log`
2. Validate configuration: `python scripts/validate_config.py --config config.yaml`
3. Test components: `python -c "from data import ProteinDesignDataset"`
4. Check CI status: GitHub Actions tab

## 📚 Documentation

- **`AB_HYDROLASE_PIPELINE.md`**: Complete AB Hydrolase training guide
- **`RUN_TRAINING.md`**: General training workflow documentation
- **`IMPLEMENTATION_SUMMARY.md`**: Technical implementation details

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure CI passes
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **ESM2 Models**: Facebook AI Research
- **FAISS**: Facebook AI Research
- **CD-HIT**: Weizhong Li Lab
- **HMMER**: Sean Eddy Lab
- **MUSCLE**: Robert Edgar

## Next Steps

1. **Run the quickstart** to get familiar with the system
2. **Customize for your protein family** by modifying the data collection
3. **Scale up training** with multi-GPU or distributed training
4. **Add new constraints** by extending the FSA constraint engine
5. **Integrate with experimental validation** pipelines

---

**Ready to design proteins? Start with the [AB Hydrolase Pipeline](AB_HYDROLASE_PIPELINE.md) for a complete example!**
