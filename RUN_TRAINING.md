# Complete Training Workflow

## Overview
This document describes the complete workflow for training the Protein Design System with data preparation, exemplar retrieval, and comprehensive evaluation.

## 1. Setup Environment

```bash
# Create conda environment
conda create -n protein_design python=3.9
conda activate protein_design

# Install dependencies
pip install -r requirements.txt

# Install bioinformatics tools
conda install -c bioconda hmmer muscle cd-hit
```

## 2. Download Data

```bash
# Download protein sequences and annotations
bash scripts/download_data.sh

# Or manually download:
# - Swiss-Prot hydrolases from UniProt
# - Pfam annotations from Pfam
# - PROSITE motifs from ExPASy
```

## 3. Prepare Training Data

```bash
# Process raw data into training format
python scripts/prepare_data.py \
    --fasta data/raw/swissprot_hydrolases.fasta \
    --annotations data/raw/annotations.csv \
    --output-dir data/processed \
    --build-retrieval
```

This will:
- Load and filter sequences by length
- Generate prompt-sequence pairs
- Extract motifs and create DSL specifications
- Create train/val/test splits
- Build FAISS retrieval index with ESM2 embeddings

## 4. Train Model

### Full training with data preparation:
```bash
python train_full_system.py \
    --config config.yaml \
    --prepare-data \
    --output-dir runs/experiment_1
```

### Training with existing data:
```bash
python train_full_system.py \
    --config config.yaml \
    --output-dir runs/experiment_1
```

### Resume from checkpoint:
```bash
python train_full_system.py \
    --config config.yaml \
    --resume runs/experiment_1/checkpoint_epoch_5.pt \
    --output-dir runs/experiment_1
```

## 5. Monitor Training

```bash
# Check training logs
tail -f runs/experiment_1/training.log

# Monitor metrics with tensorboard (if configured)
tensorboard --logdir runs/experiment_1/tensorboard
```

## 6. Evaluate Model

```bash
# Run comprehensive evaluation
python evaluate_model.py \
    --model-path runs/experiment_1/final_model.pt \
    --test-data data/processed/test.parquet \
    --output evaluation_results.json
```

## Training Components

### Data Module (`data/`)
- **ProteinDesignDataset**: Main dataset class with exemplar retrieval
- **ProteinDatasetBuilder**: Preprocesses raw FASTA/CSV into training format
- **ExemplarRetriever**: FAISS-based retrieval with ESM2 embeddings
- **AlignmentProcessor**: MUSCLE/HMMER integration for sequence alignment

### Evaluation Module (`evaluation/`)
- **ProteinDesignEvaluator**: Comprehensive metrics (constraints, motifs, liabilities, novelty)
- **ConstraintValidator**: Validates sequences against design constraints

### Training Integration
- **Exemplar Integration**: Retrieves similar sequences for pointer mechanism
- **Alignment Features**: Provides conservation scores and consensus columns
- **Curriculum Learning**: Progressive introduction of exemplars and losses

## Configuration

### Data Configuration
```yaml
data:
  min_length: 220
  max_length: 350
  exemplars_per_sample: 10
  use_exemplars: true
```

### Retrieval Configuration
```yaml
retrieval:
  use_retrieval: true
  embedding_model: "facebook/esm2_t33_650M_UR50D"
  embedding_dim: 1280
  top_k: 10
```

### Alignment Configuration
```yaml
alignment:
  use_alignment: true
  hmmbuild_path: "/usr/bin/hmmbuild"
  muscle_path: "/usr/bin/muscle"
```

## Expected Outputs

### Data Processing
- `data/processed/train.parquet` - Training dataset
- `data/processed/val.parquet` - Validation dataset  
- `data/processed/test.parquet` - Test dataset
- `data/processed/retrieval_index/` - FAISS index and metadata

### Training
- `runs/experiment_1/checkpoint_*.pt` - Model checkpoints
- `runs/experiment_1/final_model.pt` - Final trained model
- `runs/experiment_1/training.log` - Training logs
- `runs/experiment_1/metrics.json` - Training metrics

## Troubleshooting

### Common Issues

1. **Memory Issues with ESM2**
   - Use smaller model: `facebook/esm2_t12_35M_UR50D`
   - Reduce batch size in config

2. **HMMER/MUSCLE Not Found**
   - Install via conda: `conda install -c bioconda hmmer muscle`
   - Update paths in config.yaml

3. **FAISS Installation Issues**
   - Use CPU version: `pip install faiss-cpu`
   - Or GPU version: `pip install faiss-gpu`

4. **Data Format Issues**
   - Ensure FASTA files are valid
   - Check CSV column names match expected format

### Performance Tips

1. **Faster Training**
   - Use smaller ESM2 model for retrieval
   - Reduce exemplar count per sample
   - Use CPU FAISS for small datasets

2. **Better Retrieval**
   - Increase FAISS index size
   - Use hierarchical clustering for large datasets
   - Cache exemplar embeddings

## Next Steps

After training:
1. **Model Analysis**: Analyze attention patterns and exemplar usage
2. **Constraint Validation**: Test on novel constraint combinations
3. **Novelty Assessment**: Evaluate against larger protein databases
4. **Production Deployment**: Optimize for inference speed

## Support

For issues or questions:
1. Check the logs in `runs/experiment_1/`
2. Verify data format and paths
3. Ensure all dependencies are installed
4. Review configuration parameters
