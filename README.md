

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
