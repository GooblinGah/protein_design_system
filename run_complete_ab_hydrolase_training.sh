#!/bin/bash

# Complete workflow for AB Hydrolase training

echo "============================================"
echo "ALPHA/BETA HYDROLASE MODEL TRAINING PIPELINE"
echo "============================================"

# Step 1: Setup environment
echo "[1/6] Setting up environment..."
conda activate protein_design || {
    conda create -n protein_design python=3.9 -y
    conda activate protein_design
    pip install -r requirements.txt
}

# Step 2: Collect data
echo "[2/6] Collecting AB Hydrolase data..."
python scripts/collect_ab_hydrolase_data.py || exit 1

# Step 3: Prepare training data
echo "[3/6] Preparing training splits..."
python scripts/prepare_ab_hydrolase_training.py || exit 1

# Step 4: Build retrieval index
echo "[4/6] Building retrieval index..."
python scripts/build_retrieval_index.py \
    --data data/processed/train.parquet \
    --output data/processed/retrieval_index || exit 1

# Step 5: Verify dataset
echo "[5/6] Verifying dataset quality..."
python scripts/verify_ab_hydrolase_dataset.py || exit 1

# Step 6: Start training
echo "[6/6] Starting model training..."
python train_ab_hydrolase_model.py \
    --config config_ab_hydrolase.yaml \
    --output-dir runs/ab_hydrolase_$(date +%Y%m%d_%H%M%S) \
    --gpu 0

echo "Training pipeline complete!"
