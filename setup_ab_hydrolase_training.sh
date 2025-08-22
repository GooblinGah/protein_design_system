#!/bin/bash

echo "Setting up Alpha/Beta Hydrolase training dataset..."

# Create directories
mkdir -p data/raw data/processed logs

# Step 1: Collect data
echo "Step 1: Collecting AB Hydrolase data from UniProt..."
python scripts/collect_ab_hydrolase_data.py

# Step 2: Prepare training splits
echo "Step 2: Preparing training splits..."
python scripts/prepare_ab_hydrolase_training.py

# Step 3: Build retrieval index
echo "Step 3: Building exemplar retrieval index..."
python scripts/build_retrieval_index.py \
    --sequences data/processed/train.parquet \
    --output data/processed/retrieval_index

# Step 4: Verify data
echo "Step 4: Verifying dataset..."
python scripts/verify_ab_hydrolase_dataset.py

echo "Dataset preparation complete!"
echo "You can now run training with:"
echo "  python train_ab_hydrolase_model.py --config config_ab_hydrolase.yaml"
