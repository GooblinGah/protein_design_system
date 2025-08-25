#!/usr/bin/env python3
"""
Train protein design model on Alpha/Beta Hydrolase dataset.

This script uses the complete ProteinDesignTrainer to train a model
on the AB Hydrolase dataset with proper data loading, retrieval,
and training loops.

Usage:
    python train_ab_hydrolase_model.py --config config_ab_hydrolase.yaml
    python train_ab_hydrolase_model.py --config config_ab_hydrolase.yaml --resume runs/experiment_1/checkpoints/best.pt
"""

import argparse
import yaml
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from training.trainer import ProteinDesignTrainer


def main():
    parser = argparse.ArgumentParser(description="Train AB Hydrolase protein design model")
    parser.add_argument("--config", type=str, default="config_ab_hydrolase.yaml", 
                       help="Path to YAML config file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume training from")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Override output directory from config")
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"📋 Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override output directory if specified
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
                print(f"Training configuration:")
    print(f"   Model: {config.get('model', {}).get('d_model', 'N/A')}d, "
          f"{config.get('model', {}).get('n_layers', 'N/A')} layers")
    print(f"   Data: {config.get('data', {}).get('train_path', 'N/A')}")
    print(f"   Output: {config.get('output_dir', 'N/A')}")
    print(f"   Retrieval: {'Enabled' if config.get('retrieval', {}).get('use_retrieval', False) else 'Disabled'}")
    
    # Create and run trainer
    trainer = ProteinDesignTrainer(config)
    trainer.fit(resume_ckpt=args.resume)
    
                print("Training completed successfully!")


if __name__ == "__main__":
    main()
