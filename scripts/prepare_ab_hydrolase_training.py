#!/usr/bin/env python3
"""
Prepare AB Hydrolase data for training
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ABHydrolaseDataPrep:
    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(exist_ok=True, parents=True)
    
    def prepare_training_data(self):
        """Main preparation pipeline"""
        
        # Load cleaned data
        sequences_df = pd.read_csv(self.raw_dir / "ab_hydrolases_sequences_clean.csv")
        annotations_df = pd.read_csv(self.raw_dir / "ab_hydrolases_annotations_clean.csv")
        motifs_df = pd.read_csv(self.raw_dir / "ab_hydrolases_motifs_clean.csv")
        prompts_df = pd.read_csv(self.raw_dir / "ab_hydrolases_prompts.csv")
        
        # Merge for complete dataset
        complete_df = prompts_df.merge(
            sequences_df[['accession', 'sequence', 'length']], 
            on='accession'
        )
        
        # Create train/val/test splits (70/15/15)
        train_df, temp_df = train_test_split(complete_df, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        
        # Save splits
        train_df.to_parquet(self.processed_dir / "train.parquet")
        val_df.to_parquet(self.processed_dir / "val.parquet")
        test_df.to_parquet(self.processed_dir / "test.parquet")
        
        logger.info(f"Training set: {len(train_df)} samples")
        logger.info(f"Validation set: {len(val_df)} samples")
        logger.info(f"Test set: {len(test_df)} samples")
        
        # Create metadata file
        metadata = {
            'num_train': len(train_df),
            'num_val': len(val_df),
            'num_test': len(test_df),
            'num_unique_sequences': complete_df['accession'].nunique(),
            'avg_sequence_length': complete_df['length'].mean(),
            'min_length': complete_df['length'].min(),
            'max_length': complete_df['length'].max()
        }
        
        with open(self.processed_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return train_df, val_df, test_df

if __name__ == "__main__":
    prep = ABHydrolaseDataPrep()
    prep.prepare_training_data()
