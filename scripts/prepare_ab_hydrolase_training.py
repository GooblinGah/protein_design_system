#!/usr/bin/env python3
"""
Prepare AB Hydrolase data for training with leakage-aware splits.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List
import logging
import argparse

# Import clustering functionality
try:
    from scripts.cluster_splits import SequenceClusterer
except ImportError:
    from cluster_splits import SequenceClusterer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ABHydrolaseDataPrep:
    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(exist_ok=True, parents=True)
    
    def prepare_training_data(self, use_clustering: bool = True, identity_threshold: float = 0.30):
        """Main preparation pipeline with optional clustering."""
        
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
        
        if use_clustering:
            logger.info("Using CD-HIT clustering for leakage-aware splits...")
            train_df, val_df, test_df = self._create_clustered_splits(complete_df, identity_threshold)
        else:
            logger.info("Using random splits (may have sequence leakage)...")
            train_df, val_df, test_df = self._create_random_splits(complete_df)
        
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
            'max_length': complete_df['length'].max(),
            'split_method': 'clustered' if use_clustering else 'random',
            'identity_threshold': identity_threshold if use_clustering else None
        }
        
        with open(self.processed_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return train_df, val_df, test_df
    
    def _create_clustered_splits(self, complete_df: pd.DataFrame, identity_threshold: float):
        """Create leakage-aware splits using CD-HIT clustering."""
        
        # Initialize clusterer
        clusterer = SequenceClusterer()
        
        # Extract sequences and IDs
        sequences = complete_df['sequence'].tolist()
        sequence_ids = complete_df['accession'].tolist()
        
        # Cluster sequences
        cluster_results = clusterer.cluster_sequences(
            sequences=sequences,
            sequence_ids=sequence_ids,
            identity_threshold=identity_threshold,
            output_dir=self.processed_dir
        )
        
        # Create leakage-aware splits
        splits = clusterer.create_leakage_aware_splits(
            cluster_results=cluster_results,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42
        )
        
        # Create dataframes from splits
        train_ids = set(splits['train'])
        val_ids = set(splits['val'])
        test_ids = set(splits['test'])
        
        train_df = complete_df[complete_df['accession'].isin(train_ids)]
        val_df = complete_df[complete_df['accession'].isin(val_ids)]
        test_df = complete_df[complete_df['accession'].isin(test_ids)]
        
        # Save clustering metadata
        with open(self.processed_dir / "clustering_metadata.json", 'w') as f:
            json.dump(splits, f, indent=2)
        
        logger.info(f"Created {len(cluster_results['cluster_map'])} clusters")
        logger.info(f"Cluster distribution: {splits['cluster_info']['cluster_distribution']}")
        
        return train_df, val_df, test_df
    
    def _create_random_splits(self, complete_df: pd.DataFrame):
        """Create random splits (legacy method)."""
        from sklearn.model_selection import train_test_split
        
        # Create train/val/test splits (70/15/15)
        train_df, temp_df = train_test_split(complete_df, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        
        return train_df, val_df, test_df

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Prepare AB Hydrolase training data")
    parser.add_argument("--raw-dir", default="data/raw", help="Raw data directory")
    parser.add_argument("--processed-dir", default="data/processed", help="Processed data directory")
    parser.add_argument("--no-clustering", action="store_true", help="Use random splits instead of clustering")
    parser.add_argument("--identity-threshold", type=float, default=0.30, help="CD-HIT identity threshold")
    
    args = parser.parse_args()
    
    prep = ABHydrolaseDataPrep(args.raw_dir, args.processed_dir)
    
    use_clustering = not args.no_clustering
    prep.prepare_training_data(
        use_clustering=use_clustering,
        identity_threshold=args.identity_threshold
    )
    
    if use_clustering:
        logger.info("✅ Leakage-aware splits created using CD-HIT clustering")
    else:
        logger.warning("⚠️ Random splits created (may have sequence leakage)")

if __name__ == "__main__":
    main()
