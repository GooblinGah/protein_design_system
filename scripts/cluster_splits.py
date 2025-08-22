#!/usr/bin/env python3
"""
Cluster protein sequences using CD-HIT and create leakage-aware splits.
"""

import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple
import tempfile
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SequenceClusterer:
    """Cluster sequences using CD-HIT and create splits."""
    
    def __init__(self, cdhit_path: str = "cd-hit"):
        self.cdhit_path = cdhit_path
        self.check_cdhit_installation()
    
    def check_cdhit_installation(self):
        """Check if CD-HIT is available."""
        try:
            result = subprocess.run([self.cdhit_path, "-h"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError("CD-HIT not working properly")
            logger.info("CD-HIT installation verified")
        except FileNotFoundError:
            raise RuntimeError(f"CD-HIT not found at {self.cdhit_path}. "
                             "Install with: conda install -c bioconda cd-hit")
    
    def cluster_sequences(self, 
                         sequences: List[str], 
                         sequence_ids: List[str],
                         identity_threshold: float = 0.30,
                         word_length: int = 5,
                         output_dir: str = "data/processed") -> Dict:
        """
        Cluster sequences using CD-HIT.
        
        Args:
            sequences: List of protein sequences
            sequence_ids: List of sequence identifiers
            identity_threshold: CD-HIT identity threshold (0.0-1.0)
            word_length: CD-HIT word length parameter
            output_dir: Output directory for clustering results
            
        Returns:
            Dictionary with clustering results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Clustering {len(sequences)} sequences at {identity_threshold*100:.0f}% identity")
        
        # Create temporary FASTA file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            for seq_id, sequence in zip(sequence_ids, sequences):
                f.write(f">{seq_id}\n{sequence}\n")
            temp_fasta = f.name
        
        # CD-HIT output files
        cdhit_output = output_dir / f"clusters_{identity_threshold:.2f}.fasta"
        cdhit_clstr = output_dir / f"clusters_{identity_threshold:.2f}.clstr"
        
        # Run CD-HIT
        cmd = [
            self.cdhit_path,
            "-i", temp_fasta,
            "-o", str(cdhit_output),
            "-c", str(identity_threshold),
            "-n", str(word_length),
            "-M", "16000",  # Memory limit
            "-d", "0"        # Full description
        ]
        
        logger.info(f"Running CD-HIT: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("CD-HIT clustering completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"CD-HIT failed: {e.stderr}")
            raise RuntimeError(f"CD-HIT clustering failed: {e.stderr}")
        finally:
            # Clean up temp file
            Path(temp_fasta).unlink()
        
        # Parse clustering results
        cluster_map = self._parse_clustering_results(cdhit_clstr, sequence_ids)
        
        # Save clustering results
        clustering_results = {
            'identity_threshold': identity_threshold,
            'word_length': word_length,
            'num_sequences': len(sequences),
            'num_clusters': len(set(cluster_map.values())),
            'cluster_map': cluster_map,
            'cluster_sizes': self._get_cluster_sizes(cluster_map)
        }
        
        with open(output_dir / f"clustering_{identity_threshold:.2f}.json", 'w') as f:
            json.dump(clustering_results, f, indent=2)
        
        logger.info(f"Clustering results saved to {output_dir}")
        logger.info(f"Created {clustering_results['num_clusters']} clusters")
        
        return clustering_results
    
    def _parse_clustering_results(self, 
                                 clstr_file: Path, 
                                 sequence_ids: List[str]) -> Dict[str, int]:
        """Parse CD-HIT clustering output file."""
        cluster_map = {}
        current_cluster = -1
        
        with open(clstr_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                if line.startswith('>Cluster'):
                    current_cluster = int(line.split()[1])
                elif line.startswith('0\t'):
                    # Representative sequence
                    seq_id = line.split('>')[1].split('...')[0]
                    cluster_map[seq_id] = current_cluster
                elif line.startswith('\t'):
                    # Member sequence
                    seq_id = line.split('>')[1].split('...')[0]
                    cluster_map[seq_id] = current_cluster
        
        # Map all sequence IDs to clusters
        final_map = {}
        for seq_id in sequence_ids:
            if seq_id in cluster_map:
                final_map[seq_id] = cluster_map[seq_id]
            else:
                # Handle any unmapped sequences
                final_map[seq_id] = -1
                logger.warning(f"Sequence {seq_id} not mapped to any cluster")
        
        return final_map
    
    def _get_cluster_sizes(self, cluster_map: Dict[str, int]) -> Dict[int, int]:
        """Get size of each cluster."""
        cluster_sizes = {}
        for cluster_id in cluster_map.values():
            cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1
        return cluster_sizes
    
    def create_leakage_aware_splits(self,
                                   cluster_results: Dict,
                                   train_ratio: float = 0.7,
                                   val_ratio: float = 0.15,
                                   test_ratio: float = 0.15,
                                   random_seed: int = 42) -> Dict[str, List[str]]:
        """
        Create leakage-aware splits based on clustering.
        
        Args:
            cluster_results: Results from cluster_sequences
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with train/val/test sequence IDs
        """
        np.random.seed(random_seed)
        
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        cluster_map = cluster_results['cluster_map']
        cluster_sizes = cluster_results['cluster_sizes']
        
        # Group sequences by cluster
        clusters = {}
        for seq_id, cluster_id in cluster_map.items():
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(seq_id)
        
        # Sort clusters by size (largest first)
        sorted_clusters = sorted(clusters.items(), 
                               key=lambda x: len(x[1]), reverse=True)
        
        # Distribute clusters across splits
        train_ids = []
        val_ids = []
        test_ids = []
        
        for cluster_id, seq_ids in sorted_clusters:
            cluster_size = len(seq_ids)
            
            # Calculate target sizes for this cluster
            train_size = int(cluster_size * train_ratio)
            val_size = int(cluster_size * val_ratio)
            test_size = cluster_size - train_size - val_size
            
            # Shuffle sequence IDs within cluster
            np.random.shuffle(seq_ids)
            
            # Assign to splits
            train_ids.extend(seq_ids[:train_size])
            val_ids.extend(seq_ids[train_size:train_size + val_size])
            test_ids.extend(seq_ids[train_size + val_size:])
        
        # Log split statistics
        logger.info(f"Created leakage-aware splits:")
        logger.info(f"  Training: {len(train_ids)} sequences")
        logger.info(f"  Validation: {len(val_ids)} sequences")
        logger.info(f"  Test: {len(test_ids)} sequences")
        
        # Verify no leakage between splits
        train_set = set(train_ids)
        val_set = set(val_ids)
        test_set = set(test_ids)
        
        if train_set & val_set or train_set & test_set or val_set & test_set:
            raise RuntimeError("Sequence leakage detected between splits!")
        
        return {
            'train': train_ids,
            'val': val_ids,
            'test': test_ids,
            'cluster_info': {
                'num_clusters': len(clusters),
                'cluster_distribution': self._analyze_cluster_distribution(
                    clusters, train_ids, val_ids, test_ids
                )
            }
        }
    
    def _analyze_cluster_distribution(self, 
                                    clusters: Dict[int, List[str]],
                                    train_ids: List[str],
                                    val_ids: List[str],
                                    test_ids: List[str]) -> Dict:
        """Analyze how clusters are distributed across splits."""
        train_set = set(train_ids)
        val_set = set(val_ids)
        test_set = set(test_ids)
        
        distribution = {
            'train_only': 0,      # Clusters only in training
            'val_only': 0,        # Clusters only in validation
            'test_only': 0,       # Clusters only in test
            'mixed': 0,           # Clusters split across multiple sets
            'cluster_details': {}
        }
        
        for cluster_id, seq_ids in clusters.items():
            train_count = sum(1 for seq_id in seq_ids if seq_id in train_set)
            val_count = sum(1 for seq_id in seq_ids if seq_id in val_set)
            test_count = sum(1 for seq_id in seq_ids if seq_id in test_set)
            
            if train_count > 0 and val_count == 0 and test_count == 0:
                distribution['train_only'] += 1
            elif val_count > 0 and train_count == 0 and test_count == 0:
                distribution['val_only'] += 1
            elif test_count > 0 and train_count == 0 and val_count == 0:
                distribution['test_only'] += 1
            else:
                distribution['mixed'] += 1
            
            distribution['cluster_details'][cluster_id] = {
                'train': train_count,
                'val': val_count,
                'test': test_count,
                'total': len(seq_ids)
            }
        
        return distribution

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cluster sequences and create leakage-aware splits")
    parser.add_argument("--sequences", required=True, help="Path to sequences CSV/parquet")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    parser.add_argument("--identity", type=float, default=0.30, help="CD-HIT identity threshold")
    parser.add_argument("--word-length", type=int, default=5, help="CD-HIT word length")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training set ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation set ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Load sequences
    if args.sequences.endswith('.parquet'):
        df = pd.read_parquet(args.sequences)
    else:
        df = pd.read_csv(args.sequences)
    
    # Extract sequences and IDs
    sequences = df['sequence'].tolist()
    sequence_ids = df['accession'].tolist() if 'accession' in df.columns else df.index.astype(str).tolist()
    
    # Initialize clusterer
    clusterer = SequenceClusterer()
    
    # Cluster sequences
    cluster_results = clusterer.cluster_sequences(
        sequences=sequences,
        sequence_ids=sequence_ids,
        identity_threshold=args.identity,
        word_length=args.word_length,
        output_dir=args.output_dir
    )
    
    # Create splits
    splits = clusterer.create_leakage_aware_splits(
        cluster_results=cluster_results,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed
    )
    
    # Save splits
    output_dir = Path(args.output_dir)
    
    # Create split files
    for split_name, seq_ids in splits.items():
        if split_name in ['train', 'val', 'test']:
            split_df = df[df['accession'].isin(seq_ids) if 'accession' in df.columns else df.index.astype(str).isin(seq_ids)]
            split_df.to_parquet(output_dir / f"{split_name}_clustered.parquet", index=False)
            logger.info(f"Saved {split_name} split: {len(split_df)} sequences")
    
    # Save split metadata
    with open(output_dir / "split_metadata.json", 'w') as f:
        json.dump(splits, f, indent=2)
    
    logger.info("Leakage-aware splits created successfully!")

if __name__ == "__main__":
    main()
