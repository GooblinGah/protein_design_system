#!/usr/bin/env python3
"""
Main script to prepare training data from raw files
"""

import argparse
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.preprocessing import ProteinDatasetBuilder
from data.retrieval import ExemplarRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Prepare protein design training data")
    parser.add_argument("--fasta", required=True, help="Path to FASTA file")
    parser.add_argument("--annotations", required=True, help="Path to annotations CSV")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    parser.add_argument("--build-retrieval", action="store_true", help="Build retrieval index")
    
    args = parser.parse_args()
    
    # Build dataset
    logger.info("Building dataset...")
    builder = ProteinDatasetBuilder()
    
    paths = builder.build_dataset(
        fasta_file=args.fasta,
        annotation_file=args.annotations,
        output_dir=args.output_dir
    )
    
    logger.info(f"Dataset created: {paths}")
    
    # Build retrieval index if requested
    if args.build_retrieval:
        logger.info("Building retrieval index...")
        
        # Load all training sequences
        import pandas as pd
        train_data = pd.read_parquet(paths['train'])
        sequences = train_data['sequence'].unique().tolist()
        
        # Build index
        retriever = ExemplarRetriever()
        retriever.build_index(
            sequences,
            save_path=Path(args.output_dir) / "retrieval_index"
        )
        
        logger.info("Retrieval index built and saved")
    
    logger.info("Data preparation complete!")

if __name__ == "__main__":
    main()
