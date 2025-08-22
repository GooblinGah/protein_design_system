#!/usr/bin/env python3
"""
Master script to train the complete protein design system
"""

import argparse
import logging
import yaml
from pathlib import Path
import torch
import sys

# Add to path
sys.path.append(str(Path(__file__).parent))

from models.decoder.pointer_generator import PointerGeneratorDecoder
from models.tokenizer import ProteinTokenizer
from training.trainer import ProteinDesignTrainer
from data import ProteinDesignDataset, ExemplarRetriever
from evaluation.metrics import ProteinDesignEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Config file")
    parser.add_argument("--prepare-data", action="store_true", help="Prepare data first")
    parser.add_argument("--output-dir", default="runs/experiment_1", help="Output directory")
    parser.add_argument("--resume", help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Step 1: Prepare data if requested
    if args.prepare_data:
        logger.info("Preparing data...")
        from data.preprocessing import ProteinDatasetBuilder
        
        builder = ProteinDatasetBuilder(
            min_length=config['data']['min_length'],
            max_length=config['data']['max_length']
        )
        
        paths = builder.build_dataset(
            fasta_file=config['data']['raw_fasta'],
            annotation_file=config['data']['annotations'],
            output_dir=Path(config['data']['train_path']).parent
        )
        
        # Update config with generated paths
        config['data']['train_path'] = paths['train']
        config['data']['val_path'] = paths['val']
        config['data']['test_path'] = paths['test']
        
        # Build retrieval index
        if config['retrieval']['use_retrieval']:
            logger.info("Building retrieval index...")
            retriever = ExemplarRetriever(
                embedding_dim=config['retrieval']['embedding_dim']
            )
            
            import pandas as pd
            train_data = pd.read_parquet(paths['train'])
            sequences = train_data['sequence'].unique().tolist()
            
            retriever.build_index(
                sequences,
                save_path=config['retrieval']['index_path']
            )
    
    # Step 2: Initialize model
    logger.info("Initializing model...")
    
    tokenizer = ProteinTokenizer()
    
    model = PointerGeneratorDecoder(
        vocab_size=tokenizer.vocab_size,
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        n_layers=config['model']['n_layers'],
        d_ff=config['model'].get('d_ff', 2048),
        dropout=config['model'].get('dropout', 0.1),
        max_seq_length=config['data']['max_length']
    )
    
    # Load checkpoint if resuming
    if args.resume:
        logger.info(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Step 3: Setup retriever
    retriever = None
    if config['retrieval']['use_retrieval']:
        logger.info("Loading retrieval index...")
        retriever = ExemplarRetriever(
            embedding_dim=config['retrieval']['embedding_dim']
        )
        retriever.load_index(config['retrieval']['index_path'])
    
    # Step 4: Setup datasets
    logger.info("Setting up datasets...")
    
    train_dataset = ProteinDesignDataset(
        data_path=config['data']['train_path'],
        tokenizer=tokenizer,
        retriever=retriever,
        max_length=config['data']['max_length']
    )
    
    val_dataset = ProteinDesignDataset(
        data_path=config['data']['val_path'],
        tokenizer=tokenizer,
        retriever=retriever,
        max_length=config['data']['max_length']
    )
    
    # Step 5: Initialize trainer
    logger.info("Initializing trainer...")
    
    trainer = ProteinDesignTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir
    )
    
    # Step 6: Train!
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Training complete!")
    
    # Step 7: Final evaluation
    logger.info("Running final evaluation...")
    
    evaluator = ProteinDesignEvaluator()
    
    # Load test dataset
    test_dataset = ProteinDesignDataset(
        data_path=config['data']['test_path'],
        tokenizer=tokenizer,
        retriever=retriever
    )
    
    # Evaluate
    test_metrics = trainer.evaluate_dataset(test_dataset, evaluator)
    
    logger.info(f"Test metrics: {test_metrics}")
    
    # Save final model
    final_path = Path(args.output_dir) / "final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'metrics': test_metrics
    }, final_path)
    
    logger.info(f"Final model saved to {final_path}")

if __name__ == "__main__":
    main()
