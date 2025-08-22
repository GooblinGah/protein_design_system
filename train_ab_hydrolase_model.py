#!/usr/bin/env python3
"""
Train protein design model on Alpha/Beta Hydrolase dataset
"""

import argparse
import logging
from pathlib import Path
import torch
import yaml
import re

# Import your existing modules
from models.decoder.pointer_generator import PointerGeneratorDecoder
from models.tokenizer import ProteinTokenizer
from training.trainer import ProteinDesignTrainer
from data import ProteinDesignDataset, ExemplarRetriever
from evaluation.metrics import ProteinDesignEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train AB Hydrolase Model")
    parser.add_argument("--config", default="config_ab_hydrolase.yaml")
    parser.add_argument("--output-dir", default="runs/ab_hydrolase_v1")
    parser.add_argument("--resume", help="Resume from checkpoint")
    parser.add_argument("--gpu", default=0, type=int)
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize components
    logger.info("Initializing model components...")
    
    tokenizer = ProteinTokenizer()
    
    model = PointerGeneratorDecoder(
        vocab_size=tokenizer.vocab_size,
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        n_layers=config['model']['n_layers'],
        d_ff=config['model']['d_ff'],
        dropout=config['model']['dropout'],
        max_seq_length=config['model']['max_length']
    ).to(device)
    
    # Load retriever
    retriever = ExemplarRetriever(
        embedding_dim=config['retrieval']['embedding_dim']
    )
    
    if Path(config['retrieval']['index_path']).exists():
        retriever.load_index(config['retrieval']['index_path'])
        logger.info("Loaded retrieval index")
    else:
        logger.warning("Retrieval index not found - training without exemplars")
        retriever = None
    
    # Setup datasets
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
    
    # Initialize trainer
    trainer = ProteinDesignTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        device=device
    )
    
    # Add AB hydrolase specific callbacks
    trainer.add_callback('motif_monitor', ABHydrolaseMotifMonitor())
    
    # Resume if checkpoint provided
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Training complete!")

class ABHydrolaseMotifMonitor:
    """Monitor AB hydrolase specific motifs during training"""
    
    def __init__(self):
        self.gxsxg_pattern = re.compile(r'G[A-Z]S[A-Z]G')
        
    def on_validation_step(self, trainer, outputs):
        """Check motif presence in generated sequences"""
        
        generated_seqs = outputs.get('generated_sequences', [])
        
        gxsxg_count = sum(1 for seq in generated_seqs if self.gxsxg_pattern.search(seq))
        
        metrics = {
            'gxsxg_presence_rate': gxsxg_count / len(generated_seqs) if generated_seqs else 0
        }
        
        return metrics

if __name__ == "__main__":
    main()
