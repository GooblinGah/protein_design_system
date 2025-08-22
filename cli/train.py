#!/usr/bin/env python3
"""
Training CLI for the Protein Design System.
"""

import argparse
import yaml
import torch
import logging
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.decoder.pointer_generator import PointerGeneratorDecoder
from models.controller.segmental_hmm import SegmentalHMMController
from constraints.fsa import FSAConstraintEngine
from training.loops import TrainingLoop
from training.curriculum import CurriculumManager
from training.monitors import TrainingMonitor
from models.tokenizer import ProteinTokenizer


def main():
    parser = argparse.ArgumentParser(description='Train Protein Design System')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--output_dir', type=str, default='runs/', help='Output directory')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Set device
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    logger.info(f"Using device: {device}")
    
    # Initialize components
    logger.info("Initializing model components...")
    
    # Tokenizer
    tokenizer = ProteinTokenizer()
    
    # Model
    model = PointerGeneratorDecoder(
        vocab_size=config['model']['vocab_size'],
        d_model=config['model']['d_model'],
        n_layers=config['model']['n_layers'],
        n_heads=config['model']['n_heads'],
        dropout=config['model']['dropout']
    ).to(device)
    
    # Controller
    controller = SegmentalHMMController().to(device)
    
    # Constraint engine
    constraint_engine = FSAConstraintEngine()
    
    # Training components
    curriculum = CurriculumManager(config)
    monitor = TrainingMonitor(config)
    
    # Training loop
    training_loop = TrainingLoop(model, controller, constraint_engine, config)
    
    # Optimizer
    param_groups = model.get_parameter_groups()
    optimizer = torch.optim.AdamW([
        {'params': param_groups['decoder'], 'lr': config['optimizer']['lr_decoder']},
        {'params': param_groups['pointer'], 'lr': config['optimizer']['lr_pointer']},
        {'params': param_groups['gate'], 'lr': config['optimizer']['lr_pointer']}
    ], betas=config['optimizer']['betas'], weight_decay=config['optimizer']['weight_decay'])
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['epochs']
    )
    
    # Training loop
    logger.info("Starting training...")
    
    for epoch in range(config['training']['epochs']):
        logger.info(f"Epoch {epoch + 1}/{config['training']['epochs']}")
        
        # Update curriculum
        curriculum.update_epoch(epoch)
        curriculum.log_stage_info()
        
        # Get current training config
        train_config = curriculum.get_training_config()
        
        # Training step (placeholder - would iterate over actual data)
        for step in range(100):  # Placeholder
            # This would be actual training data
            batch = {
                'input_ids': torch.randint(0, 23, (4, 64)),
                'exemplars': torch.randint(0, 23, (4, 8, 128)),
                'column_feats': torch.randn(4, 128, 6),
                'c_t': torch.randint(0, 128, (4, 64)),
                'copy_eligible': torch.randint(0, 2, (4, 64)).bool(),
                'target_ids': torch.randint(0, 23, (4, 64))
            }
            
            # Training step
            losses = training_loop.train_step(batch, optimizer, scheduler)
            
            # Update monitor
            monitor.update_metrics(step, losses)
            
            # Check for alerts
            alerts = monitor.check_alerts()
            if alerts:
                logger.warning(f"Training alerts at step {step}: {alerts}")
        
        # Log epoch summary
        epoch_summary = monitor.get_training_summary()
        logger.info(f"Epoch {epoch + 1} summary: {epoch_summary}")
        
        # Save checkpoint
        checkpoint_path = Path(args.output_dir) / f"checkpoint_epoch_{epoch + 1}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'controller_state_dict': controller.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': config,
            'curriculum_state': curriculum.current_stage,
            'metrics': epoch_summary
        }, checkpoint_path)
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    logger.info("Training completed!")
    
    # Save final metrics
    final_summary = monitor.get_training_summary()
    metrics_path = Path(args.output_dir) / "final_metrics.json"
    monitor.save_metrics(str(metrics_path))


if __name__ == '__main__':
    main()
