import torch
from typing import Dict, Any, List
import logging


class CurriculumManager:
    """Manages curriculum-based training with staged losses."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.current_epoch = 0
        
        # Curriculum stages
        self.stages = config.get('curriculum', {}).get('stages', [])
        self.current_stage = 0
        
        # Loss weights
        self.loss_weights = config.get('loss_weights', {})
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initialized curriculum with {len(self.stages)} stages")
    
    def get_current_stage(self) -> Dict[str, Any]:
        """Get current curriculum stage."""
        if self.current_stage < len(self.stages):
            return self.stages[self.current_stage]
        return self.stages[-1] if self.stages else {}
    
    def should_advance_stage(self) -> bool:
        """Check if we should advance to next stage."""
        current_stage = self.get_current_stage()
        epoch_end = current_stage.get('epoch_end', float('inf'))
        return self.current_epoch >= epoch_end
    
    def advance_stage(self):
        """Advance to next curriculum stage."""
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            self.logger.info(f"Advanced to curriculum stage {self.current_stage}")
        else:
            self.logger.info("Reached final curriculum stage")
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Get current loss weights based on curriculum stage."""
        weights = self.loss_weights.copy()
        
        # Stage 0: CE only, gate fixed
        if self.current_stage == 0:
            weights['gate'] = 0.0
            weights['copy'] = 0.0
            weights['identity'] = 0.0
        
        # Stage 1: Add gate loss
        elif self.current_stage == 1:
            weights['gate'] = self.loss_weights.get('gate', 0.5)
            weights['copy'] = 0.0
            weights['identity'] = 0.0
        
        # Stage 2: Add copy loss, enable controller
        elif self.current_stage == 2:
            weights['gate'] = self.loss_weights.get('gate', 0.5)
            weights['copy'] = self.loss_weights.get('copy', 0.5)
            weights['identity'] = 0.0
        
        # Stage 3: Add identity regularizer
        elif self.current_stage >= 3:
            weights['gate'] = self.loss_weights.get('gate', 0.5)
            weights['copy'] = self.loss_weights.get('copy', 0.5)
            weights['identity'] = self.loss_weights.get('identity', 0.2)
        
        return weights
    
    def get_controller_settings(self) -> Dict[str, Any]:
        """Get controller settings based on curriculum stage."""
        settings = {}
        
        # Stage 0-1: No controller
        if self.current_stage < 2:
            settings['enabled'] = False
            settings['z_soft'] = float('inf')
            settings['z_hard'] = float('inf')
        
        # Stage 2+: Enable controller
        else:
            settings['enabled'] = True
            settings['z_soft'] = 0.7
            settings['z_hard'] = 1.5
        
        return settings
    
    def get_copy_forcing_rate(self) -> float:
        """Get copy forcing rate based on curriculum stage."""
        if self.current_stage < 2:
            return 0.0  # No copy forcing in early stages
        
        # Start with light copy forcing, increase if needed
        base_rate = 0.05
        stage_bonus = (self.current_stage - 2) * 0.05
        return min(base_rate + stage_bonus, 0.20)  # Cap at 20%
    
    def update_epoch(self, epoch: int):
        """Update current epoch and check for stage advancement."""
        self.current_epoch = epoch
        
        if self.should_advance_stage():
            self.advance_stage()
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get complete training configuration for current stage."""
        return {
            'loss_weights': self.get_loss_weights(),
            'controller_settings': self.get_controller_settings(),
            'copy_forcing_rate': self.get_copy_forcing_rate(),
            'current_stage': self.current_stage,
            'current_epoch': self.current_epoch
        }
    
    def log_stage_info(self):
        """Log current curriculum stage information."""
        config = self.get_training_config()
        
        self.logger.info(f"Curriculum Stage {self.current_stage}")
        self.logger.info(f"  Loss weights: {config['loss_weights']}")
        self.logger.info(f"  Controller enabled: {config['controller_settings']['enabled']}")
        self.logger.info(f"  Copy forcing rate: {config['copy_forcing_rate']}")
        self.logger.info(f"  Epoch: {self.current_epoch}")


class DynamicLossAdjuster:
    """Dynamically adjusts loss weights based on training metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_weights = config.get('loss_weights', {})
        self.current_weights = self.base_weights.copy()
        
        # Adjustment thresholds
        self.copy_rate_target = 0.5
        self.copy_rate_threshold = 0.4
        self.identity_threshold = 0.68
        
        # Adjustment factors
        self.adjustment_factor = 1.2
        self.max_adjustment = 2.0
        
        self.logger = logging.getLogger(__name__)
    
    def adjust_weights(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Adjust loss weights based on current metrics."""
        
        # Adjust copy weight based on copy rate
        copy_rate = metrics.get('copy_rate', 0.0)
        if copy_rate < self.copy_rate_threshold:
            # Increase copy weight
            self.current_weights['copy'] = min(
                self.current_weights['copy'] * self.adjustment_factor,
                self.base_weights['copy'] * self.max_adjustment
            )
            self.logger.info(f"Adjusted copy weight to {self.current_weights['copy']:.3f}")
        
        # Adjust identity weight based on max identity
        max_identity = metrics.get('max_identity', 0.0)
        if max_identity > self.identity_threshold:
            # Increase identity weight
            self.current_weights['identity'] = min(
                self.current_weights['identity'] * self.adjustment_factor,
                self.base_weights['identity'] * self.max_adjustment
            )
            self.logger.info(f"Adjusted identity weight to {self.current_weights['copy']:.3f}")
        
        return self.current_weights
    
    def reset_weights(self):
        """Reset weights to base values."""
        self.current_weights = self.base_weights.copy()
        self.logger.info("Reset loss weights to base values")
