import torch
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from collections import defaultdict, deque
import time


class TrainingMonitor:
    """Monitors training progress and metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Metric tracking
        self.metrics = defaultdict(list)
        self.running_metrics = defaultdict(deque)
        self.window_size = 100  # For running averages
        
        # Thresholds and targets
        self.copy_rate_target = 0.5
        self.gate_entropy_threshold = 0.5
        self.identity_warning = 0.65
        self.identity_clamp = 0.70
        self.duration_error_threshold = 0.3
        self.constraint_pass_rate_target = 0.99
        
        # Auto-nudge settings
        self.copy_rate_threshold = 0.4
        self.identity_soft_estimate_threshold = 0.68
        
        # History for trend analysis
        self.history = defaultdict(list)
        
        # Timing
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.log_interval = 100  # Log every N steps
    
    def update_metrics(self, step: int, metrics: Dict[str, float]):
        """Update metrics for current step."""
        current_time = time.time()
        
        # Store all metrics
        for key, value in metrics.items():
            self.metrics[key].append(value)
            self.running_metrics[key].append(value)
            
            # Keep only recent values for running averages
            if len(self.running_metrics[key]) > self.window_size:
                self.running_metrics[key].popleft()
        
        # Store step and timing info
        self.metrics['step'].append(step)
        self.metrics['time'].append(current_time - self.start_time)
        
        # Log periodically
        if step % self.log_interval == 0:
            self._log_metrics(step)
            self.last_log_time = current_time
    
    def _log_metrics(self, step: int):
        """Log current metrics."""
        current_metrics = self._get_current_metrics()
        
        self.logger.info(f"Step {step} Metrics:")
        self.logger.info(f"  Copy rate: {current_metrics['copy_rate']:.3f} (target: {self.copy_rate_target})")
        self.logger.info(f"  Gate entropy: {current_metrics['gate_entropy']:.3f} (max: {self.gate_entropy_threshold})")
        self.logger.info(f"  Max identity: {current_metrics['max_identity']:.3f} (warning: {self.identity_warning})")
        self.logger.info(f"  Duration error: {current_metrics['duration_error']:.3f} (target: {self.duration_error_threshold})")
        self.logger.info(f"  Constraint pass rate: {current_metrics['constraint_pass_rate']:.3f} (target: {self.constraint_pass_rate_target})")
    
    def _get_current_metrics(self) -> Dict[str, float]:
        """Get current metric values."""
        current = {}
        
        for key in ['copy_rate', 'gate_entropy', 'max_identity', 'duration_error', 'constraint_pass_rate']:
            if self.running_metrics[key]:
                current[key] = np.mean(list(self.running_metrics[key]))
            else:
                current[key] = 0.0
        
        return current
    
    def check_alerts(self) -> Dict[str, Any]:
        """Check for training alerts and recommendations."""
        current_metrics = self._get_current_metrics()
        alerts = {}
        
        # Copy rate alerts
        if current_metrics['copy_rate'] < self.copy_rate_threshold:
            alerts['copy_rate'] = {
                'level': 'warning',
                'message': f"Copy rate {current_metrics['copy_rate']:.3f} below threshold {self.copy_rate_threshold}",
                'recommendation': 'Increase copy weight or enable copy forcing'
            }
        
        # Gate entropy alerts
        if current_metrics['gate_entropy'] > self.gate_entropy_threshold:
            alerts['gate_entropy'] = {
                'level': 'warning',
                'message': f"Gate entropy {current_metrics['gate_entropy']:.3f} above threshold {self.gate_entropy_threshold}",
                'recommendation': 'Check gate head training or add regularization'
            }
        
        # Identity alerts
        if current_metrics['max_identity'] > self.identity_warning:
            level = 'error' if current_metrics['max_identity'] > self.identity_clamp else 'warning'
            alerts['identity'] = {
                'level': level,
                'message': f"Max identity {current_metrics['max_identity']:.3f} {'exceeds' if level == 'error' else 'approaching'} threshold {self.identity_clamp}",
                'recommendation': 'Increase identity regularizer weight or resample'
            }
        
        # Duration error alerts
        if current_metrics['duration_error'] > self.duration_error_threshold:
            alerts['duration_error'] = {
                'level': 'warning',
                'message': f"Duration error {current_metrics['duration_error']:.3f} above threshold {self.duration_error_threshold}",
                'recommendation': 'Check controller training or adjust thresholds'
            }
        
        # Constraint pass rate alerts
        if current_metrics['constraint_pass_rate'] < self.constraint_pass_rate_target:
            alerts['constraint_pass_rate'] = {
                'level': 'error',
                'message': f"Constraint pass rate {current_metrics['constraint_pass_rate']:.3f} below target {self.constraint_pass_rate_target}",
                'recommendation': 'Check FSA constraints or model training'
            }
        
        return alerts
    
    def get_auto_nudge_recommendations(self) -> Dict[str, Any]:
        """Get automatic nudge recommendations for training."""
        current_metrics = self._get_current_metrics()
        nudges = {}
        
        # Copy rate nudges
        if current_metrics['copy_rate'] < self.copy_rate_threshold:
            nudges['copy_weight'] = {
                'action': 'increase',
                'factor': 1.2,
                'reason': f"Copy rate {current_metrics['copy_rate']:.3f} below threshold"
            }
            
            nudges['copy_forcing'] = {
                'action': 'enable',
                'rate': 0.05,
                'reason': 'Low copy rate detected'
            }
        
        # Identity nudges
        if current_metrics['max_identity'] > self.identity_soft_estimate_threshold:
            nudges['identity_weight'] = {
                'action': 'increase',
                'factor': 1.5,
                'reason': f"Identity {current_metrics['max_identity']:.3f} approaching limit"
            }
        
        return nudges
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        if not self.metrics:
            return {}
        
        summary = {}
        
        # Final metrics
        for key in ['copy_rate', 'gate_entropy', 'max_identity', 'duration_error', 'constraint_pass_rate']:
            if self.metrics[key]:
                summary[f'final_{key}'] = self.metrics[key][-1]
                summary[f'mean_{key}'] = np.mean(self.metrics[key])
                summary[f'std_{key}'] = np.std(self.metrics[key])
        
        # Training duration
        if self.metrics['time']:
            summary['total_training_time'] = self.metrics['time'][-1]
            summary['total_steps'] = len(self.metrics['step'])
        
        # Trends
        summary['trends'] = self._analyze_trends()
        
        return summary
    
    def _analyze_trends(self) -> Dict[str, str]:
        """Analyze metric trends over time."""
        trends = {}
        
        for key in ['copy_rate', 'gate_entropy', 'max_identity']:
            if len(self.metrics[key]) >= 10:
                recent = self.metrics[key][-10:]
                early = self.metrics[key][:10]
                
                recent_mean = np.mean(recent)
                early_mean = np.mean(early)
                
                if recent_mean > early_mean * 1.1:
                    trends[key] = 'improving'
                elif recent_mean < early_mean * 0.9:
                    trends[key] = 'degrading'
                else:
                    trends[key] = 'stable'
        
        return trends
    
    def save_metrics(self, filepath: str):
        """Save metrics to file."""
        import json
        
        summary = self.get_training_summary()
        
        with open(filepath, 'filepath', 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Metrics saved to {filepath}")
    
    def reset(self):
        """Reset monitor state."""
        self.metrics.clear()
        self.running_metrics.clear()
        self.history.clear()
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.logger.info("Training monitor reset")
