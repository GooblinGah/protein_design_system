import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import logging

from models.decoder.pointer_generator import PointerGeneratorDecoder
from models.controller.segmental_hmm import SegmentalHMMController
from constraints.fsa import FSAConstraintEngine


class TrainingLoop:
    """Training loop for the protein design system."""
    
    def __init__(self, 
                 model: PointerGeneratorDecoder,
                 controller: SegmentalHMMController,
                 constraint_engine: FSAConstraintEngine,
                 config: Dict[str, Any]):
        self.model = model
        self.controller = controller
        self.constraint_engine = constraint_engine
        self.config = config
        
        # Loss weights
        self.alpha = config.get('loss_weights', {}).get('gate', 0.5)
        self.beta = config.get('loss_weights', {}).get('copy', 0.5)
        self.gamma = config.get('loss_weights', {}).get('identity', 0.2)
        self.tau = config.get('novelty', {}).get('max_single_identity', 0.70)
        
        # Device
        self.device = next(model.parameters()).device
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train_step(self, 
                  batch: Dict[str, torch.Tensor],
                  optimizer: torch.optim.Optimizer,
                  scheduler: torch.optim.lr_scheduler._LRScheduler = None) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Training batch
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            
        Returns:
            Dictionary of losses and metrics
        """
        self.model.train()
        self.controller.train()
        
        # Unpack batch
        input_ids = batch['input_ids'].to(self.device)
        exemplars = batch['exemplars'].to(self.device)
        column_feats = batch['column_feats'].to(self.device)
        c_t = batch['c_t'].to(self.device)
        copy_eligible = batch['copy_eligible'].to(self.device)
        target_ids = batch['target_ids'].to(self.device)
        
        # Forward pass
        logits_vocab, p_copy, gate_logits, lambda_ik = self.model(
            input_ids, exemplars, column_feats, c_t, None
        )
        
        # Compute losses
        losses = self._compute_losses(
            logits_vocab, p_copy, gate_logits, lambda_ik,
            target_ids, copy_eligible, c_t, exemplars
        )
        
        total_loss = losses['total']
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('grad_clip', 1.0))
        
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        return losses
    
    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single evaluation step.
        
        Args:
            batch: Evaluation batch
            
        Returns:
            Dictionary of losses and metrics
        """
        self.model.eval()
        self.controller.eval()
        
        with torch.no_grad():
            # Unpack batch
            input_ids = batch['input_ids'].to(self.device)
            exemplars = batch['exemplars'].to(self.device)
            column_feats = batch['column_feats'].to(self.device)
            c_t = batch['c_t'].to(self.device)
            copy_eligible = batch['copy_eligible'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            
            # Forward pass
            logits_vocab, p_copy, gate_logits, lambda_ik = self.model(
                input_ids, exemplars, column_feats, c_t, None
            )
            
            # Compute losses
            losses = self._compute_losses(
                logits_vocab, p_copy, gate_logits, lambda_ik,
                target_ids, copy_eligible, c_t, exemplars
            )
            
            # Compute additional metrics
            metrics = self._compute_metrics(
                logits_vocab, p_copy, gate_logits, lambda_ik,
                target_ids, copy_eligible, c_t, exemplars
            )
            
            losses.update(metrics)
            
        return losses
    
    def _compute_losses(self,
                        logits_vocab: torch.Tensor,
                        p_copy: torch.Tensor,
                        gate_logits: torch.Tensor,
                        lambda_ik: torch.Tensor,
                        target_ids: torch.Tensor,
                        copy_eligible: torch.Tensor,
                        c_t: torch.Tensor,
                        exemplars: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute all training losses."""
        
        # Cross-entropy loss
        p_mixture = self.model.compute_mixture_distribution(logits_vocab, p_copy, gate_logits)
        ce_loss = F.cross_entropy(
            p_mixture.view(-1, p_mixture.size(-1)),
            target_ids.view(-1),
            ignore_index=0  # PAD token
        )
        
        # Gate loss
        gate_targets = (~copy_eligible).float()
        gate_probs = torch.sigmoid(gate_logits)
        gate_loss = F.binary_cross_entropy(gate_probs, gate_targets, reduction='none')
        gate_loss = gate_loss[target_ids != 0].mean()  # Ignore padding
        
        # Copy loss
        copy_loss = self._compute_copy_loss(lambda_ik, exemplars, c_t, target_ids, copy_eligible)
        
        # Identity regularizer
        identity_reg = self._compute_identity_regularizer(lambda_ik, exemplars, c_t, target_ids)
        
        # Total loss
        total_loss = (ce_loss + 
                     self.alpha * gate_loss + 
                     self.beta * copy_loss + 
                     self.gamma * identity_reg)
        
        return {
            'total': total_loss,
            'ce': ce_loss,
            'gate': gate_loss,
            'copy': copy_loss,
            'identity': identity_reg
        }
    
    def _compute_copy_loss(self,
                           lambda_ik: torch.Tensor,
                           exemplars: torch.Tensor,
                           c_t: torch.Tensor,
                           target_ids: torch.Tensor,
                           copy_eligible: torch.Tensor) -> torch.Tensor:
        """Compute copy loss when target matches exemplar."""
        
        batch_size, seq_len, num_exemplars = lambda_ik.shape
        copy_loss = torch.tensor(0.0, device=self.device)
        num_valid = 0
        
        for t in range(seq_len):
            eligible_positions = copy_eligible[:, t]
            if not eligible_positions.any():
                continue
            
            c_t_current = c_t[eligible_positions, t]
            targets = target_ids[eligible_positions, t]
            
            for b_idx, valid_b in enumerate(eligible_positions.nonzero().squeeze(-1)):
                c_col = c_t_current[b_idx]
                target_aa = targets[b_idx]
                
                # Find matching exemplars
                matching = (exemplars[valid_b, :, c_col] == target_aa)
                if matching.any():
                    matching_weight = lambda_ik[valid_b, t, matching].sum()
                    if matching_weight > 0:
                        copy_loss -= torch.log(matching_weight + 1e-8)
                        num_valid += 1
        
        if num_valid > 0:
            copy_loss = copy_loss / num_valid
        
        return copy_loss
    
    def _compute_identity_regularizer(self,
                                     lambda_ik: torch.Tensor,
                                     exemplars: torch.Tensor,
                                     c_t: torch.Tensor,
                                     target_ids: torch.Tensor) -> torch.Tensor:
        """Compute identity regularizer to maintain novelty."""
        
        batch_size, seq_len, num_exemplars = lambda_ik.shape
        
        # Compute per-exemplar identity
        identity_scores = torch.zeros(batch_size, num_exemplars, device=self.device)
        
        for t in range(seq_len):
            valid_positions = (c_t[:, t] >= 0)
            if not valid_positions.any():
                continue
            
            c_t_current = c_t[valid_positions, t]
            targets = target_ids[valid_positions, t]
            
            for b_idx, valid_b in enumerate(valid_positions.nonzero().squeeze(-1)):
                c_col = c_t_current[b_idx]
                target_aa = targets[b_idx]
                
                # Check if target matches exemplar
                exemplar_aas = exemplars[valid_b, :, c_col]
                matches = (exemplar_aas == target_aa)
                
                # Add to identity scores
                identity_scores[valid_b] += lambda_ik[valid_b, t, :] * matches.float()
        
        # Normalize by sequence length
        seq_lengths = (c_t >= 0).sum(dim=1, keepdim=True).float()
        identity_scores = identity_scores / (seq_lengths + 1e-8)
        
        # Apply soft penalty when exceeding threshold
        penalty = torch.clamp(identity_scores - self.tau, min=0)
        identity_reg = (penalty ** 2).mean()
        
        return identity_reg
    
    def _compute_metrics(self,
                         logits_vocab: torch.Tensor,
                         p_copy: torch.Tensor,
                         gate_logits: torch.Tensor,
                         lambda_ik: torch.Tensor,
                         target_ids: torch.Tensor,
                         copy_eligible: torch.Tensor,
                         c_t: torch.Tensor,
                         exemplars: torch.Tensor) -> Dict[str, float]:
        """Compute additional training metrics."""
        
        # Copy rate in conserved columns
        copy_rate = 0.0
        if copy_eligible.any():
            copy_decisions = (torch.sigmoid(gate_logits) < 0.5)
            copy_rate = (copy_decisions & copy_eligible).float().mean().item()
        
        # Gate entropy in motif windows
        gate_entropy = 0.0
        if gate_logits.numel() > 0:
            gate_probs = torch.sigmoid(gate_logits)
            # Compute binary entropy
            gate_entropy = -(gate_probs * torch.log(gate_probs + 1e-8) + 
                           (1 - gate_probs) * torch.log(1 - gate_probs + 1e-8))
            gate_entropy = gate_entropy.mean().item()
        
        # Per-exemplar identity (max)
        max_identity = 0.0
        if lambda_ik.numel() > 0:
            # This is a simplified version - would need proper computation
            max_identity = 0.5  # Placeholder
        
        return {
            'copy_rate': copy_rate,
            'gate_entropy': gate_entropy,
            'max_identity': max_identity
        }
