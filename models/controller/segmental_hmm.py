import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SegmentalHMMController(nn.Module):
    """
    Segmental-HMM controller for pacing between motif anchors.
    
    Predicts duration statistics for inter-motif segments and controls
    generation pacing through tier-based policies.
    """
    
    def __init__(self, feature_dim: int = 13, hidden_dim: int = 64):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # MLP for predicting duration statistics
        self.duration_predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2)  # mu and log_sigma
        )
        
        # Hysteresis state
        self.hysteresis_count = 0
        self.current_tier = "normal"
        
        # Tier thresholds
        self.z_soft = 0.7
        self.z_hard = 1.5
        self.hysteresis_min_tokens = 3
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self, segment_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to predict duration statistics.
        
        Args:
            segment_features: [B, feature_dim] features for current segment
            
        Returns:
            mu: [B] predicted mean duration
            sigma: [B] predicted standard deviation
        """
        # Predict duration statistics
        duration_params = self.duration_predictor(segment_features)
        mu = duration_params[:, 0]
        log_sigma = duration_params[:, 1]
        
        # Ensure positive sigma with softplus
        sigma = F.softplus(log_sigma) + 1e-3
        
        return mu, sigma
    
    def compute_tier(self, 
                    emitted_j: int,
                    mu_j: float,
                    sigma_j: float) -> Tuple[str, float, float]:
        """
        Compute tier and gate bias based on current pacing.
        
        Args:
            emitted_j: Number of tokens emitted since segment start
            mu_j: Predicted mean duration for this segment
            sigma_j: Predicted std duration for this segment
            
        Returns:
            tier: "normal", "stretched", or "sparse"
            gate_bias: Bias to apply to gate head
            advance_factor: Factor to multiply column advance by
        """
        # Compute z-score
        z = (emitted_j - mu_j) / sigma_j
        
        # Determine tier based on z-score
        if z <= self.z_soft:
            tier = "normal"
            gate_bias = 0.0
            advance_factor = 1.0
        elif z <= self.z_hard:
            tier = "stretched"
            gate_bias = -0.4
            advance_factor = 0.8
        else:
            tier = "sparse"
            gate_bias = 1.0  # Force generation
            advance_factor = 0.0  # Hold column index
        
        # Apply hysteresis
        if tier != self.current_tier:
            if self.hysteresis_count < self.hysteresis_min_tokens:
                # Keep current tier
                tier = self.current_tier
                if tier == "normal":
                    gate_bias = 0.0
                    advance_factor = 1.0
                elif tier == "stretched":
                    gate_bias = -0.4
                    advance_factor = 0.8
                else:  # sparse
                    gate_bias = 1.0
                    advance_factor = 0.0
            else:
                # Switch tier
                self.current_tier = tier
                self.hysteresis_count = 0
        else:
            self.hysteresis_count += 1
        
        return tier, gate_bias, advance_factor
    
    def get_provenance_tier(self, tier: str) -> str:
        """
        Get provenance confidence tier.
        
        Args:
            tier: Current pacing tier
            
        Returns:
            Provenance tier: "normal", "stretched", or "sparse"
        """
        if tier == "sparse":
            return "sparse"
        elif tier == "stretched":
            return "stretched"
        else:
            return "normal"
    
    def create_segment_features(self,
                              exemplar_lengths: torch.Tensor,  # [B, K]
                              consensus_features: torch.Tensor,  # [B, Lc, F]
                              dsl_length_target: torch.Tensor,  # [B]
                              secondary_structure: Optional[torch.Tensor] = None,  # [B, Lc, 3]
                              hydropathy: Optional[torch.Tensor] = None  # [B, Lc]
                              ) -> torch.Tensor:
        """
        Create feature vector for segment duration prediction.
        
        Args:
            exemplar_lengths: [B, K] lengths of K exemplars
            consensus_features: [B, Lc, F] consensus column features
            dsl_length_target: [B] target length for this segment
            secondary_structure: [B, Lc, 3] helix/sheet/coil fractions
            hydropathy: [B, Lc] hydropathy scores
            
        Returns:
            [B, 13] feature vector for duration prediction
        """
        batch_size = exemplar_lengths.shape[0]
        
        # Exemplar length statistics
        len_mean = torch.mean(exemplar_lengths.float(), dim=1)  # [B]
        len_std = torch.std(exemplar_lengths.float(), dim=1)   # [B]
        len_min = torch.min(exemplar_lengths.float(), dim=1)[0]  # [B]
        len_max = torch.max(exemplar_lengths.float(), dim=1)[0]  # [B]
        
        # Quartiles
        len_q25 = torch.quantile(exemplar_lengths.float(), 0.25, dim=1)  # [B]
        len_q75 = torch.quantile(exemplar_lengths.float(), 0.75, dim=1)  # [B]
        
        # Gap fraction (placeholder - would need actual gap info)
        gap_frac = torch.zeros(batch_size, device=self.device)
        
        # Consensus features (mean across columns)
        cons_mean = torch.mean(consensus_features, dim=1)  # [B, F]
        hmm_match_mean = torch.mean(consensus_features[:, :, 2], dim=1)  # [B] assuming index 2 is HMM match
        
        # Secondary structure fractions
        if secondary_structure is not None:
            secstruct_helix = torch.mean(secondary_structure[:, :, 0], dim=1)  # [B]
            secstruct_sheet = torch.mean(secondary_structure[:, :, 1], dim=1)  # [B]
        else:
            secstruct_helix = torch.zeros(batch_size, device=self.device)
            secstruct_sheet = torch.zeros(batch_size, device=self.device)
        
        # Hydropathy mean
        if hydropathy is not None:
            hydropathy_mean = torch.mean(hydropathy, dim=1)  # [B]
        else:
            hydropathy_mean = torch.zeros(batch_size, device=self.device)
        
        # Combine all features
        features = torch.stack([
            len_mean, len_std, len_min, len_max, len_q25, len_q75, gap_frac,
            cons_mean.mean(dim=1), hmm_match_mean, dsl_length_target,
            secstruct_helix, secstruct_sheet, hydropathy_mean
        ], dim=1)  # [B, 13]
        
        return features
    
    def reset_state(self):
        """Reset controller state for new sequence."""
        self.hysteresis_count = 0
        self.current_tier = "normal"
