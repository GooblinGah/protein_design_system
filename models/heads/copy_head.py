import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CopyHead(nn.Module):
    """
    Copy head for pointer-generator mechanism.
    
    Computes per-exemplar weights and maps them to sparse AA distributions.
    """
    
    def __init__(self, hidden_dim: int, exemplar_dim: int, feature_dim: int):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.exemplar_dim = exemplar_dim
        self.feature_dim = feature_dim
        
        # MLP for computing exemplar scores
        self.exemplar_scorer = nn.Sequential(
            nn.Linear(hidden_dim + exemplar_dim + feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self,
                hidden_states: torch.Tensor,      # [B, T, H]
                exemplar_embeddings: torch.Tensor, # [B, K, Lc, E]
                column_features: torch.Tensor,     # [B, Lc, F]
                c_t: torch.Tensor,                # [B, T]
                exemplar_aa_ids: torch.Tensor     # [B, K, Lc]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of copy head.
        
        Args:
            hidden_states: Decoder hidden states
            exemplar_embeddings: Exemplar sequence embeddings
            column_features: Consensus column features
            c_t: Column mapping for each position
            exemplar_aa_ids: Exemplar amino acid IDs
            
        Returns:
            p_copy: [B, T, V] sparse copy distribution
            lambda_ik: [B, T, K] per-exemplar weights
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        num_exemplars = exemplar_embeddings.shape[1]
        vocab_size = 23  # 20 AA + BOS/EOS/PAD
        
        # Initialize outputs
        p_copy = torch.zeros(batch_size, seq_len, vocab_size, device=self.device)
        lambda_ik = torch.zeros(batch_size, seq_len, num_exemplars, device=self.device)
        
        # Process each position
        for t in range(seq_len):
            # Get current hidden state and column
            h_t = hidden_states[:, t, :]  # [B, H]
            c_t_current = c_t[:, t]      # [B]
            
            # Skip padding positions
            valid_positions = (c_t_current >= 0)
            if not valid_positions.any():
                continue
            
            # Get exemplar embeddings and features for current column
            # exemplar_embeddings is [B, K, Lc, E], we need to get embeddings for current column c_t_current
            exemplar_embs = exemplar_embeddings[valid_positions, :, c_t_current[valid_positions], :]  # [valid_B, K, E]
            col_feats = column_features[valid_positions, c_t_current[valid_positions]]  # [valid_B, F]
            
            # Expand hidden state for each exemplar
            h_t_expanded = h_t[valid_positions].unsqueeze(1).expand(-1, num_exemplars, -1)  # [valid_B, K, H]
            
            # Concatenate features for scoring
            features = torch.cat([h_t_expanded, exemplar_embs, col_feats.unsqueeze(1).expand(-1, num_exemplars, -1)], dim=-1)
            
            # Compute exemplar scores
            scores = self.exemplar_scorer(features).squeeze(-1)  # [valid_B, K]
            
            # Apply softmax to get weights
            weights = F.softmax(scores, dim=-1)  # [valid_B, K]
            
            # Store weights for provenance
            lambda_ik[valid_positions, t, :] = weights
            
            # Build sparse copy distribution
            for b_idx, valid_b in enumerate(valid_positions.nonzero().squeeze(-1)):
                for k in range(num_exemplars):
                    # Get AA at current column for this exemplar
                    aa_id = exemplar_aa_ids[valid_b, k, c_t_current[valid_b]]
                    
                    # Add weight to copy distribution
                    p_copy[valid_b, t, aa_id] += weights[b_idx, k]
        
        return p_copy, lambda_ik
    
    def compute_copy_loss(self,
                         lambda_ik: torch.Tensor,      # [B, T, K]
                         exemplar_aa_ids: torch.Tensor, # [B, K, Lc]
                         c_t: torch.Tensor,            # [B, T]
                         target_aa_ids: torch.Tensor,  # [B, T]
                         copy_eligible: torch.Tensor   # [B, T]
                         ) -> torch.Tensor:
        """
        Compute copy loss when target matches exemplar.
        
        Args:
            lambda_ik: Per-exemplar weights
            exemplar_aa_ids: Exemplar amino acid IDs
            c_t: Column mapping
            target_aa_ids: Target amino acid IDs
            copy_eligible: Whether position is copy-eligible
            
        Returns:
            Copy loss
        """
        batch_size, seq_len, num_exemplars = lambda_ik.shape
        
        # Initialize loss
        copy_loss = torch.tensor(0.0, device=self.device)
        num_valid_positions = 0
        
        # Process each position
        for t in range(seq_len):
            # Check if position is copy-eligible
            eligible_positions = copy_eligible[:, t]
            if not eligible_positions.any():
                continue
            
            # Get current column and target
            c_t_current = c_t[eligible_positions, t]  # [eligible_B]
            targets = target_aa_ids[eligible_positions, t]  # [eligible_B]
            
            # Find exemplars with matching AA at current column
            for b_idx, valid_b in enumerate(eligible_positions.nonzero().squeeze(-1)):
                c_col = c_t_current[b_idx]
                target_aa = targets[b_idx]
                
                # Check which exemplars have matching AA
                matching_exemplars = (exemplar_aa_ids[valid_b, :, c_col] == target_aa)
                
                if matching_exemplars.any():
                    # Sum weights for matching exemplars
                    matching_weight = lambda_ik[valid_b, t, matching_exemplars].sum()
                    
                    # Add negative log likelihood
                    if matching_weight > 0:
                        copy_loss -= torch.log(matching_weight + 1e-8)
                        num_valid_positions += 1
        
        # Average over valid positions
        if num_valid_positions > 0:
            copy_loss = copy_loss / num_valid_positions
        
        return copy_loss
