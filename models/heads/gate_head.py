import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GateHead(nn.Module):
    """
    Gate head for pointer-generator mechanism.
    
    Controls the mixture between vocabulary generation and copy mechanism.
    """
    
    def __init__(self, hidden_dim: int, feature_dim: int):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        
        # MLP for computing gate value
        self.gate_predictor = nn.Sequential(
            nn.Linear(hidden_dim + feature_dim + 1, hidden_dim),  # +1 for motif indicator
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self,
                hidden_states: torch.Tensor,      # [B, T, H]
                column_features: torch.Tensor,     # [B, Lc, F]
                c_t: torch.Tensor,                # [B, T]
                motif_indicators: torch.Tensor,   # [B, T] bool
                gate_bias: Optional[torch.Tensor] = None  # [B, T]
                ) -> torch.Tensor:
        """
        Forward pass of gate head.
        
        Args:
            hidden_states: Decoder hidden states
            column_features: Consensus column features
            c_t: Column mapping for each position
            motif_indicators: Whether position is in motif window
            gate_bias: Optional bias from controller
            
        Returns:
            gate_logits: [B, T] gate logits
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Initialize output
        gate_logits = torch.zeros(batch_size, seq_len, device=self.device)
        
        # Process each position
        for t in range(seq_len):
            # Get current hidden state and column
            h_t = hidden_states[:, t, :]  # [B, T]
            c_t_current = c_t[:, t]      # [B]
            
            # Skip padding positions
            valid_positions = (c_t_current >= 0)
            if not valid_positions.any():
                continue
            
            # Get column features for current position
            col_feats = column_features[valid_positions, c_t_current[valid_positions]]  # [valid_B, F]
            
            # Get motif indicator for current position
            motif_ind = motif_indicators[valid_positions, t].float().unsqueeze(-1)  # [valid_B, 1]
            
            # Concatenate features
            features = torch.cat([h_t[valid_positions], col_feats, motif_ind], dim=-1)
            
            # Compute gate logits
            logits = self.gate_predictor(features).squeeze(-1)  # [valid_B]
            
            # Apply bias if provided
            if gate_bias is not None:
                bias = gate_bias[valid_positions, t]
                logits = logits + bias
            
            # Store logits
            gate_logits[valid_positions, t] = logits
        
        return gate_logits
    
    def compute_gate_loss(self,
                          gate_logits: torch.Tensor,    # [B, T]
                          copy_eligible: torch.Tensor,   # [B, T]
                          target_gate: Optional[torch.Tensor] = None  # [B, T]
                          ) -> torch.Tensor:
        """
        Compute gate loss.
        
        Args:
            gate_logits: Predicted gate logits
            copy_eligible: Whether position is copy-eligible
            target_gate: Target gate values (if None, use copy_eligible)
            
        Returns:
            Gate loss
        """
        if target_gate is None:
            # If copy-eligible, target gate=0 (favor copy)
            # If not copy-eligible, target gate=1 (favor generation)
            target_gate = (~copy_eligible).float()
        
        # Compute binary cross entropy loss
        gate_probs = torch.sigmoid(gate_logits)
        gate_loss = F.binary_cross_entropy(gate_probs, target_gate, reduction='none')
        
        # Only compute loss on non-padding positions
        valid_mask = (gate_logits != 0)  # Simple heuristic for valid positions
        if valid_mask.any():
            gate_loss = gate_loss[valid_mask].mean()
        else:
            gate_loss = torch.tensor(0.0, device=self.device)
        
        return gate_loss
