import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from models.heads.copy_head import CopyHead
from models.heads.gate_head import GateHead


class PointerGeneratorDecoder(nn.Module):
    """
    Pointer-Generator decoder for protein sequence generation.
    
    Combines vocabulary generation with copy mechanism from exemplars.
    """
    
    def __init__(self, 
                 vocab_size: int = 23,
                 d_model: int = 896,
                 n_layers: int = 14,
                 n_heads: int = 14,
                 d_ff: int = 3584,
                 dropout: float = 0.1,
                 max_seq_length: int = 1024):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.max_seq_length = max_seq_length
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        
        # Decoder layers
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.decoder_layers = nn.ModuleList([decoder_layer for _ in range(n_layers)])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Copy and gate heads
        self.copy_head = CopyHead(d_model, d_model, 6)  # Assuming 6 column features
        self.gate_head = GateHead(d_model, 6)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self,
                input_ids: torch.LongTensor,      # [B, T]
                exemplars: torch.LongTensor,      # [B, K, Lc]
                column_feats: torch.FloatTensor,  # [B, Lc, F]
                c_t: torch.LongTensor,            # [B, T]
                attn_mask: torch.BoolTensor,      # [B, 1, T, T]
                exemplar_embeddings: Optional[torch.Tensor] = None,  # [B, K, Lc, E]
                gate_bias: Optional[torch.Tensor] = None,           # [B, T]
                motif_indicators: Optional[torch.Tensor] = None     # [B, T]
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of pointer-generator decoder.
        
        Args:
            input_ids: Input token IDs
            exemplars: Exemplar amino acid IDs
            column_feats: Consensus column features
            c_t: Column mapping for each position
            attn_mask: Attention mask
            exemplar_embeddings: Pre-computed exemplar embeddings
            gate_bias: Bias for gate head
            motif_indicators: Whether position is in motif window
            
        Returns:
            logits_vocab: [B, T, V] vocabulary logits
            p_copy: [B, T, V] sparse copy distribution
            gate_logits: [B, T] gate logits
            lambda_ik: [B, T, K] per-exemplar weights
        """
        batch_size, seq_len = input_ids.shape
        
        # Create position IDs
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        
        # Combine embeddings
        embeddings = token_embeds + position_embeds
        embeddings = self.dropout_layer(embeddings)
        
        # Create causal mask for decoder
        causal_mask = self._create_causal_mask(seq_len)
        
        # Decoder forward pass
        hidden_states = embeddings
        for layer in self.decoder_layers:
            hidden_states = layer(
                hidden_states,
                memory=None,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=(input_ids == 0)  # PAD token
            )
        
        hidden_states = self.layer_norm(hidden_states)
        
        # Vocabulary logits
        logits_vocab = self.output_projection(hidden_states)
        
        # Copy head
        if exemplar_embeddings is None:
            # Create exemplar embeddings if not provided
            exemplar_embeddings = self.token_embedding(exemplars)
        
        p_copy, lambda_ik = self.copy_head(
            hidden_states, exemplar_embeddings, column_feats, c_t, exemplars
        )
        
        # Gate head
        if motif_indicators is None:
            # Create default motif indicators (all False)
            motif_indicators = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=self.device)
        
        gate_logits = self.gate_head(
            hidden_states, column_feats, c_t, motif_indicators, gate_bias
        )
        
        return logits_vocab, p_copy, gate_logits, lambda_ik
    
    def _create_causal_mask(self, seq_length: int) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1)
        return mask.bool().to(self.device)
    
    def compute_mixture_distribution(self,
                                   logits_vocab: torch.Tensor,  # [B, T, V]
                                   p_copy: torch.Tensor,        # [B, T, V]
                                   gate_logits: torch.Tensor    # [B, T]
                                   ) -> torch.Tensor:
        """
        Compute final mixture distribution.
        
        Args:
            logits_vocab: Vocabulary logits
            p_copy: Copy distribution
            gate_logits: Gate logits
            
        Returns:
            Final distribution [B, T, V]
        """
        # Apply softmax to vocabulary logits
        p_vocab = F.softmax(logits_vocab, dim=-1)
        
        # Apply sigmoid to gate logits
        gate_probs = torch.sigmoid(gate_logits).unsqueeze(-1)  # [B, T, 1]
        
        # Mixture: p = gate * p_vocab + (1 - gate) * p_copy
        p_final = gate_probs * p_vocab + (1 - gate_probs) * p_copy
        
        return p_final
    
    def get_parameter_groups(self):
        """Get parameter groups for different learning rates."""
        return {
            'decoder': list(self.decoder_layers.parameters()) + 
                      list(self.token_embedding.parameters()) +
                      list(self.position_embedding.parameters()) +
                      list(self.output_projection.parameters()),
            'pointer': list(self.copy_head.parameters()),
            'gate': list(self.gate_head.parameters())
        }
