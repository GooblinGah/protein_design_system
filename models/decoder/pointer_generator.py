import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

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
        
        # Decoder layers - using TransformerEncoderLayer for decoder-only model
        import copy
        base_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.decoder_layers = nn.ModuleList([copy.deepcopy(base_layer) for _ in range(n_layers)])
        
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
                sequences: torch.LongTensor,      # [B, T] - renamed from input_ids
                attention_mask: Optional[torch.BoolTensor] = None,  # [B, T] - renamed from attn_mask
                exemplars: Optional[torch.Tensor] = None,           # [B, K, Lc] or dict
                dsl_specs: Optional[list] = None,                  # DSL specifications
                column_feats: Optional[torch.FloatTensor] = None,   # [B, Lc, F] - optional
                c_t: Optional[torch.LongTensor] = None,            # [B, T] - optional
                gate_bias: Optional[torch.Tensor] = None,          # [B, T] - optional
                motif_indicators: Optional[torch.Tensor] = None     # [B, T] - optional
                ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of pointer-generator decoder.
        
        Args:
            sequences: Input token IDs [B, T]
            attention_mask: Attention mask [B, T] (True for real tokens, False for PAD)
            exemplars: Exemplar data - either tensor [B, K, Lc] or dict with 'tokens'
            dsl_specs: DSL specifications for constraints
            column_feats: Consensus column features [B, Lc, F] (optional)
            c_t: Column mapping for each position [B, T] (optional)
            gate_bias: Bias for gate head [B, T] (optional)
            motif_indicators: Motif indicators [B, T] (optional)
            
        Returns:
            Dictionary with keys: logits_vocab, p_copy, gate_logits, lambda_ik
        """
        batch_size, seq_len = sequences.shape
        
        # Handle exemplars - extract tokens if dict
        if isinstance(exemplars, dict) and "tokens" in exemplars:
            exemplar_tokens = exemplars["tokens"]
        else:
            exemplar_tokens = exemplars
        
        # Create position IDs
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        token_embeds = self.token_embedding(sequences)
        position_embeds = self.position_embedding(position_ids)
        
        # Combine embeddings
        embeddings = token_embeds + position_embeds
        embeddings = self.dropout_layer(embeddings)
        
        # Prepare attention masks
        src_key_padding_mask = None
        if attention_mask is not None:
            # Create padding mask for transformer layers (True = ignore, False = attend)
            src_key_padding_mask = ~attention_mask  # Invert: True for PAD tokens to ignore
            
            # Also zero out embeddings for PAD tokens
            padding_mask = ~attention_mask
            embeddings = embeddings.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        
        # Decoder forward pass with proper masking
        hidden_states = embeddings
        for layer in self.decoder_layers:
            # Pass padding mask to transformer layer
            hidden_states = layer(hidden_states, src_key_padding_mask=src_key_padding_mask)
        
        hidden_states = self.layer_norm(hidden_states)
        
        # Vocabulary logits
        logits_vocab = self.output_projection(hidden_states)
        
        # Initialize default values for optional features
        if column_feats is None:
            column_feats = torch.zeros(batch_size, seq_len, 6, device=self.device)
        if c_t is None:
            c_t = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        
        # Copy head (simplified for now - create dummy outputs)
        p_copy = torch.zeros_like(logits_vocab)
        lambda_ik = torch.zeros(batch_size, seq_len, 1, device=self.device)
        
        # Gate head (simplified for now - create dummy outputs)
        # NOTE: This is a placeholder. Any gate-entropy metrics won't be meaningful
        # until you wire a real gate head. Currently fine for CE-only training.
        gate_logits = torch.zeros(batch_size, seq_len, device=self.device)
        
        # Return dictionary format
        return {
            "logits_vocab": logits_vocab,
            "p_copy": p_copy,
            "gate_logits": gate_logits,
            "lambda_ik": lambda_ik
        }
    
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
