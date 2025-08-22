import torch
import torch.nn as nn
from typing import List, Tuple


class FSAConstraintEngine:
    """Vectorized FSA constraint engine for motif validation."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def allowed_tokens(self, 
                      step: int,
                      windows: torch.IntTensor,  # [B, M, 2]
                      dfa_tables: List[torch.BoolTensor],  # M × [L, 20]
                      pos1: torch.IntTensor  # [B]
                      ) -> torch.BoolTensor:  # [B, 20]
        """
        Compute allowed amino acids at step t given active motif windows.
        
        Args:
            step: Current decoding step
            windows: [B, M, 2] start/end positions for M motifs
            dfa_tables: List of M DFA tables, each [L, 20] bool
            pos1: [B] current position in sequence for each batch item
            
        Returns:
            [B, 20] boolean mask of allowed amino acids
        """
        batch_size = pos1.shape[0]
        num_motifs = len(dfa_tables)
        
        # Initialize mask as all-ones (no constraints)
        allowed_mask = torch.ones(batch_size, 20, dtype=torch.bool, device=self.device)
        
        if num_motifs == 0:
            return allowed_mask
        
        # For each motif, check if it's active at current position
        for motif_idx in range(num_motifs):
            # Get window bounds for this motif
            motif_windows = windows[:, motif_idx, :]  # [B, 2]
            start_pos = motif_windows[:, 0]  # [B]
            end_pos = motif_windows[:, 1]  # [B]
            
            # Check if current position is within motif window
            in_window = (pos1 >= start_pos) & (pos1 < end_pos)  # [B]
            
            if not in_window.any():
                continue
            
            # Get DFA table for this motif
            dfa_table = dfa_tables[motif_idx]  # [L, 20]
            motif_length = dfa_table.shape[0]
            
            # For positions in window, compute offset within motif
            motif_offset = pos1 - start_pos  # [B]
            
            # Only apply constraints where offset is valid
            valid_offset = (motif_offset >= 0) & (motif_offset < motif_length)  # [B]
            active_constraints = in_window & valid_offset  # [B]
            
            if not active_constraints.any():
                continue
            
            # Get allowed AAs for current offset
            offset_indices = motif_offset[active_constraints]  # [active_B]
            allowed_at_offset = dfa_table[offset_indices]  # [active_B, 20]
            
            # Apply constraints by intersecting with current mask
            allowed_mask[active_constraints] &= allowed_at_offset
        
        return allowed_mask
    
    def validate_sequence(self,
                         sequence: torch.LongTensor,  # [B, T]
                         windows: torch.IntTensor,    # [B, M, 2]
                         dfa_tables: List[torch.BoolTensor]  # M × [L, 20]
                         ) -> torch.BoolTensor:  # [B]
        """
        Validate entire sequences against all motif constraints.
        
        Args:
            sequence: [B, T] amino acid sequences
            windows: [B, M, 2] motif windows
            dfa_tables: List of M DFA tables
            
        Returns:
            [B] boolean mask of valid sequences
        """
        batch_size, seq_len = sequence.shape
        valid_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        
        for step in range(seq_len):
            pos1 = torch.full((batch_size,), step, dtype=torch.long, device=self.device)
            allowed = self.allowed_tokens(step, windows, dfa_tables, pos1)
            
            # Check if current token is allowed
            current_tokens = sequence[:, step]  # [B]
            token_allowed = allowed[torch.arange(batch_size), current_tokens]  # [B]
            
            # Update validity mask
            valid_mask &= token_allowed
            
            # Early termination if no sequences are valid
            if not valid_mask.any():
                break
        
        return valid_mask


def create_dfa_table(pattern: str, alphabet: str = "ACDEFGHIKLMNPQRSTVWY") -> torch.BoolTensor:
    """
    Create DFA table from pattern string.
    
    Args:
        pattern: Pattern like "G X S X G" where X means any AA
        alphabet: Amino acid alphabet
        
    Returns:
        [L, 20] boolean table where True means allowed
    """
    aa_to_idx = {aa: idx for idx, aa in enumerate(alphabet)}
    pattern_tokens = pattern.split()
    
    dfa_table = torch.zeros(len(pattern_tokens), len(alphabet), dtype=torch.bool)
    
    for pos, token in enumerate(pattern_tokens):
        if token == "X":
            # X means any amino acid is allowed
            dfa_table[pos, :] = True
        else:
            # Specific amino acid
            if token in aa_to_idx:
                dfa_table[pos, aa_to_idx[token]] = True
    
    return dfa_table
