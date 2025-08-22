import torch
from typing import List, Tuple, Dict, Optional
import numpy as np


class ProteinTokenizer:
    """Tokenizer for protein sequences with special tokens."""
    
    def __init__(self):
        # Standard amino acid alphabet
        self.aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"
        
        # Special tokens
        self.special_tokens = ["[BOS]", "[EOS]", "[PAD]"]
        
        # Create vocabulary
        self.vocab = self.special_tokens + list(self.aa_alphabet)
        self.vocab_size = len(self.vocab)
        
        # Create token to id mapping
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}
        
        # Special token IDs
        self.bos_id = self.token_to_id["[BOS]"]
        self.eos_id = self.token_to_id["[EOS]"]
        self.pad_id = self.token_to_id["[PAD]"]
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def encode(self, sequence: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode protein sequence to token IDs.
        
        Args:
            sequence: Protein sequence string
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of token IDs
        """
        # Validate sequence
        if not all(aa in self.aa_alphabet for aa in sequence):
            raise ValueError(f"Invalid amino acid in sequence: {sequence}")
        
        # Convert to IDs
        token_ids = [self.token_to_id[aa] for aa in sequence]
        
        if add_special_tokens:
            token_ids = [self.bos_id] + token_ids + [self.eos_id]
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to protein sequence.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Protein sequence string
        """
        if skip_special_tokens:
            token_ids = [tid for tid in token_ids 
                        if tid not in [self.bos_id, self.eos_id, self.pad_id]]
        
        return "".join([self.id_to_token[tid] for tid in token_ids])
    
    def batch_encode(self, sequences: List[str], 
                    add_special_tokens: bool = True,
                    padding: bool = True,
                    return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        """
        Encode batch of sequences.
        
        Args:
            sequences: List of protein sequences
            add_special_tokens: Whether to add BOS/EOS tokens
            padding: Whether to pad to max length
            return_tensors: Return format ("pt" for PyTorch)
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Encode all sequences
        encoded_sequences = [self.encode(seq, add_special_tokens) for seq in sequences]
        
        if padding:
            # Find max length
            max_length = max(len(seq) for seq in encoded_sequences)
            
            # Pad sequences
            padded_sequences = []
            attention_masks = []
            
            for seq in encoded_sequences:
                # Pad with PAD token
                padded_seq = seq + [self.pad_id] * (max_length - len(seq))
                padded_sequences.append(padded_seq)
                
                # Create attention mask (1 for real tokens, 0 for padding)
                attention_mask = [1] * len(seq) + [0] * (max_length - len(seq))
                attention_masks.append(attention_mask)
            
            encoded_sequences = padded_sequences
        else:
            attention_masks = [[1] * len(seq) for seq in encoded_sequences]
        
        # Convert to tensors
        if return_tensors == "pt":
            input_ids = torch.tensor(encoded_sequences, dtype=torch.long, device=self.device)
            attention_mask = torch.tensor(attention_masks, dtype=torch.bool, device=self.device)
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
        else:
            return {
                "input_ids": encoded_sequences,
                "attention_mask": attention_masks
            }
    
    def create_causal_mask(self, seq_length: int) -> torch.Tensor:
        """
        Create causal attention mask for decoder-only model.
        
        Args:
            seq_length: Length of sequence
            
        Returns:
            [seq_length, seq_length] causal mask
        """
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1)
        return mask.bool().to(self.device)
    
    def create_padding_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Create padding mask from input IDs.
        
        Args:
            input_ids: [B, T] input token IDs
            
        Returns:
            [B, T] padding mask (True for real tokens, False for padding)
        """
        return (input_ids != self.pad_id).to(self.device)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size
    
    def get_special_token_ids(self) -> Dict[str, int]:
        """Get special token IDs."""
        return {
            "bos": self.bos_id,
            "eos": self.eos_id,
            "pad": self.pad_id
        }
