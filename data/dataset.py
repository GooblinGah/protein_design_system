import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Optional
import pickle

class ProteinDesignDataset(Dataset):
    """Main dataset class for protein design training"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        retriever: Optional['ExemplarRetriever'] = None,
        aligner: Optional['AlignmentProcessor'] = None,
        max_length: int = 350,
        exemplars_per_sample: int = 10
    ):
        """
        Args:
            data_path: Path to preprocessed parquet file
            tokenizer: Protein tokenizer instance
            retriever: Exemplar retriever for pointer mechanism
            aligner: Alignment processor for consensus columns
        """
        self.data = pd.read_parquet(data_path)
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.aligner = aligner
        self.max_length = max_length
        self.exemplars_per_sample = exemplars_per_sample
        
        # Cache for alignments
        self.alignment_cache = {}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Basic data
        prompt = row['prompt']
        sequence = row['sequence']
        dsl_spec = row['dsl_spec']
        
        # Tokenize sequence
        seq_tokens = self.tokenizer.encode(sequence)
        
        # Get exemplars if retriever available
        exemplars = None
        alignments = None
        conservation = None
        
        if self.retriever:
            exemplar_seqs, distances = self.retriever.retrieve_exemplars(
                sequence, k=self.exemplars_per_sample
            )
            exemplars = {
                'sequences': exemplar_seqs,
                'distances': distances,
                'tokens': [self.tokenizer.encode(s) for s in exemplar_seqs]
            }
            
            # Get alignments if aligner available
            if self.aligner:
                if idx not in self.alignment_cache:
                    alignment_data = self.aligner.align_to_exemplars(
                        sequence, exemplar_seqs
                    )
                    self.alignment_cache[idx] = alignment_data
                
                alignments = self.alignment_cache[idx]['alignments']
                conservation = self.alignment_cache[idx]['conservation']
        
        return {
            'prompt': prompt,
            'sequence': sequence,
            'seq_tokens': torch.tensor(seq_tokens),
            'dsl_spec': dsl_spec,
            'exemplars': exemplars,
            'alignments': alignments,
            'conservation': conservation,
            'length': len(sequence)
        }
    
    def collate_fn(self, batch):
        """Custom collation with padding and alignment handling"""
        # Find max length in batch
        max_len = max(item['length'] for item in batch)
        
        # Pad sequences
        padded_sequences = []
        attention_masks = []
        
        for item in batch:
            seq = item['seq_tokens']
            pad_len = max_len - len(seq)
            padded_seq = torch.cat([seq, torch.zeros(pad_len, dtype=torch.long)])
            mask = torch.cat([torch.ones(len(seq)), torch.zeros(pad_len)])
            
            padded_sequences.append(padded_seq)
            attention_masks.append(mask)
        
        # Stack tensors
        batch_dict = {
            'sequences': torch.stack(padded_sequences),
            'attention_mask': torch.stack(attention_masks),
            'prompts': [item['prompt'] for item in batch],
            'dsl_specs': [item['dsl_spec'] for item in batch],
        }
        
        # Handle exemplars if present
        if batch[0]['exemplars'] is not None:
            batch_dict['exemplars'] = self._collate_exemplars(batch)
            
        return batch_dict
    
    def _collate_exemplars(self, batch):
        """Collate exemplar data across batch"""
        # Pad exemplar token lists to max_k and max_len
        max_k = max(len(b['exemplars']['tokens']) for b in batch)
        max_len = max(max(len(t) for t in b['exemplars']['tokens']) for b in batch)
        
        pad = lambda t: torch.cat([torch.tensor(t), torch.zeros(max_len - len(t), dtype=torch.long)])
        
        toks = []
        dists = []
        
        for b in batch:
            toks_k = [pad(t) for t in b['exemplars']['tokens']]
            # pad K dimension
            while len(toks_k) < max_k:
                toks_k.append(torch.zeros(max_len, dtype=torch.long))
            toks.append(torch.stack(toks_k))  # [K, L]
            
            dist = b['exemplars']['distances']
            d = np.pad(dist, (0, max_k - len(dist)), constant_values=np.inf)
            dists.append(torch.tensor(d, dtype=torch.float32))
        
        return {
            "tokens": torch.stack(toks),      # [B,K,L]
            "distances": torch.stack(dists)   # [B,K]
        }
