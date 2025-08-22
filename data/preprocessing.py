import pandas as pd
import numpy as np
from Bio import SeqIO
from typing import List, Dict, Tuple, Optional
import re
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ProteinDatasetBuilder:
    """Preprocesses raw protein data into training format"""
    
    def __init__(
        self,
        min_length: int = 220,
        max_length: int = 350,
        identity_threshold: float = 0.30
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.identity_threshold = identity_threshold
        
        # Prompt templates for generation
        self.prompt_templates = [
            "Design a {domain} protein with {motif} motif, approximately {length} amino acids",
            "Create a {localization} {enzyme_class} with {motif} active site",
            "Generate a {function} enzyme containing {motif}, length {length}aa",
            "Synthesize a {domain} family protein, {properties}, with {motif}",
            "Engineer a {localization} protein with {enzyme_class} activity and {motif}"
        ]
        
    def build_dataset(
        self,
        fasta_file: str,
        annotation_file: str,
        output_dir: str
    ) -> Dict[str, str]:
        """Main pipeline to build dataset"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 1. Load and filter sequences
        logger.info("Loading sequences...")
        sequences = self.load_and_filter_sequences(fasta_file)
        
        # 2. Load annotations
        logger.info("Loading annotations...")
        annotations = pd.read_csv(annotation_file)
        
        # 3. Generate prompt-sequence pairs
        logger.info("Generating prompts...")
        pairs = self.generate_prompt_pairs(sequences, annotations)
        
        # 4. Extract motifs
        logger.info("Extracting motifs...")
        pairs = self.extract_motifs(pairs)
        
        # 5. Create splits
        logger.info("Creating train/val/test splits...")
        splits = self.create_splits(pairs)
        
        # 6. Save datasets
        paths = {}
        for split_name, split_data in splits.items():
            path = output_dir / f"{split_name}.parquet"
            split_data.to_parquet(path)
            paths[split_name] = str(path)
            logger.info(f"Saved {split_name}: {len(split_data)} samples to {path}")
        
        return paths
    
    def load_and_filter_sequences(self, fasta_file: str) -> pd.DataFrame:
        """Load sequences and apply length filters"""
        sequences = []
        
        for record in SeqIO.parse(fasta_file, "fasta"):
            seq_len = len(record.seq)
            if self.min_length <= seq_len <= self.max_length:
                sequences.append({
                    'id': record.id,
                    'sequence': str(record.seq),
                    'length': seq_len,
                    'description': record.description
                })
        
        return pd.DataFrame(sequences)
    
    def generate_prompt_pairs(
        self,
        sequences: pd.DataFrame,
        annotations: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate multiple prompts per sequence"""
        pairs = []
        
        for _, seq_row in sequences.iterrows():
            seq_id = seq_row['id']
            sequence = seq_row['sequence']
            
            # Get annotations for this sequence
            seq_annot = annotations[annotations['protein_id'] == seq_id]
            
            if seq_annot.empty:
                continue
            
            annot = seq_annot.iloc[0]
            
            # Generate 3-5 prompts per sequence
            for template in np.random.choice(self.prompt_templates, size=3, replace=False):
                prompt = template.format(
                    domain=annot.get('domain', 'hydrolase'),
                    motif=annot.get('motif', 'GXSXG'),
                    length=seq_row['length'],
                    localization=annot.get('localization', 'cytoplasmic'),
                    enzyme_class=annot.get('ec_class', 'hydrolase'),
                    function=annot.get('function', 'catalytic'),
                    properties=annot.get('properties', 'stable at pH 7')
                )
                
                # Create DSL specification
                dsl_spec = self.create_dsl_spec(annot, seq_row)
                
                pairs.append({
                    'prompt': prompt,
                    'sequence': sequence,
                    'protein_id': seq_id,
                    'dsl_spec': dsl_spec,
                    'length': seq_row['length']
                })
        
        return pd.DataFrame(pairs)
    
    def create_dsl_spec(self, annotations, seq_info) -> Dict:
        """Create DSL specification from annotations"""
        dsl = {
            'length': [self.min_length, min(seq_info['length'] + 20, self.max_length)],
            'motifs': [],
            'tags': []
        }
        
        # Add motifs
        if 'motif' in annotations:
            motif_pattern = annotations['motif']
            # Find motif position in sequence
            motif_start = self.find_motif_position(seq_info['sequence'], motif_pattern)
            if motif_start:
                dsl['motifs'].append({
                    'name': f"{annotations.get('domain', 'domain')}_motif",
                    'pattern': motif_pattern,
                    'window': [max(0, motif_start - 10), motif_start + 40]
                })
        
        # Add tags
        if 'localization' in annotations:
            dsl['tags'].append(annotations['localization'])
        
        return dsl
    
    def find_motif_position(self, sequence: str, pattern: str) -> Optional[int]:
        """Find motif position in sequence"""
        # Convert pattern like "GXSXG" to regex
        regex_pattern = pattern.replace('X', '.')
        match = re.search(regex_pattern, sequence)
        return match.start() if match else None
    
    def extract_motifs(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """Extract and validate motifs"""
        # Implementation for motif extraction
        return pairs
    
    def create_splits(
        self,
        data: pd.DataFrame,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> Dict[str, pd.DataFrame]:
        """Create train/val/test splits using clustering"""
        
        # For now, simple random split
        # TODO: Implement CD-HIT clustering at 30% identity
        
        n = len(data)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        
        # Shuffle
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        splits = {
            'test': data[:n_test],
            'val': data[n_test:n_test + n_val],
            'train': data[n_test + n_val:]
        }
        
        return splits
