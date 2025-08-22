import subprocess
import tempfile
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from Bio import AlignIO, SeqIO
from Bio.Align import MultipleSeqAlignment
import logging

logger = logging.getLogger(__name__)

class AlignmentProcessor:
    """Handles sequence alignment and HMM operations"""
    
    def __init__(
        self,
        hmmbuild_path: str = "hmmbuild",
        hmmalign_path: str = "hmmalign",
        muscle_path: str = "muscle"
    ):
        self.hmmbuild_path = hmmbuild_path
        self.hmmalign_path = hmmalign_path
        self.muscle_path = muscle_path
        self.profile_hmms = {}
        
    def build_profile_hmm(
        self,
        sequences: List[str],
        motif_positions: Dict[str, List[int]] = None,
        save_path: Optional[str] = None
    ) -> str:
        """Build profile HMM from sequences with motif anchors"""
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            for i, seq in enumerate(sequences):
                f.write(f">seq_{i}\n{seq}\n")
            input_fasta = f.name
        
        # Create MSA using MUSCLE
        output_msa = tempfile.mktemp(suffix='.afa')
        cmd = [self.muscle_path, '-in', input_fasta, '-out', output_msa]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"MUSCLE alignment failed: {e.stderr}")
            raise RuntimeError(f"MUSCLE alignment failed: {e.stderr}")
        
        # Build HMM using HMMER
        output_hmm = save_path or tempfile.mktemp(suffix='.hmm')
        cmd = [self.hmmbuild_path, output_hmm, output_msa]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"HMM build failed: {e.stderr}")
            raise RuntimeError(f"HMM build failed: {e.stderr}")
        
        # Add motif anchors if provided
        if motif_positions:
            self._add_motif_anchors(output_hmm, motif_positions)
        
        logger.info(f"Built profile HMM: {output_hmm}")
        return output_hmm
    
    def _add_motif_anchors(self, hmm_file: str, motif_positions: Dict):
        """Add motif anchor information to HMM"""
        # This would modify the HMM file to mark motif positions
        # For now, we store separately
        self.motif_anchors = motif_positions
    
    def align_to_profile(
        self,
        query_sequence: str,
        profile_hmm: str
    ) -> Dict:
        """Align query to profile HMM using Viterbi"""
        
        # Create temp query file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(f">query\n{query_sequence}\n")
            query_file = f.name
        
        # Run hmmalign
        output_file = tempfile.mktemp(suffix='.sto')
        cmd = [self.hmmalign_path, '--outformat', 'afa', profile_hmm, query_file]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"HMM align failed: {e.stderr}")
            raise RuntimeError(f"HMM align failed: {e.stderr}")
        
        # Parse alignment
        alignment = self._parse_alignment(result.stdout)
        
        # Map to consensus columns
        consensus_mapping = self._map_to_consensus(alignment)
        
        return {
            'alignment': alignment,
            'consensus_columns': consensus_mapping,
            'match_states': self._extract_match_states(alignment)
        }
    
    def _parse_alignment(self, alignment_text: str) -> str:
        """Parse alignment output"""
        # Simple parsing - real implementation would be more robust
        lines = alignment_text.strip().split('\n')
        for line in lines:
            if line.startswith('>'):
                continue
            return line.replace('-', '')
        return ""
    
    def _map_to_consensus(self, alignment: str) -> List[int]:
        """Map alignment positions to consensus columns"""
        consensus = []
        col = 0
        for char in alignment:
            if char != '-':
                consensus.append(col)
            col += 1
        return consensus
    
    def _extract_match_states(self, alignment: str) -> List[bool]:
        """Extract which positions are match states"""
        return [char.isupper() for char in alignment]
    
    def align_to_exemplars(
        self,
        query: str,
        exemplars: List[str]
    ) -> Dict:
        """Align query to multiple exemplars"""
        
        all_sequences = [query] + exemplars
        
        # Create MSA
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            for i, seq in enumerate(all_sequences):
                name = "query" if i == 0 else f"exemplar_{i-1}"
                f.write(f">{name}\n{seq}\n")
            input_file = f.name
        
        # Run MUSCLE
        output_file = tempfile.mktemp(suffix='.afa')
        cmd = [self.muscle_path, '-in', input_file, '-out', output_file]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"MUSCLE alignment failed: {e.stderr}")
            raise RuntimeError(f"MUSCLE alignment failed: {e.stderr}")
        
        # Parse MSA
        alignment = AlignIO.read(output_file, "fasta")
        
        # Compute conservation scores
        conservation = self.compute_conservation(alignment)
        
        # Extract pairwise alignments
        pairwise_alignments = []
        query_aligned = str(alignment[0].seq)
        
        for i in range(1, len(alignment)):
            exemplar_aligned = str(alignment[i].seq)
            pairwise_alignments.append({
                'query': query_aligned,
                'exemplar': exemplar_aligned,
                'exemplar_idx': i - 1
            })
        
        return {
            'alignments': pairwise_alignments,
            'conservation': conservation,
            'msa': alignment
        }
    
    def compute_conservation(self, alignment: MultipleSeqAlignment) -> np.ndarray:
        """Compute per-position conservation scores"""
        
        conservation = []
        
        for col_idx in range(alignment.get_alignment_length()):
            column = alignment[:, col_idx]
            
            # Skip gap-only columns
            if all(res == '-' for res in column):
                conservation.append(0.0)
                continue
            
            # Compute Shannon entropy
            residues = [res for res in column if res != '-']
            if not residues:
                conservation.append(0.0)
                continue
            
            # Count frequencies
            from collections import Counter
            counts = Counter(residues)
            total = len(residues)
            
            # Shannon entropy
            entropy = 0
            for count in counts.values():
                p = count / total
                if p > 0:
                    entropy -= p * np.log2(p)
            
            # Convert to conservation (1 - normalized entropy)
            max_entropy = np.log2(20)  # 20 amino acids
            cons_score = 1 - (entropy / max_entropy)
            conservation.append(cons_score)
        
        return np.array(conservation)
    
    def segment_alignment(
        self,
        alignment: str,
        motif_positions: List[Tuple[int, int]]
    ) -> List[Dict]:
        """Segment alignment into motif and inter-motif regions"""
        
        segments = []
        last_end = 0
        
        for i, (start, end) in enumerate(motif_positions):
            # Inter-motif segment
            if start > last_end:
                segments.append({
                    'type': 'inter_motif',
                    'start': last_end,
                    'end': start,
                    'index': f'segment_{i}'
                })
            
            # Motif segment
            segments.append({
                'type': 'motif',
                'start': start,
                'end': end,
                'index': f'motif_{i}'
            })
            
            last_end = end
        
        # Final segment
        if last_end < len(alignment):
            segments.append({
                'type': 'inter_motif',
                'start': last_end,
                'end': len(alignment),
                'index': f'segment_{len(motif_positions)}'
            })
        
        return segments
