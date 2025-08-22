import numpy as np
import torch
from typing import Dict, List, Tuple
from Bio import pairwise2
import re

class ProteinDesignEvaluator:
    """Comprehensive evaluation metrics for protein design"""
    
    def __init__(self):
        self.metrics_history = []
        
    def evaluate_batch(
        self,
        generated: List[str],
        targets: List[str],
        constraints: List[Dict],
        exemplar_database: List[str] = None
    ) -> Dict:
        """Evaluate a batch of generated sequences"""
        
        results = {
            'constraint_satisfaction': [],
            'motif_correctness': [],
            'identity_scores': [],
            'liability_scores': [],
            'novelty_valid': []
        }
        
        for gen, target, const in zip(generated, targets, constraints):
            # Constraint satisfaction
            const_sat, const_details = self.check_constraints(gen, const)
            results['constraint_satisfaction'].append(const_sat)
            
            # Motif correctness
            motif_score = self.check_motif_correctness(gen, const.get('motifs', []))
            results['motif_correctness'].append(motif_score)
            
            # Identity to target
            identity = self.compute_identity(gen, target)
            results['identity_scores'].append(identity)
            
            # Liability scores
            liabilities = self.compute_liabilities(gen)
            results['liability_scores'].append(liabilities['total_score'])
            
            # Novelty check
            if exemplar_database:
                is_novel, max_identity = self.check_novelty(gen, exemplar_database)
                results['novelty_valid'].append(is_novel)
        
        # Aggregate metrics
        aggregated = {
            'constraint_satisfaction_rate': np.mean(results['constraint_satisfaction']),
            'motif_correctness_mean': np.mean(results['motif_correctness']),
            'identity_mean': np.mean(results['identity_scores']),
            'liability_score_mean': np.mean(results['liability_scores']),
            'novelty_rate': np.mean(results['novelty_valid']) if results['novelty_valid'] else 0.0
        }
        
        return aggregated
    
    def check_constraints(self, sequence: str, constraints: Dict) -> Tuple[bool, Dict]:
        """Check if sequence satisfies all constraints"""
        
        details = {}
        
        # Length constraint
        length_range = constraints.get('length', [220, 350])
        length_valid = length_range[0] <= len(sequence) <= length_range[1]
        details['length'] = length_valid
        
        # Motif constraints
        motifs_valid = True
        for motif in constraints.get('motifs', []):
            pattern = motif['pattern'].replace('X', '.')
            window = motif.get('window', [0, len(sequence)])
            
            # Check if motif exists in window
            subsequence = sequence[window[0]:window[1]]
            if not re.search(pattern, subsequence):
                motifs_valid = False
                break
        
        details['motifs'] = motifs_valid
        
        # Signal peptide (if required)
        if 'secreted' in constraints.get('tags', []):
            # Simple check - real implementation would use SignalP
            signal_valid = sequence.startswith('M') and len(sequence) > 20
            details['signal_peptide'] = signal_valid
        
        all_valid = all(details.values())
        return all_valid, details
    
    def check_motif_correctness(self, sequence: str, motifs: List[Dict]) -> float:
        """Check exact motif placement and order"""
        
        if not motifs:
            return 1.0
        
        correct = 0
        for motif in motifs:
            pattern = motif['pattern'].replace('X', '.')
            window = motif.get('window', [0, len(sequence)])
            
            # Check if motif is in correct window
            subsequence = sequence[window[0]:window[1]]
            if re.search(pattern, subsequence):
                correct += 1
        
        return correct / len(motifs)
    
    def compute_liabilities(self, sequence: str) -> Dict:
        """Compute various liability scores"""
        
        liabilities = {}
        
        # Aggregation propensity (simple hydrophobicity)
        hydrophobic = set('VILMFYW')
        hydro_runs = []
        current_run = 0
        
        for aa in sequence:
            if aa in hydrophobic:
                current_run += 1
            else:
                if current_run > 0:
                    hydro_runs.append(current_run)
                current_run = 0
        
        max_hydro_run = max(hydro_runs) if hydro_runs else 0
        liabilities['max_hydrophobic_run'] = max_hydro_run
        liabilities['aggregation_prone'] = max_hydro_run > 6
        
        # Repeat regions
        repeats = self.find_repeats(sequence)
        liabilities['has_repeats'] = len(repeats) > 0
        liabilities['num_repeats'] = len(repeats)
        
        # Total liability score (higher is worse)
        liabilities['total_score'] = (
            (max_hydro_run / 10) * 0.5 +
            (len(repeats) / 5) * 0.5
        )
        
        return liabilities
    
    def find_repeats(self, sequence: str, min_length: int = 3) -> List[str]:
        """Find repeat regions in sequence"""
        repeats = []
        
        for length in range(min_length, 10):
            for i in range(len(sequence) - length):
                pattern = sequence[i:i + length]
                if sequence.count(pattern) >= 3:
                    repeats.append(pattern)
        
        return list(set(repeats))
    
    def compute_identity(self, seq1: str, seq2: str) -> float:
        """Compute sequence identity"""
        
        if len(seq1) == 0 or len(seq2) == 0:
            return 0.0
        
        alignments = pairwise2.align.globalxx(seq1, seq2)
        if alignments:
            best_alignment = alignments[0]
            matches = sum(1 for a, b in zip(best_alignment[0], best_alignment[1]) if a == b and a != '-')
            return matches / max(len(seq1), len(seq2))
        
        return 0.0
    
    def check_novelty(
        self,
        sequence: str,
        database: List[str],
        max_identity: float = 0.70
    ) -> Tuple[bool, float]:
        """Check if sequence is novel (< max_identity to any database sequence)"""
        
        highest_identity = 0.0
        
        for db_seq in database:
            identity = self.compute_identity(sequence, db_seq)
            highest_identity = max(highest_identity, identity)
            
            if identity >= max_identity:
                return False, identity
        
        return True, highest_identity
