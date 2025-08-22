from typing import Dict, List, Tuple, Optional
import re
from Bio import pairwise2

class ConstraintValidator:
    """Validates protein sequences against design constraints"""
    
    def __init__(self):
        self.validation_cache = {}
    
    def validate_sequence(
        self,
        sequence: str,
        constraints: Dict,
        exemplar_database: Optional[List[str]] = None
    ) -> Dict:
        """
        Comprehensive validation of a protein sequence.
        
        Args:
            sequence: Protein sequence to validate
            constraints: Design constraints dictionary
            exemplar_database: Database of exemplar sequences for novelty check
            
        Returns:
            Validation results dictionary
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'details': {}
        }
        
        # Check length constraints
        length_valid, length_details = self._validate_length(sequence, constraints)
        results['details']['length'] = length_details
        if not length_valid:
            results['valid'] = False
            results['errors'].append(f"Length constraint violation: {length_details}")
        
        # Check motif constraints
        motif_valid, motif_details = self._validate_motifs(sequence, constraints)
        results['details']['motifs'] = motif_details
        if not motif_valid:
            results['valid'] = False
            results['errors'].append(f"Motif constraint violation: {motif_details}")
        
        # Check sequence properties
        prop_valid, prop_details = self._validate_properties(sequence, constraints)
        results['details']['properties'] = prop_details
        if not prop_valid:
            results['warnings'].append(f"Property constraint violation: {prop_details}")
        
        # Check novelty if exemplar database provided
        if exemplar_database:
            novelty_valid, novelty_details = self._validate_novelty(sequence, exemplar_database, constraints)
            results['details']['novelty'] = novelty_details
            if not novelty_valid:
                results['warnings'].append(f"Novelty constraint violation: {novelty_details}")
        
        return results
    
    def _validate_length(self, sequence: str, constraints: Dict) -> Tuple[bool, Dict]:
        """Validate sequence length constraints"""
        length_range = constraints.get('length', [220, 350])
        min_len, max_len = length_range
        
        actual_length = len(sequence)
        valid = min_len <= actual_length <= max_len
        
        details = {
            'min_length': min_len,
            'max_length': max_len,
            'actual_length': actual_length,
            'valid': valid
        }
        
        return valid, details
    
    def _validate_motifs(self, sequence: str, constraints: Dict) -> Tuple[bool, Dict]:
        """Validate motif constraints"""
        motifs = constraints.get('motifs', [])
        if not motifs:
            return True, {'motifs_found': 0, 'motifs_required': 0}
        
        motif_results = []
        all_valid = True
        
        for i, motif in enumerate(motifs):
            pattern = motif['pattern'].replace('X', '.')
            window = motif.get('window', [0, len(sequence)])
            
            # Check if motif exists in window
            subsequence = sequence[window[0]:window[1]]
            match = re.search(pattern, subsequence)
            
            if match:
                motif_results.append({
                    'motif_id': i,
                    'pattern': motif['pattern'],
                    'window': window,
                    'found': True,
                    'position': window[0] + match.start(),
                    'sequence': match.group()
                })
            else:
                motif_results.append({
                    'motif_id': i,
                    'pattern': motif['pattern'],
                    'window': window,
                    'found': False,
                    'position': None,
                    'sequence': None
                })
                all_valid = False
        
        details = {
            'motifs_found': sum(1 for m in motif_results if m['found']),
            'motifs_required': len(motifs),
            'motif_details': motif_results
        }
        
        return all_valid, details
    
    def _validate_properties(self, sequence: str, constraints: Dict) -> Tuple[bool, Dict]:
        """Validate sequence property constraints"""
        properties = constraints.get('properties', {})
        if not properties:
            return True, {}
        
        prop_results = {}
        all_valid = True
        
        # Check for signal peptide
        if properties.get('signal_peptide', False):
            has_signal = sequence.startswith('M') and len(sequence) > 20
            prop_results['signal_peptide'] = {
                'required': True,
                'found': has_signal,
                'valid': has_signal
            }
            if not has_signal:
                all_valid = False
        
        # Check for transmembrane domains (simple hydrophobicity)
        if properties.get('transmembrane', False):
            # Simple check for hydrophobic regions
            hydrophobic = set('VILMFYW')
            hydro_count = sum(1 for aa in sequence if aa in hydrophobic)
            hydro_ratio = hydro_count / len(sequence)
            
            has_tm = hydro_ratio > 0.3  # Simple threshold
            prop_results['transmembrane'] = {
                'required': True,
                'found': has_tm,
                'valid': has_tm,
                'hydrophobic_ratio': hydro_ratio
            }
            if not has_tm:
                all_valid = False
        
        return all_valid, prop_results
    
    def _validate_novelty(
        self,
        sequence: str,
        exemplar_database: List[str],
        constraints: Dict
    ) -> Tuple[bool, Dict]:
        """Validate novelty constraints"""
        max_identity = constraints.get('novelty', {}).get('max_identity', 0.70)
        
        highest_identity = 0.0
        closest_exemplar = None
        
        for i, exemplar in enumerate(exemplar_database):
            identity = self._compute_identity(sequence, exemplar)
            if identity > highest_identity:
                highest_identity = identity
                closest_exemplar = i
        
        is_novel = highest_identity < max_identity
        
        details = {
            'max_identity_threshold': max_identity,
            'highest_identity': highest_identity,
            'closest_exemplar_idx': closest_exemplar,
            'is_novel': is_novel
        }
        
        return is_novel, details
    
    def _compute_identity(self, seq1: str, seq2: str) -> float:
        """Compute sequence identity between two sequences"""
        if len(seq1) == 0 or len(seq2) == 0:
            return 0.0
        
        # Use pairwise2 for global alignment
        alignments = pairwise2.align.globalxx(seq1, seq2)
        if alignments:
            best_alignment = alignments[0]
            matches = sum(1 for a, b in zip(best_alignment[0], best_alignment[1]) 
                         if a == b and a != '-')
            return matches / max(len(seq1), len(seq2))
        
        return 0.0
    
    def batch_validate(
        self,
        sequences: List[str],
        constraints: List[Dict],
        exemplar_database: Optional[List[str]] = None
    ) -> List[Dict]:
        """Validate multiple sequences"""
        results = []
        for seq, const in zip(sequences, constraints):
            result = self.validate_sequence(seq, const, exemplar_database)
            results.append(result)
        return results
    
    def get_validation_summary(self, validation_results: List[Dict]) -> Dict:
        """Generate summary statistics from validation results"""
        total = len(validation_results)
        valid_count = sum(1 for r in validation_results if r['valid'])
        error_count = sum(len(r['errors']) for r in validation_results)
        warning_count = sum(len(r['warnings']) for r in validation_results)
        
        return {
            'total_sequences': total,
            'valid_sequences': valid_count,
            'invalid_sequences': total - valid_count,
            'validation_rate': valid_count / total if total > 0 else 0.0,
            'total_errors': error_count,
            'total_warnings': warning_count,
            'avg_errors_per_sequence': error_count / total if total > 0 else 0.0,
            'avg_warnings_per_sequence': warning_count / total if total > 0 else 0.0
        }
