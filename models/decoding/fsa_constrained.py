import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
import logging

from constraints.fsa import FSAConstraintEngine
from models.controller.segmental_hmm import SegmentalHMMController


class FSAConstrainedDecoder:
    """
    FSA-constrained decoder for protein sequence generation.
    
    Implements beam search with constraint satisfaction and identity governance.
    """
    
    def __init__(self, 
                 model,
                 controller: SegmentalHMMController,
                 constraint_engine: FSAConstraintEngine,
                 config: Dict[str, Any]):
        self.model = model
        self.controller = controller
        self.constraint_engine = constraint_engine
        self.config = config
        
        # Decoding parameters
        self.max_length = config.get('max_length', 1000)
        self.beam_size = config.get('beam_size', 6)
        self.temperature = config.get('temperature', 1.0)
        self.top_k = config.get('top_k', 50)
        self.top_p = config.get('top_p', 0.9)
        
        # Novelty constraints
        self.max_identity = config.get('novelty', {}).get('max_single_identity', 0.70)
        
        # Device
        self.device = next(model.parameters()).device
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def decode(self,
               prompt: str,
               dsl_constraints: Dict[str, Any],
               exemplars: torch.Tensor,
               column_feats: torch.Tensor,
               c_t: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Decode protein sequence with FSA constraints.
        
        Args:
            prompt: Natural language prompt
            dsl_constraints: Compiled DSL constraints
            exemplars: Exemplar sequences
            column_feats: Column features
            c_t: Column mapping
            
        Returns:
            List of decoded sequences with metadata
        """
        self.logger.info(f"Starting FSA-constrained decoding with beam size {self.beam_size}")
        
        # Store constraints and exemplars for access in other methods
        self.dsl_constraints = dsl_constraints
        self.exemplars = exemplars
        
        # Initialize beam state
        beam_states = self._initialize_beam()
        
        # Decode step by step
        for step in range(self.max_length):
            # Check if all beams are finished
            if all(state['finished'] for state in beam_states):
                break
            
            # Expand beams
            beam_states = self._expand_beams(step, beam_states, dsl_constraints, 
                                           exemplars, column_feats, c_t)
            
            # Prune beams based on identity constraints
            beam_states = self._prune_beams(beam_states)
            
            # Keep top beams
            beam_states = self._select_top_beams(beam_states)
        
        # Finalize sequences
        results = self._finalize_sequences(beam_states)
        
        self.logger.info(f"Decoding completed. Generated {len(results)} sequences")
        return results
    
    def _initialize_beam(self) -> List[Dict[str, Any]]:
        """Initialize beam search state."""
        initial_state = {
            'prefix_ids': [0],  # BOS token
            'logprob': 0.0,
            'pos1': 0,
            'dfa_states': [],
            'segment_id': 0,
            'emitted_j': 0,
            'c_t': 0,
            'gate_bias_state': 0.0,
            'provenance_cache': {
                'lambda_history': [],
                'tiers': [],
                'boundary_reset': False
            },
            'finished': False
        }
        
        return [initial_state]
    
    def _expand_beams(self, 
                      step: int,
                      beam_states: List[Dict[str, Any]],
                      dsl_constraints: Dict[str, Any],
                      exemplars: torch.Tensor,
                      column_feats: torch.Tensor,
                      c_t: torch.Tensor) -> List[Dict[str, Any]]:
        """Expand all beams with next tokens."""
        expanded_states = []
        
        for state in beam_states:
            if state['finished']:
                expanded_states.append(state)
                continue
            
            # Get current position and constraints
            pos1 = state['pos1']
            windows = torch.tensor([dsl_constraints['windows']], device=self.device)
            dfa_tables = [torch.tensor(table, device=self.device) for table in dsl_constraints['dfa_tables']]
            
            # Get allowed tokens from FSA
            allowed_tokens = self.constraint_engine.allowed_tokens(
                step, windows, dfa_tables, torch.tensor([pos1], device=self.device)
            )
            
            # Get model predictions
            input_ids = torch.tensor([state['prefix_ids']], device=self.device)
            logits_vocab, p_copy, gate_logits, lambda_ik = self.model(
                input_ids, exemplars.unsqueeze(0), column_feats.unsqueeze(0), 
                torch.tensor([state['c_t']], device=self.device), None
            )
            
            # Apply constraints
            logits_vocab = logits_vocab.squeeze(0)
            logits_vocab[~allowed_tokens[0]] = float('-inf')
            
            # Get top candidates
            top_logits, top_indices = torch.topk(logits_vocab, self.top_k)
            
            # Expand beam for each candidate
            for logit, token_id in zip(top_logits, top_indices):
                if token_id == 1:  # EOS token
                    new_state = state.copy()
                    new_state['finished'] = True
                    expanded_states.append(new_state)
                else:
                    new_state = self._create_new_state(state, token_id, logit, step)
                    expanded_states.append(new_state)
        
        return expanded_states
    
    def _create_new_state(self, 
                          old_state: Dict[str, Any],
                          token_id: int,
                          logit: float,
                          step: int) -> Dict[str, Any]:
        """Create new beam state."""
        new_state = old_state.copy()
        
        # Update sequence
        new_state['prefix_ids'] = old_state['prefix_ids'] + [token_id.item()]
        new_state['logprob'] = old_state['logprob'] + logit.item()
        
        # Update position
        new_state['pos1'] = step + 1
        
        # Apply motif snapping if entering motif window
        if hasattr(self, 'dsl_constraints'):
            new_state = self._apply_motif_snapping(new_state, self.dsl_constraints)
        
        # Update controller state
        if hasattr(self.controller, 'compute_tier'):
            tier, gate_bias, advance_factor = self.controller.compute_tier(
                new_state['emitted_j'], 0.0, 1.0  # Placeholder values
            )
            new_state['gate_bias_state'] = gate_bias
            new_state['provenance_cache']['tiers'].append(tier)
        
        # Update provenance
        new_state['provenance_cache']['lambda_history'].append(0.0)  # Placeholder
        
        return new_state
    
    def _prune_beams(self, beam_states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prune beams based on identity constraints."""
        pruned_states = []
        
        for state in beam_states:
            # Check identity constraints after some tokens
            if len(state['prefix_ids']) > 10:  # Only check after some tokens
                # Compute actual identity score
                identity_score = self._compute_identity_score(
                    state['prefix_ids'], 
                    self.exemplars
                )
                
                # Enforce hard cap
                if identity_score > self.max_identity:
                    continue  # Prune this beam
            
            pruned_states.append(state)
        
        return pruned_states
    
    def _select_top_beams(self, beam_states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select top beams based on log probability."""
        # Sort by log probability
        beam_states.sort(key=lambda x: x['logprob'], reverse=True)
        
        # Return top beams
        return beam_states[:self.beam_size]
    
    def _finalize_sequences(self, beam_states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Finalize decoded sequences."""
        results = []
        
        for i, state in enumerate(beam_states):
            if not state['finished']:
                # Add EOS token if not finished
                state['prefix_ids'].append(1)
            
            # Compute final identity score
            final_identity = self._compute_identity_score(
                state['prefix_ids'], 
                self.exemplars
            )
            
            # Enforce hard cap at finalization
            if final_identity > self.max_identity:
                self.logger.warning(f"Beam {i} exceeds identity cap: {final_identity:.3f} > {self.max_identity}")
                continue  # Skip this sequence
            
            # Create result
            result = {
                'sequence': state['prefix_ids'],
                'logprob': state['logprob'],
                'length': len(state['prefix_ids']),
                'provenance': {
                    'tiers': state['provenance_cache']['tiers'],
                    'lambda_history': state['provenance_cache']['lambda_history'],
                    'max_identity': final_identity,
                    'boundary_reset': state.get('boundary_reset', False)
                },
                'beam_rank': i
            }
            
            results.append(result)
        
        return results
    
    def _compute_identity_score(self, 
                               sequence: List[int],
                               exemplars: torch.Tensor) -> float:
        """
        Compute global % identity of current prefix vs each exemplar.
        
        Args:
            sequence: Current sequence prefix (including BOS/EOS/PAD tokens)
            exemplars: [K, L] exemplar sequences where K is number of exemplars
            
        Returns:
            Maximum identity score across all exemplars
        """
        # Filter out special tokens (BOS=0, EOS=1, PAD=2)
        valid_tokens = [token for token in sequence if token > 2]
        
        if not valid_tokens:
            return 0.0
        
        num_exemplars = exemplars.shape[0]
        max_identity = 0.0
        
        for k in range(num_exemplars):
            exemplar = exemplars[k]
            
            # Get exemplar tokens (filter special tokens)
            exemplar_tokens = exemplar[exemplar > 2].tolist()
            
            if not exemplar_tokens:
                continue
            
            # Compute identity for this exemplar
            # Use shorter length to avoid index errors
            min_len = min(len(valid_tokens), len(exemplar_tokens))
            
            if min_len == 0:
                continue
            
            # Count matches
            matches = sum(1 for i in range(min_len) if valid_tokens[i] == exemplar_tokens[i])
            identity = matches / min_len
            
            max_identity = max(max_identity, identity)
        
        return max_identity
    
    def _apply_motif_snapping(self, 
                              state: Dict[str, Any],
                              dsl_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply motif snapping when entering motif windows.
        
        When pos1 enters a motif window, snap to the DFA path and smooth gate bias
        for ~3 tokens; set boundary_reset=true in provenance.
        """
        updated_state = state.copy()
        pos1 = state['pos1']
        
        # Check if we're entering any motif window
        motif_windows = dsl_constraints.get('windows', [])
        dfa_tables = dsl_constraints.get('dfa_tables', [])
        
        for motif_idx, (start_pos, end_pos) in enumerate(motif_windows):
            # Check if we just entered this motif window
            if pos1 == start_pos and pos1 < end_pos:
                # Snap to DFA path
                dfa_table = dfa_tables[motif_idx]
                
                # Set motif indicator for next few tokens
                motif_length = min(3, end_pos - start_pos)  # ~3 tokens or remaining window
                
                # Update state with motif snapping info
                updated_state['motif_active'] = True
                updated_state['motif_idx'] = motif_idx
                updated_state['motif_start'] = start_pos
                updated_state['motif_end'] = end_pos
                updated_state['motif_length'] = motif_length
                updated_state['dfa_table'] = dfa_table
                
                # Set boundary reset flag in provenance
                if 'provenance_cache' not in updated_state:
                    updated_state['provenance_cache'] = {}
                if 'boundary_reset' not in updated_state['provenance_cache']:
                    updated_state['provenance_cache']['boundary_reset'] = False
                
                updated_state['provenance_cache']['boundary_reset'] = True
                
                # Smooth gate bias for motif tokens
                if hasattr(self, 'controller'):
                    # Compute smoothed gate bias for motif region
                    motif_gate_bias = self._compute_motif_gate_bias(
                        start_pos, end_pos, motif_length
                    )
                    updated_state['motif_gate_bias'] = motif_gate_bias
                
                break
        
        return updated_state
    
    def _compute_motif_gate_bias(self, start_pos: int, end_pos: int, motif_length: int) -> torch.Tensor:
        """Compute smoothed gate bias for motif region."""
        # Create smooth transition for gate bias
        # Start with lower copy probability (higher gate) and gradually increase
        gate_bias = torch.zeros(motif_length)
        
        for i in range(motif_length):
            # Smooth transition: start with high gate (low copy), end with balanced
            progress = i / (motif_length - 1) if motif_length > 1 else 0.0
            gate_bias[i] = 0.8 - 0.3 * progress  # 0.8 -> 0.5
        
        return gate_bias
