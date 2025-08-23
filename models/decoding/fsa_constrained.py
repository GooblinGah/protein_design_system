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
        
        # PAD token ID for identity computation
        self.pad_id = config.get('model', {}).get('pad_id', 2)
        
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
            'finished': False,
            'provenance_cache': {
                'tiers': [],
                'lambda_history': [],
                'boundary_reset': False
            }
        }
        
        return [initial_state]
    
    def _expand_beams(self, 
                      step: int,
                      beam_states: List[Dict[str, Any]],
                      dsl_constraints: Dict[str, Any],
                      exemplars: torch.Tensor,
                      column_feats: torch.Tensor,
                      c_t: torch.Tensor) -> List[Dict[str, Any]]:
        """Expand all beams with next token predictions."""
        expanded_states = []
        
        for state in beam_states:
            if state['finished']:
                expanded_states.append(state)
                continue
            
            # Get next token predictions
            next_tokens = self._get_next_tokens(state, dsl_constraints, exemplars, 
                                              column_feats, c_t)
            
            # Create new states for each token
            for token_id, logit in next_tokens:
                new_state = self._create_new_state(state, token_id, logit, step)
                
                # Apply motif snapping
                new_state = self._apply_motif_snapping(new_state, dsl_constraints)
                
                # Check identity cap during expansion (fast estimate)
                if len(new_state['prefix_ids']) > 5:  # Check early
                    max_id = self._fast_identity_estimate(
                        new_state['prefix_ids'], 
                        exemplars
                    )
                    if max_id >= self.max_identity:
                        # Prune beam by setting score to -inf
                        new_state['logprob'] = float('-inf')
                        self.logger.debug(f"Pruned beam at step {step}: identity {max_id:.3f} >= {self.max_identity}")
                
                expanded_states.append(new_state)
        
        return expanded_states
    
    def _fast_identity_estimate(self, 
                               prefix_ids: List[int], 
                               exemplars: torch.Tensor) -> float:
        """
        Fast identity estimation for beam pruning.
        Uses prefix matching for speed during expansion.
        """
        # Filter out special tokens
        valid_tokens = [token for token in prefix_ids if token > 2]
        
        if not valid_tokens:
            return 0.0
        
        max_identity = 0.0
        
        for k in range(exemplars.shape[0]):
            exemplar = exemplars[k]
            exemplar_tokens = exemplar[exemplar > 2].tolist()
            
            if not exemplar_tokens:
                continue
            
            # Use shorter length for prefix matching
            min_len = min(len(valid_tokens), len(exemplar_tokens))
            if min_len == 0:
                continue
            
            # Count matches (fast prefix comparison)
            matches = sum(1 for i in range(min_len) 
                         if valid_tokens[i] == exemplar_tokens[i])
            
            # Estimate identity (conservative estimate)
            identity = matches / min_len
            max_identity = max(max_identity, identity)
            
            # Early exit if we're already over threshold
            if max_identity >= self.max_identity:
                break
        
        return max_identity
    
    def _prune_beams(self, beam_states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prune beams based on constraints and identity."""
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
            
            # Compute final identity score using full alignment
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
    
    def _compute_identity_score(self, prefix_ids: torch.Tensor, exemplars) -> float:
        """
        Approximate max identity of the current prefix against K exemplars.
        Works on token IDs (AA vocab only), ignores pads/specials.
        Returns a value in [0,1].
        """
        if isinstance(exemplars, dict):
            ex_tokens = exemplars["tokens"]     # [B,K,L]
        else:
            ex_tokens = exemplars                # [B,K,L]
        
        # Use CPU numpy for simplicity here
        import numpy as np
        p = prefix_ids.detach().cpu().numpy()    # [T]
        T = len(p)
        if T == 0:
            return 0.0
        
        ex = ex_tokens[0].detach().cpu().numpy() # assume batch size 1 inside decode
        if ex.ndim == 1:
            # Single exemplar case
            K, L = 1, ex.shape[0]
            ex = ex.reshape(1, -1)
        else:
            K, L = ex.shape
        
        # sliding compare of prefix against each exemplar; take best window
        best = 0.0
        for k in range(K):
            row = ex[k]
            # skip pads (assume pad_id known)
            pad_id = getattr(self, "pad_id", 2)
            row = row[row != pad_id]
            for s in range(0, max(1, len(row) - T + 1)):
                window = row[s:s+T]
                if len(window) != T: 
                    break
                ident = (window == p).mean()
                if ident > best:
                    best = float(ident)
        return best
    
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

    def _get_next_tokens(self, 
                         state: Dict[str, Any],
                         dsl_constraints: Dict[str, Any],
                         exemplars: torch.Tensor,
                         column_feats: torch.Tensor,
                         c_t: torch.Tensor) -> List[Tuple[int, float]]:
        """Get next token predictions for a beam state."""
        # Get current position and constraints
        pos1 = state['pos1']
        windows = torch.tensor([dsl_constraints['windows']], device=self.device)
        dfa_tables = [torch.tensor(table, device=self.device) for table in dsl_constraints['dfa_tables']]
        
        # Get allowed tokens from FSA
        allowed_tokens = self.constraint_engine.allowed_tokens(
            pos1, windows, dfa_tables, torch.tensor([pos1], device=self.device)
        )
        
        # Get model predictions
        input_ids = torch.tensor([state['prefix_ids']], device=self.device)
        logits_vocab, p_copy, gate_logits, lambda_ik = self.model(
            input_ids, exemplars.unsqueeze(0), column_feats.unsqueeze(0), 
            torch.tensor([c_t], device=self.device), None
        )
        
        # Apply constraints
        logits_vocab = logits_vocab.squeeze(0)
        logits_vocab[~allowed_tokens[0]] = float('-inf')
        
        # Get top candidates
        top_logits, top_indices = torch.topk(logits_vocab, self.top_k)
        
        # Return token candidates
        tokens = []
        for logit, token_id in zip(top_logits, top_indices):
            if token_id == 1:  # EOS token
                # Mark state as finished
                state['finished'] = True
                tokens.append((token_id.item(), logit.item()))
            else:
                tokens.append((token_id.item(), logit.item()))
        
        return tokens
    
    def _create_new_state(self, 
                          old_state: Dict[str, Any],
                          token_id: int,
                          logit: float,
                          step: int) -> Dict[str, Any]:
        """Create new beam state."""
        new_state = old_state.copy()
        
        # Update sequence
        new_state['prefix_ids'] = old_state['prefix_ids'] + [token_id]
        new_state['logprob'] = old_state['logprob'] + logit
        
        # Update position
        new_state['pos1'] = step + 1
        
        # Update controller state
        if hasattr(self.controller, 'compute_tier'):
            tier, gate_bias, advance_factor = self.controller.compute_tier(
                step, 0.0, 1.0  # Placeholder values
            )
            new_state['provenance_cache']['tiers'].append(tier)
        
        # Update provenance
        new_state['provenance_cache']['lambda_history'].append(0.0)  # Placeholder
        
        return new_state
