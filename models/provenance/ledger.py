import json
import hashlib
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class ProvenanceLedger:
    """Hash-chained ledger for tracking protein design provenance."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ledger_file = config.get('paths', {}).get('ledger_file', 'ledger.jsonl')
        self.provenance_dir = config.get('paths', {}).get('provenance_dir', 'provenance/')
        
        # Initialize ledger
        self.ledger_entries = []
        self.last_hash = None
        
        # Load existing ledger if it exists
        self._load_ledger()
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def _load_ledger(self):
        """Load existing ledger from file."""
        try:
            with open(self.ledger_file, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        self.ledger_entries.append(entry)
                        self.last_hash = entry.get('output_hash')
        except FileNotFoundError:
            self.logger.info("No existing ledger found. Starting new ledger.")
        except Exception as e:
            self.logger.error(f"Error loading ledger: {e}")
    
    def add_entry(self,
                  prompt: str,
                  dsl_constraints: Dict[str, Any],
                  retrieval_ids: List[str],
                  model_checkpoint: str,
                  training_config: Dict[str, Any],
                  output_sequences: List[Dict[str, Any]],
                  metrics: Dict[str, float]) -> str:
        """
        Add new entry to the ledger.
        
        Args:
            prompt: Natural language prompt
            dsl_constraints: Compiled DSL constraints
            retrieval_ids: IDs of retrieved exemplars
            model_checkpoint: Model checkpoint hash
            training_config: Training configuration
            output_sequences: Generated sequences
            metrics: Generation metrics
            
        Returns:
            Hash of the new entry
        """
        # Create entry
        entry = {
            'prev_hash': self.last_hash,
            'ts': datetime.now().isoformat(),
            'prompt_sha256': self._sha256(prompt),
            'dsl_sha256': self._sha256(json.dumps(dsl_constraints, sort_keys=True)),
            'retrieval_ids': retrieval_ids,
            'ckpt_sha': model_checkpoint,
            'knobs': {
                'alpha': training_config.get('loss_weights', {}).get('gate', 0.5),
                'beta': training_config.get('loss_weights', {}).get('copy', 0.5),
                'gamma': training_config.get('loss_weights', {}).get('identity', 0.2),
                'tau': training_config.get('novelty', {}).get('max_single_identity', 0.70)
            },
            'monitors': {
                'copy_rate': metrics.get('copy_rate', 0.0),
                'gate_entropy': metrics.get('gate_entropy', 0.0),
                'id_max': metrics.get('max_identity', 0.0)
            }
        }
        
        # Compute output hash
        output_content = json.dumps(output_sequences, sort_keys=True)
        entry['output_hash'] = self._sha256(output_content)
        
        # Add to ledger
        self.ledger_entries.append(entry)
        self.last_hash = entry['output_hash']
        
        # Save to file
        self._save_entry(entry)
        
        # Save detailed provenance
        self._save_provenance(entry, output_sequences)
        
        self.logger.info(f"Added ledger entry with hash: {entry['output_hash']}")
        return entry['output_hash']
    
    def _sha256(self, content: str) -> str:
        """Compute SHA256 hash of content."""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _save_entry(self, entry: Dict[str, Any]):
        """Save ledger entry to file."""
        with open(self.ledger_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    
    def _save_provenance(self, entry: Dict[str, Any], sequences: List[Dict[str, Any]]):
        """Save detailed provenance information."""
        import os
        os.makedirs(self.provenance_dir, exist_ok=True)
        
        # Save per-residue provenance
        for i, seq_data in enumerate(sequences):
            sequence = seq_data['sequence']
            provenance = seq_data.get('provenance', {})
            
            # Create per-residue data
            residue_data = []
            for pos, (aa_id, tier, lambda_val) in enumerate(zip(
                sequence,
                provenance.get('tiers', ['normal'] * len(sequence)),
                provenance.get('lambda_history', [0.0] * len(sequence))
            )):
                residue_data.append({
                    'idx': pos,
                    'aa': aa_id,
                    'tier': tier,
                    'argmax_exemplar': 0,  # Would be computed from lambda
                    'lambda': lambda_val
                })
            
            # Save as parquet
            df = pd.DataFrame(residue_data)
            filename = f"{self.provenance_dir}/sequence_{entry['output_hash'][:8]}_{i}.parquet"
            df.to_parquet(filename, index=False)
    
    def verify_chain(self) -> bool:
        """Verify the integrity of the hash chain."""
        if not self.ledger_entries:
            return True
        
        for i, entry in enumerate(self.ledger_entries):
            if i == 0:
                # First entry should have no previous hash
                if entry.get('prev_hash') is not None:
                    self.logger.error(f"First entry has previous hash: {entry['prev_hash']}")
                    return False
            else:
                # Check that previous hash matches
                prev_entry = self.ledger_entries[i-1]
                expected_prev_hash = prev_entry.get('output_hash')
                actual_prev_hash = entry.get('prev_hash')
                
                if expected_prev_hash != actual_prev_hash:
                    self.logger.error(f"Hash chain broken at entry {i}")
                    self.logger.error(f"Expected: {expected_prev_hash}")
                    self.logger.error(f"Expected: {expected_prev_hash}")
                    self.logger.error(f"Actual: {actual_prev_hash}")
                    return False
        
        self.logger.info("Hash chain verification passed")
        return True
    
    def get_entry(self, entry_hash: str) -> Optional[Dict[str, Any]]:
        """Get ledger entry by hash."""
        for entry in self.ledger_entries:
            if entry.get('output_hash') == entry_hash:
                return entry
        return None
    
    def get_recent_entries(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get most recent ledger entries."""
        return self.ledger_entries[-n:]
    
    def search_entries(self, 
                      prompt_pattern: Optional[str] = None,
                      dsl_pattern: Optional[str] = None,
                      date_range: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Search ledger entries by criteria."""
        results = []
        
        for entry in self.ledger_entries:
            # Check prompt pattern
            if prompt_pattern and prompt_pattern.lower() not in entry.get('prompt_sha256', '').lower():
                continue
            
            # Check DSL pattern
            if dsl_pattern and dsl_pattern.lower() not in entry.get('dsl_sha256', '').lower():
                continue
            
            # Check date range
            if date_range:
                entry_time = datetime.fromisoformat(entry['ts'])
                if not (date_range[0] <= entry_time <= date_range[1]):
                    continue
            
            results.append(entry)
        
        return results
    
    def export_summary(self, filepath: str):
        """Export ledger summary to file."""
        summary = {
            'total_entries': len(self.ledger_entries),
            'first_entry': self.ledger_entries[0]['ts'] if self.ledger_entries else None,
            'last_entry': self.ledger_entries[-1]['ts'] if self.ledger_entries else None,
            'last_hash': self.last_hash,
            'chain_verified': self.verify_chain()
        }
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Ledger summary exported to {filepath}")
    
    def cleanup_old_entries(self, max_age_days: int = 30):
        """Clean up old ledger entries."""
        cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 3600)
        
        # Keep entries newer than cutoff
        self.ledger_entries = [
            entry for entry in self.ledger_entries
            if datetime.fromisoformat(entry['ts']).timestamp() > cutoff_time
        ]
        
        # Rewrite ledger file
        with open(self.ledger_file, 'w') as f:
            for entry in self.ledger_entries:
                f.write(json.dumps(entry) + '\n')
        
        self.logger.info(f"Cleaned up ledger entries older than {max_age_days} days")
