#!/usr/bin/env python3
"""
Comprehensive test script for the Protein Design System.
"""

import sys
import os
import torch
import logging
import tempfile
import shutil
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all modules can be imported."""
    logger.info("Testing module imports...")
    
    try:
        from constraints.fsa import FSAConstraintEngine, create_dfa_table
        from dsl.compiler import DSLCompiler
        from models.tokenizer import ProteinTokenizer
        from models.controller.segmental_hmm import SegmentalHMMController
        from models.heads.copy_head import CopyHead
        from models.heads.gate_head import GateHead
        from models.decoder.pointer_generator import PointerGeneratorDecoder
        from training.loops import TrainingLoop
        from training.curriculum import CurriculumManager
        from training.monitors import TrainingMonitor
        from models.decoding.fsa_constrained import FSAConstrainedDecoder
        from models.provenance.ledger import ProvenanceLedger
        
        logger.info("✅ All modules imported successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_fsa_constraints():
    """Test FSA constraint engine."""
    logger.info("Testing FSA constraint engine...")
    
    try:
        from constraints.fsa import FSAConstraintEngine, create_dfa_table
        
        # Test DFA table creation
        pattern = "G X S X G"
        dfa_table = create_dfa_table(pattern)
        assert dfa_table.shape == (5, 20), f"Expected shape (5, 20), got {dfa_table.shape}"
        
        # Test constraint engine
        engine = FSAConstraintEngine()
        
        # Test data
        windows = torch.tensor([[[10, 15]]], dtype=torch.long)
        dfa_tables = [dfa_table]
        pos1 = torch.tensor([12])
        
        allowed = engine.allowed_tokens(0, windows, dfa_tables, pos1)
        assert allowed.shape == (1, 20), f"Expected shape (1, 20), got {allowed.shape}"
        
        logger.info("✅ FSA constraint engine tests passed")
        return True
    except Exception as e:
        logger.error(f"❌ FSA constraint engine tests failed: {e}")
        return False

def test_dsl_compiler():
    """Test DSL compiler."""
    logger.info("Testing DSL compiler...")
    
    try:
        from dsl.compiler import DSLCompiler
        
        compiler = DSLCompiler()
        
        # Test DSL parsing
        dsl_json = '''
        {
            "length": [230, 330],
            "motifs": [
                {
                    "name": "lipase_gxsxg",
                    "dfa": "G X S X G",
                    "window": [50, 90]
                }
            ],
            "tags": ["pH~7"]
        }
        '''
        
        dsl_obj = compiler.parse_dsl(dsl_json)
        assert 'length' in dsl_obj
        assert 'motifs' in dsl_obj
        assert len(dsl_obj['motifs']) == 1
        
        # Test compilation
        compiled = compiler.compile_to_constraints(dsl_obj)
        assert 'dfa_tables' in compiled
        assert 'windows' in compiled
        
        logger.info("✅ DSL compiler tests passed")
        return True
    except Exception as e:
        logger.error(f"❌ DSL compiler tests failed: {e}")
        return False

def test_tokenizer():
    """Test protein tokenizer."""
    logger.info("Testing protein tokenizer...")
    
    try:
        from models.tokenizer import ProteinTokenizer
        
        tokenizer = ProteinTokenizer()
        
        # Test encoding
        sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        encoded = tokenizer.encode(sequence)
        assert len(encoded) == len(sequence) + 2  # +2 for BOS/EOS
        
        # Test decoding
        decoded = tokenizer.decode(encoded)
        assert decoded == sequence
        
        # Test batch encoding
        sequences = [sequence, "ACDEFGHIKLMNPQRSTVWY"]
        batch = tokenizer.batch_encode(sequences)
        assert 'input_ids' in batch
        assert 'attention_mask' in batch
        
        logger.info("✅ Tokenizer tests passed")
        return True
    except Exception as e:
        logger.error(f"❌ Tokenizer tests failed: {e}")
        return False

def test_controller():
    """Test Segmental-HMM controller."""
    logger.info("Testing Segmental-HMM controller...")
    
    try:
        from models.controller.segmental_hmm import SegmentalHMMController
        
        controller = SegmentalHMMController()
        
        # Test forward pass
        features = torch.randn(2, 13)
        mu, sigma = controller(features)
        assert mu.shape == (2,)
        assert sigma.shape == (2,)
        
        # Test tier computation
        tier, gate_bias, advance_factor = controller.compute_tier(5, 10.0, 2.0)
        assert tier in ["normal", "stretched", "sparse"]
        
        logger.info("✅ Controller tests passed")
        return True
    except Exception as e:
        logger.error(f"❌ Controller tests failed: {e}")
        return False

def test_heads():
    """Test copy and gate heads."""
    logger.info("Testing copy and gate heads...")
    
    try:
        from models.heads.copy_head import CopyHead
        from models.heads.gate_head import GateHead
        
        # Test copy head
        copy_head = CopyHead(128, 128, 6)
        hidden_states = torch.randn(2, 10, 128)
        exemplar_embeddings = torch.randn(2, 8, 20, 128)
        column_feats = torch.randn(2, 20, 6)
        c_t = torch.randint(0, 20, (2, 10))
        exemplar_aa_ids = torch.randint(0, 23, (2, 8, 20))
        
        p_copy, lambda_ik = copy_head(hidden_states, exemplar_embeddings, column_feats, c_t, exemplar_aa_ids)
        assert p_copy.shape == (2, 10, 23)
        assert lambda_ik.shape == (2, 10, 8)
        
        # Test gate head
        gate_head = GateHead(128, 6)
        motif_indicators = torch.randint(0, 2, (2, 10)).bool()
        
        gate_logits = gate_head(hidden_states, column_feats, c_t, motif_indicators)
        assert gate_logits.shape == (2, 10)
        
        logger.info("✅ Heads tests passed")
        return True
    except Exception as e:
        logger.error(f"❌ Heads tests failed: {e}")
        return False

def test_decoder():
    """Test pointer-generator decoder."""
    logger.info("Testing pointer-generator decoder...")
    
    try:
        from models.decoder.pointer_generator import PointerGeneratorDecoder
        
        # Create smaller model for testing
        model = PointerGeneratorDecoder(
            vocab_size=23,
            d_model=128,
            n_layers=2,
            n_heads=4,
            d_ff=512
        )
        
        # Test forward pass
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 23, (batch_size, seq_len))
        exemplars = torch.randint(0, 23, (batch_size, 8, 20))
        column_feats = torch.randn(batch_size, 20, 6)
        c_t = torch.randint(0, 20, (batch_size, seq_len))
        attn_mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool)
        
        logits_vocab, p_copy, gate_logits, lambda_ik = model(
            input_ids, exemplars, column_feats, c_t, attn_mask
        )
        
        assert logits_vocab.shape == (batch_size, seq_len, 23)
        assert p_copy.shape == (batch_size, seq_len, 23)
        assert gate_logits.shape == (batch_size, seq_len)
        assert lambda_ik.shape == (batch_size, seq_len, 8)
        
        logger.info("✅ Decoder tests passed")
        return True
    except Exception as e:
        logger.error(f"❌ Decoder tests failed: {e}")
        return False

def test_training_components():
    """Test training components."""
    logger.info("Testing training components...")
    
    try:
        from training.curriculum import CurriculumManager
        from training.monitors import TrainingMonitor
        
        # Test curriculum manager
        config = {
            'curriculum': {
                'stages': [{'epoch_end': 2}, {'epoch_end': 5}, {'epoch_end': 8}]
            },
            'loss_weights': {'gate': 0.5, 'copy': 0.5, 'identity': 0.2}
        }
        
        curriculum = CurriculumManager(config)
        assert curriculum.current_stage == 0
        
        # Test monitor
        monitor = TrainingMonitor(config)
        metrics = {'copy_rate': 0.3, 'gate_entropy': 0.6}
        monitor.update_metrics(100, metrics)
        
        logger.info("✅ Training components tests passed")
        return True
    except Exception as e:
        logger.error(f"❌ Training components tests failed: {e}")
        return False

def test_end_to_end():
    """Test end-to-end system integration."""
    logger.info("Testing end-to-end system integration...")
    
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Test DSL compilation
        from dsl.compiler import DSLCompiler
        compiler = DSLCompiler()
        
        dsl_json = '''
        {
            "length": [100, 200],
            "motifs": [
                {
                    "name": "test_motif",
                    "dfa": "A B C",
                    "window": [10, 20]
                }
            ]
        }
        '''
        
        dsl_obj = compiler.parse_dsl(dsl_json)
        constraints = compiler.compile_to_constraints(dsl_obj)
        
        # Test FSA constraints
        from constraints.fsa import FSAConstraintEngine, create_dfa_table
        engine = FSAConstraintEngine()
        
        dfa_table = create_dfa_table("A B C")
        windows = torch.tensor([[[10, 20]]])
        pos1 = torch.tensor([15])
        
        allowed = engine.allowed_tokens(0, windows, [dfa_table], pos1)
        
        # Test tokenizer
        from models.tokenizer import ProteinTokenizer
        tokenizer = ProteinTokenizer()
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        encoded = tokenizer.encode(sequence)
        decoded = tokenizer.decode(encoded)
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        logger.info("✅ End-to-end integration tests passed")
        return True
    except Exception as e:
        logger.error(f"❌ End-to-end integration tests failed: {e}")
        return False

def test_identity_computation():
    """Test identity score computation."""
    logger.info("Testing identity score computation...")
    
    try:
        from models.decoding.fsa_constrained import FSAConstrainedDecoder
        from models.controller.segmental_hmm import SegmentalHMMController
        from constraints.fsa import FSAConstraintEngine
        
        # Create mock components
        class MockModel:
            def __init__(self):
                self.device = torch.device('cpu')
            
            def parameters(self):
                return iter([torch.randn(1)])  # Mock parameter iterator
            
            def __call__(self, *args, **kwargs):
                return torch.randn(1, 10, 23), torch.randn(1, 10, 23), torch.randn(1, 10), torch.randn(1, 10, 5)
        
        controller = SegmentalHMMController()
        constraint_engine = FSAConstraintEngine()
        config = {'max_length': 100, 'beam_size': 4, 'novelty': {'max_single_identity': 0.7}}
        
        decoder = FSAConstrainedDecoder(MockModel(), controller, constraint_engine, config)
        
        # Test identity computation
        sequence = [0, 5, 10, 15, 20, 1]  # BOS, AAs, EOS
        exemplars = torch.tensor([
            [0, 5, 10, 15, 20, 1],  # Same sequence
            [0, 6, 11, 16, 21, 1],  # Different sequence
            [0, 5, 10, 15, 20, 1]   # Same sequence
        ])
        
        identity = decoder._compute_identity_score(sequence, exemplars)
        assert 0.0 <= identity <= 1.0, f"Identity should be between 0 and 1, got {identity}"
        
        # Test with identical sequence
        identity_identical = decoder._compute_identity_score(sequence, exemplars[:1])
        assert identity_identical == 1.0, f"Identical sequence should have identity 1.0, got {identity_identical}"
        
        logger.info("✅ Identity computation tests passed")
        return True
    except Exception as e:
        logger.error(f"❌ Identity computation tests failed: {e}")
        return False

def test_motif_snapping():
    """Test motif snapping functionality."""
    logger.info("Testing motif snapping...")
    
    try:
        from models.decoding.fsa_constrained import FSAConstrainedDecoder
        from models.controller.segmental_hmm import SegmentalHMMController
        from constraints.fsa import FSAConstraintEngine
        
        # Create mock components
        class MockModel:
            def __init__(self):
                self.device = torch.device('cpu')
            
            def parameters(self):
                return iter([torch.randn(1)])  # Mock parameter iterator
            
            def __call__(self, *args, **kwargs):
                return torch.randn(1, 10, 23), torch.randn(1, 10, 23), torch.randn(1, 10), torch.randn(1, 10, 5)
        
        controller = SegmentalHMMController()
        constraint_engine = FSAConstraintEngine()
        config = {'max_length': 100, 'beam_size': 4, 'novelty': {'max_single_identity': 0.7}}
        
        decoder = FSAConstrainedDecoder(MockModel(), controller, constraint_engine, config)
        
        # Test motif snapping
        state = {
            'pos1': 50,
            'provenance_cache': {'boundary_reset': False}
        }
        
        dsl_constraints = {
            'windows': [[50, 90]],  # Motif window starting at position 50
            'dfa_tables': [torch.randn(40, 20).bool()]  # Mock DFA table
        }
        
        updated_state = decoder._apply_motif_snapping(state, dsl_constraints)
        
        # Check that motif snapping was applied
        assert updated_state['motif_active'] == True
        assert updated_state['motif_idx'] == 0
        assert updated_state['motif_start'] == 50
        assert updated_state['motif_end'] == 90
        assert updated_state['provenance_cache']['boundary_reset'] == True
        assert 'motif_gate_bias' in updated_state
        
        # Test gate bias computation
        gate_bias = decoder._compute_motif_gate_bias(50, 90, 3)
        assert gate_bias.shape == (3,)
        assert torch.all(gate_bias >= 0.5)  # Should be high gate values
        
        logger.info("✅ Motif snapping tests passed")
        return True
    except Exception as e:
        logger.error(f"❌ Motif snapping tests failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting comprehensive system tests...")
    
    tests = [
        test_imports,
        test_fsa_constraints,
        test_dsl_compiler,
        test_tokenizer,
        test_controller,
        test_heads,
        test_decoder,
        test_training_components,
        test_end_to_end,
        test_identity_computation,
        test_motif_snapping
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            logger.error(f"Test {test.__name__} crashed: {e}")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! System is ready for use.")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed. Please check the implementation.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
