#!/usr/bin/env python3
"""
Validate configuration files and check system requirements.
"""

import yaml
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List
import logging
import hashlib
import torch
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigValidator:
    """Validate configuration files and system requirements."""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Schema definition
        self.required_sections = [
            'model', 'data', 'training', 'retrieval', 'evaluation'
        ]
        
        self.schema = {
            'model': {
                'required': ['d_model', 'n_heads', 'n_layers', 'd_ff', 'max_length'],
                'ranges': {
                    'd_model': (64, 2048),
                    'n_heads': (1, 32),
                    'n_layers': (1, 48),
                    'd_ff': (128, 8192),
                    'max_length': (100, 1000)
                }
            },
            'data': {
                'required': ['train_path', 'val_path', 'test_path', 'min_length', 'max_length'],
                'ranges': {
                    'min_length': (50, 500),
                    'max_length': (100, 1000),
                    'exemplars_per_sample': (1, 100)
                }
            },
            'training': {
                'required': ['batch_size', 'epochs', 'learning_rate'],
                'ranges': {
                    'batch_size': (1, 128),
                    'epochs': (1, 1000),
                    'learning_rate': (1e-6, 1e-2)
                }
            },
            'retrieval': {
                'required': ['embedding_model', 'embedding_dim', 'top_k'],
                'ranges': {
                    'embedding_dim': (64, 4096),
                    'top_k': (1, 100)
                }
            },
            'evaluation': {
                'required': ['max_identity_threshold'],
                'ranges': {
                    'max_identity_threshold': (0.0, 1.0)
                }
            }
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded config from {self.config_path}")
        return config
    
    def validate_schema(self) -> bool:
        """Validate configuration schema."""
        logger.info("Validating configuration schema...")
        
        errors = []
        
        # Check required sections
        for section in self.required_sections:
            if section not in self.config:
                errors.append(f"Missing required section: {section}")
                continue
            
            section_config = self.config[section]
            section_schema = self.schema[section]
            
            # Check required fields
            for field in section_schema['required']:
                if field not in section_config:
                    errors.append(f"Missing required field: {section}.{field}")
            
            # Check numeric ranges
            if 'ranges' in section_schema:
                for field, (min_val, max_val) in section_schema['ranges'].items():
                    if field in section_config:
                        value = section_config[field]
                        if not isinstance(value, (int, float)):
                            errors.append(f"Field {section}.{field} must be numeric, got {type(value)}")
                        elif value < min_val or value > max_val:
                            errors.append(f"Field {section}.{field} = {value} outside range [{min_val}, {max_val}]")
        
        if errors:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            return False
        
        logger.info("Configuration schema validation passed")
        return True
    
    def validate_paths(self) -> bool:
        """Validate that all specified paths exist."""
        logger.info("Validating file paths...")
        
        errors = []
        path_fields = ['train_path', 'val_path', 'test_path', 'index_path']
        
        for section in ['data', 'retrieval']:
            if section in self.config:
                for field in path_fields:
                    if field in self.config[section]:
                        path = Path(self.config[section][field])
                        if not path.exists():
                            errors.append(f"Path does not exist: {section}.{field} = {path}")
        
        if errors:
            logger.error("Path validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            return False
        
        logger.info("Path validation passed")
        return True
    
    def check_external_tools(self) -> bool:
        """Check if required external tools are available."""
        logger.info("Checking external tools...")
        
        tools_to_check = {
            'cd-hit': 'conda install -c bioconda cd-hit',
            'muscle': 'conda install -c bioconda muscle',
            'hmmbuild': 'conda install -c bioconda hmmer',
            'hmmalign': 'conda install -c bioconda hmmer'
        }
        
        missing_tools = []
        
        for tool, install_cmd in tools_to_check.items():
            try:
                result = subprocess.run([tool, '--help'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info(f"✓ {tool} is available")
                else:
                    missing_tools.append((tool, install_cmd))
            except FileNotFoundError:
                missing_tools.append((tool, install_cmd))
        
        if missing_tools:
            logger.warning("Some external tools are missing:")
            for tool, install_cmd in missing_tools:
                logger.warning(f"  - {tool}: {install_cmd}")
            logger.warning("Consider installing missing tools for full functionality")
        else:
            logger.info("All external tools are available")
        
        return len(missing_tools) == 0
    
    def check_system_resources(self) -> bool:
        """Check system resources and capabilities."""
        logger.info("Checking system resources...")
        
        # Check PyTorch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Check memory
        import psutil
        memory = psutil.virtual_memory()
        logger.info(f"System RAM: {memory.total / 1e9:.1f} GB")
        logger.info(f"Available RAM: {memory.available / 1e9:.1f} GB")
        
        # Check disk space
        disk = psutil.disk_usage('.')
        logger.info(f"Disk space: {disk.free / 1e9:.1f} GB available")
        
        return True
    
    def validate_embedding_model(self) -> bool:
        """Validate embedding model configuration."""
        logger.info("Validating embedding model configuration...")
        
        if 'retrieval' not in self.config:
            return True
        
        embedding_model = self.config['retrieval'].get('embedding_model')
        if not embedding_model:
            logger.warning("No embedding model specified in retrieval config")
            return True
        
        # Check if it's a valid ESM2 model
        valid_models = [
            'esm2_t6_8M_UR50D',
            'esm2_t12_35M_UR50D', 
            'esm2_t30_150M_UR50D',
            'esm2_t33_650M_UR50D',
            'esm2_t36_3B_UR50D'
        ]
        
        if embedding_model not in valid_models:
            logger.warning(f"Unknown embedding model: {embedding_model}")
            logger.warning(f"Valid models: {valid_models}")
        
        # Check embedding dimension
        embedding_dim = self.config['retrieval'].get('embedding_dim')
        if embedding_dim:
            expected_dims = {
                'esm2_t6_8M_UR50D': 320,
                'esm2_t12_35M_UR50D': 480,
                'esm2_t30_150M_UR50D': 640,
                'esm2_t33_650M_UR50D': 1280,
                'esm2_t36_3B_UR50D': 2560
            }
            
            if embedding_model in expected_dims and embedding_dim != expected_dims[embedding_model]:
                logger.warning(f"Embedding dimension mismatch: expected {expected_dims[embedding_model]} for {embedding_model}, got {embedding_dim}")
        
        return True
    
    def generate_run_manifest(self, output_dir: str = "runs") -> str:
        """Generate run manifest with git SHA, requirements hash, and config."""
        logger.info("Generating run manifest...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        manifest = {
            'timestamp': str(Path().cwd()),
            'config_file': str(self.config_path),
            'config_hash': self._hash_config(),
            'system_info': self._get_system_info(),
            'dependencies': self._get_dependencies_info()
        }
        
        # Try to get git information
        try:
            git_sha = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                   capture_output=True, text=True, check=True)
            manifest['git_sha'] = git_sha.stdout.strip()
            
            git_status = subprocess.run(['git', 'status', '--porcelain'], 
                                      capture_output=True, text=True, check=True)
            manifest['git_dirty'] = len(git_status.stdout.strip()) > 0
        except (subprocess.CalledProcessError, FileNotFoundError):
            manifest['git_sha'] = 'unknown'
            manifest['git_dirty'] = False
        
        # Save manifest
        manifest_path = output_dir / "run_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Run manifest saved to {manifest_path}")
        return str(manifest_path)
    
    def _hash_config(self) -> str:
        """Generate hash of configuration."""
        config_str = yaml.dump(self.config, default_flow_style=False)
        return hashlib.sha256(config_str.encode()).hexdigest()[:8]
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        import platform
        
        return {
            'platform': platform.platform(),
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'numpy_version': np.__version__
        }
    
    def _get_dependencies_info(self) -> Dict[str, str]:
        """Get dependencies information."""
        try:
            import pkg_resources
            requirements = Path('requirements.txt')
            
            if requirements.exists():
                with open(requirements, 'r') as f:
                    lines = f.readlines()
                
                deps = {}
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '==' in line:
                            package, version = line.split('==')
                            deps[package] = version
                        else:
                            deps[line] = 'latest'
                
                return deps
        except ImportError:
            pass
        
        return {}
    
    def echo_resolved_config(self):
        """Echo resolved configuration for reproducibility."""
        logger.info("Resolved configuration:")
        logger.info("=" * 50)
        
        # Echo key configuration values
        config_str = yaml.dump(self.config, default_flow_style=False, indent=2)
        logger.info(config_str)
        
        logger.info("=" * 50)
    
    def validate_all(self) -> bool:
        """Run all validation checks."""
        logger.info("Running comprehensive configuration validation...")
        
        checks = [
            ('Schema validation', self.validate_schema),
            ('Path validation', self.validate_paths),
            ('External tools', self.check_external_tools),
            ('System resources', self.check_system_resources),
            ('Embedding model', self.validate_embedding_model)
        ]
        
        all_passed = True
        
        for check_name, check_func in checks:
            try:
                logger.info(f"\n--- {check_name} ---")
                if not check_func():
                    all_passed = False
            except Exception as e:
                logger.error(f"{check_name} failed with error: {e}")
                all_passed = False
        
        if all_passed:
            logger.info("\n✅ All validation checks passed!")
        else:
            logger.error("\n❌ Some validation checks failed!")
        
        return all_passed

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate configuration files")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--generate-manifest", action="store_true", help="Generate run manifest")
    parser.add_argument("--output-dir", default="runs", help="Output directory for manifest")
    
    args = parser.parse_args()
    
    try:
        validator = ConfigValidator(args.config)
        
        # Echo resolved config
        validator.echo_resolved_config()
        
        # Run validation
        if validator.validate_all():
            logger.info("Configuration is valid and ready for training!")
            
            # Generate manifest if requested
            if args.generate_manifest:
                manifest_path = validator.generate_run_manifest(args.output_dir)
                logger.info(f"Run manifest generated: {manifest_path}")
            
            sys.exit(0)
        else:
            logger.error("Configuration validation failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
