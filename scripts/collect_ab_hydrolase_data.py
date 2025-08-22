#!/usr/bin/env python3
"""
Collect Alpha/Beta Hydrolase superfamily data for protein design training
"""

import requests
import pandas as pd
import numpy as np
from Bio import SeqIO
from pathlib import Path
import time
import logging
import json
import re
from typing import Dict, List, Tuple
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ABHydrolaseCollector:
    """Specialized collector for Alpha/Beta Hydrolase superfamily"""
    
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Known motifs in alpha/beta hydrolases
        self.known_motifs = {
            'nucleophile_elbow': {
                'pattern': 'G[LIVMFY]S[LIVMFY]G',  # GXSXG motif
                'description': 'Nucleophile elbow',
                'typical_position': [100, 180]  # Typical position range
            },
            'catalytic_serine': {
                'pattern': 'S[DE][HY]',  # Catalytic triad pattern
                'description': 'Catalytic triad region',
                'typical_position': [150, 250]
            },
            'oxyanion_hole': {
                'pattern': '[HQN]G[GA]',
                'description': 'Oxyanion hole',
                'typical_position': [70, 120]
            },
            'beta_strand_5': {
                'pattern': '[VIL][VIL][LIVMFY][LIVMFY]G',
                'description': 'Beta strand 5 signature',
                'typical_position': [90, 140]
            }
        }
        
    def collect_complete_dataset(self) -> Dict[str, pd.DataFrame]:
        """Collect comprehensive AB hydrolase dataset"""
        
        logger.info("Starting Alpha/Beta Hydrolase data collection...")
        
        # Step 1: Download sequences
        sequences_df = self.download_ab_hydrolase_sequences()
        
        # Step 2: Download detailed annotations
        annotations_df = self.download_detailed_annotations(sequences_df)
        
        # Step 3: Extract motif information
        motifs_df = self.extract_motif_information(sequences_df)
        
        # Step 4: Download structural representatives
        structural_df = self.download_structural_examples()
        
        # Step 5: Create prompt templates
        prompts_df = self.generate_training_prompts(annotations_df, motifs_df)
        
        # Step 6: Validate and clean
        cleaned_data = self.validate_and_clean(sequences_df, annotations_df, motifs_df)
        
        return {
            'sequences': cleaned_data['sequences'],
            'annotations': cleaned_data['annotations'],
            'motifs': cleaned_data['motifs'],
            'prompts': prompts_df,
            'structural': structural_df
        }
    
    def download_ab_hydrolase_sequences(self) -> pd.DataFrame:
        """Download all alpha/beta hydrolase sequences from UniProt"""
        
        logger.info("Downloading sequences from UniProt...")
        
        # UniProt REST API query
        query = ' AND '.join([
            '(family:"alpha/beta hydrolase" OR family:"ab hydrolase" OR pfam:PF00561)',
            '(reviewed:true)',  # Swiss-Prot only
            '(length:[220 TO 350])',  # Length constraints
            '(organism_id:9606 OR organism_id:10090 OR organism_id:559292 OR organism_id:83333 OR organism_id:3702)',  # Common organisms
        ])
        
        # Download sequences
        base_url = "https://rest.uniprot.org/uniprotkb/stream"
        
        # Get FASTA
        fasta_url = f"{base_url}?format=fasta&query={query}&size=500"
        
        sequences = []
        offset = 0
        batch_size = 500
        
        while True:
            logger.info(f"Downloading batch starting at {offset}...")
            
            response = requests.get(f"{fasta_url}&offset={offset}")
            
            if response.status_code != 200:
                logger.error(f"Failed to download: {response.status_code}")
                break
            
            if not response.text.strip():
                break
            
            # Parse sequences
            temp_file = self.output_dir / f"temp_batch_{offset}.fasta"
            with open(temp_file, 'w') as f:
                f.write(response.text)
            
            for record in SeqIO.parse(temp_file, "fasta"):
                sequences.append({
                    'accession': record.id.split('|')[1] if '|' in record.id else record.id,
                    'full_id': record.id,
                    'description': record.description,
                    'sequence': str(record.seq),
                    'length': len(record.seq)
                })
            
            temp_file.unlink()  # Delete temp file
            
            if len(sequences) >= 50000:  # Limit to 50k sequences
                break
            
            offset += batch_size
            time.sleep(0.5)  # Be nice to the server
        
        df = pd.DataFrame(sequences)
        
        # Save raw sequences
        output_file = self.output_dir / "ab_hydrolases_raw.fasta"
        with open(output_file, 'w') as f:
            for _, row in df.iterrows():
                f.write(f">{row['full_id']} {row['description']}\n")
                f.write(f"{row['sequence']}\n")
        
        logger.info(f"Downloaded {len(df)} sequences")
        
        return df
    
    def download_detailed_annotations(self, sequences_df: pd.DataFrame) -> pd.DataFrame:
        """Download detailed annotations for each sequence"""
        
        logger.info("Downloading detailed annotations...")
        
        # Prepare accessions
        accessions = sequences_df['accession'].unique()
        
        # UniProt fields to retrieve
        fields = [
            'accession', 'id', 'protein_name', 'gene_names',
            'organism_name', 'organism_id', 'ec', 
            'go_id', 'go_f', 'go_p', 'go_c',  # GO annotations
            'interpro', 'pfam', 'supfam',  # Domain annotations
            'ft_motif', 'ft_domain', 'ft_region',  # Features
            'ft_act_site', 'ft_binding', 'ft_site',  # Active sites
            'cc_function', 'cc_catalytic_activity',  # Function
            'keywords', 'ft_signal', 'ft_transmem',  # Localization
            'sequence_length', 'mass', 'cc_subcellular_location'
        ]
        
        # Download in batches
        batch_size = 100
        all_annotations = []
        
        for i in tqdm(range(0, len(accessions), batch_size)):
            batch = accessions[i:i+batch_size]
            
            # Create query for batch
            acc_query = ' OR '.join([f'(accession:{acc})' for acc in batch])
            
            url = f"https://rest.uniprot.org/uniprotkb/stream?format=tsv&fields={','.join(fields)}&query={acc_query}"
            
            try:
                batch_df = pd.read_csv(url, sep='\t')
                all_annotations.append(batch_df)
            except Exception as e:
                logger.error(f"Failed to download batch {i}: {e}")
                continue
            
            time.sleep(0.2)
        
        annotations_df = pd.concat(all_annotations, ignore_index=True)
        
        # Save annotations
        annotations_df.to_csv(self.output_dir / "ab_hydrolases_annotations.csv", index=False)
        
        logger.info(f"Downloaded annotations for {len(annotations_df)} sequences")
        
        return annotations_df
    
    def extract_motif_information(self, sequences_df: pd.DataFrame) -> pd.DataFrame:
        """Extract and validate known AB hydrolase motifs"""
        
        logger.info("Extracting motif information...")
        
        motif_data = []
        
        for _, row in tqdm(sequences_df.iterrows(), total=len(sequences_df)):
            sequence = row['sequence']
            accession = row['accession']
            
            motifs_found = {}
            
            # Search for each known motif
            for motif_name, motif_info in self.known_motifs.items():
                pattern = motif_info['pattern']
                
                # Convert pattern to regex (X = any amino acid)
                regex_pattern = pattern.replace('[', '').replace(']', '')
                regex_pattern = re.sub(r'([A-Z])', r'[\1]', regex_pattern)
                regex_pattern = regex_pattern.replace('X', '.')
                
                matches = list(re.finditer(regex_pattern, sequence))
                
                if matches:
                    # Take the match closest to typical position
                    typical_pos = sum(motif_info['typical_position']) / 2
                    best_match = min(matches, key=lambda m: abs(m.start() - typical_pos))
                    
                    motifs_found[motif_name] = {
                        'start': best_match.start(),
                        'end': best_match.end(),
                        'sequence': best_match.group(),
                        'confidence': 1.0 if len(matches) == 1 else 0.8
                    }
            
            # Special handling for GXSXG motif (most important)
            gxsxg_pattern = r'G[A-Z]S[A-Z]G'
            gxsxg_matches = list(re.finditer(gxsxg_pattern, sequence))
            
            if gxsxg_matches:
                # Usually around position 100-180
                best_gxsxg = min(gxsxg_matches, key=lambda m: abs(m.start() - 140))
                motifs_found['gxsxg_motif'] = {
                    'start': best_gxsxg.start(),
                    'end': best_gxsxg.end(),
                    'sequence': best_gxsxg.group(),
                    'confidence': 1.0
                }
            
            motif_data.append({
                'accession': accession,
                'motifs': json.dumps(motifs_found),
                'num_motifs': len(motifs_found),
                'has_gxsxg': 'gxsxg_motif' in motifs_found,
                'has_catalytic_triad': 'catalytic_serine' in motifs_found
            })
        
        motifs_df = pd.DataFrame(motif_data)
        
        # Save motif data
        motifs_df.to_csv(self.output_dir / "ab_hydrolases_motifs.csv", index=False)
        
        logger.info(f"Extracted motifs for {len(motifs_df)} sequences")
        logger.info(f"Sequences with GXSXG: {motifs_df['has_gxsxg'].sum()}")
        logger.info(f"Sequences with catalytic triad: {motifs_df['has_catalytic_triad'].sum()}")
        
        return motifs_df
    
    def download_structural_examples(self) -> pd.DataFrame:
        """Download PDB examples for structural validation"""
        
        logger.info("Downloading structural examples from PDB...")
        
        # Query PDB for alpha/beta hydrolases with good resolution
        pdb_query = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_polymer_entity.rcsb_ec_classification",
                            "operator": "contains_phrase",
                            "value": "3.1"  # Hydrolases acting on ester bonds
                        }
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_entry_info.resolution_combined",
                            "operator": "less",
                            "value": 2.5
                        }
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "entity_poly.rcsb_sample_sequence_length",
                            "operator": "range",
                            "value": {"from": 220, "to": 350}
                        }
                    }
                ]
            },
            "return_type": "polymer_entity",
            "request_options": {
                "results_content_type": ["experimental"],
                "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}],
                "results_verbosity": "verbose"
            }
        }
        
        # Query PDB
        response = requests.post(
            "https://search.rcsb.org/rcsbsearch/v2/query",
            json=pdb_query
        )
        
        structural_data = []
        
        if response.status_code == 200:
            results = response.json()
            
            for result in results.get('result_set', [])[:100]:  # Top 100 structures
                identifier = result['identifier']
                pdb_id = identifier.split('_')[0]
                
                structural_data.append({
                    'pdb_id': pdb_id,
                    'entity_id': identifier,
                    'resolution': result.get('score', 0)
                })
        
        structural_df = pd.DataFrame(structural_data)
        
        # Save structural data
        structural_df.to_csv(self.output_dir / "ab_hydrolases_structures.csv", index=False)
        
        logger.info(f"Found {len(structural_df)} structural examples")
        
        return structural_df
    
    def generate_training_prompts(
        self,
        annotations_df: pd.DataFrame,
        motifs_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate diverse training prompts for each sequence"""
        
        logger.info("Generating training prompts...")
        
        # Merge dataframes
        merged_df = annotations_df.merge(motifs_df, on='accession', how='inner')
        
        prompt_templates = [
            "Design a {organism} {enzyme_type} with {motif_desc} motif, approximately {length} amino acids",
            "Create a {localization} hydrolase enzyme containing {motif_desc}, active at pH {ph}",
            "Generate an alpha/beta hydrolase fold protein with {catalytic_mechanism}, length {length}aa",
            "Engineer a {enzyme_type} from {organism} with {motif_desc} active site",
            "Synthesize a {function} enzyme with alpha/beta hydrolase fold containing {motif_desc}",
            "Design a thermostable {enzyme_type} with canonical {motif_desc} motif",
            "Create a {localization} {enzyme_type} optimized for {substrate} hydrolysis",
            "Generate a {length}aa hydrolase with {motif_desc} and {secondary_features}"
        ]
        
        prompts_data = []
        
        for _, row in merged_df.iterrows():
            # Parse EC number for enzyme type
            ec = str(row.get('EC number', '3.1.-.-'))
            enzyme_type = self._get_enzyme_type(ec)
            
            # Parse organism
            organism = str(row.get('Organism', 'bacterial')).split('(')[0].strip()
            
            # Parse localization
            subcell = str(row.get('Subcellular location [CC]', 'cytoplasmic'))
            localization = self._parse_localization(subcell)
            
            # Get motif description
            has_gxsxg = row.get('has_gxsxg', False)
            motif_desc = "GXSXG" if has_gxsxg else "serine hydrolase"
            
            # Generate 3-5 prompts per sequence
            for template in np.random.choice(prompt_templates, size=min(3, len(prompt_templates)), replace=False):
                prompt = template.format(
                    organism=organism[:20],  # Truncate if too long
                    enzyme_type=enzyme_type,
                    motif_desc=motif_desc,
                    length=row.get('Length', 280),
                    localization=localization,
                    ph=np.random.choice(['7', '7.5', '8', 'neutral']),
                    catalytic_mechanism="serine nucleophile",
                    function=row.get('Protein names', 'hydrolase')[:30],
                    substrate=np.random.choice(['ester', 'lipid', 'peptide']),
                    secondary_features="oxyanion hole stabilization"
                )
                
                # Create DSL specification
                dsl_spec = self._create_dsl_spec(row)
                
                prompts_data.append({
                    'accession': row['accession'],
                    'prompt': prompt,
                    'dsl_spec': json.dumps(dsl_spec),
                    'enzyme_type': enzyme_type,
                    'has_complete_motifs': has_gxsxg
                })
        
        prompts_df = pd.DataFrame(prompts_data)
        
        # Save prompts
        prompts_df.to_csv(self.output_dir / "ab_hydrolases_prompts.csv", index=False)
        
        logger.info(f"Generated {len(prompts_df)} training prompts")
        
        return prompts_df
    
    def _get_enzyme_type(self, ec_number: str) -> str:
        """Map EC number to enzyme type"""
        
        ec_map = {
            '3.1.1': 'lipase',
            '3.1.2': 'thioesterase',
            '3.1.3': 'phosphatase',
            '3.1.4': 'phosphodiesterase',
            '3.1.6': 'sulfatase',
            '3.1.8': 'phosphotriesterase',
            '3.1': 'esterase'
        }
        
        # Match most specific EC number
        for ec_prefix in sorted(ec_map.keys(), key=len, reverse=True):
            if ec_number.startswith(ec_prefix):
                return ec_map[ec_prefix]
        
        return 'hydrolase'
    
    def _parse_localization(self, subcell_location: str) -> str:
        """Parse subcellular localization"""
        
        subcell_lower = subcell_location.lower()
        
        if 'secreted' in subcell_lower or 'extracellular' in subcell_lower:
            return 'secreted'
        elif 'membrane' in subcell_lower:
            return 'membrane-associated'
        elif 'periplasm' in subcell_lower:
            return 'periplasmic'
        elif 'mitochondri' in subcell_lower:
            return 'mitochondrial'
        else:
            return 'cytoplasmic'
    
    def _create_dsl_spec(self, row) -> Dict:
        """Create DSL specification from annotations"""
        
        # Parse motifs
        motifs_str = row.get('motifs', '{}')
        motifs = json.loads(motifs_str) if isinstance(motifs_str, str) else {}
        
        dsl = {
            'length': [220, min(int(row.get('Length', 280)) + 30, 350)],
            'motifs': [],
            'tags': [],
            'constraints': []
        }
        
        # Add GXSXG motif if present
        if 'gxsxg_motif' in motifs:
            motif_info = motifs['gxsxg_motif']
            dsl['motifs'].append({
                'name': 'nucleophile_elbow',
                'pattern': 'G X S X G',
                'window': [max(0, motif_info['start'] - 20), 
                          min(350, motif_info['end'] + 20)]
            })
        
        # Add catalytic triad if present
        if 'catalytic_serine' in motifs:
            motif_info = motifs['catalytic_serine']
            dsl['motifs'].append({
                'name': 'catalytic_triad',
                'pattern': 'S [DE] H',
                'window': [max(0, motif_info['start'] - 30),
                          min(350, motif_info['end'] + 30)]
            })
        
        # Add localization tags
        subcell = str(row.get('Subcellular location [CC]', ''))
        if 'secret' in subcell.lower():
            dsl['tags'].append('secreted')
            dsl['constraints'].append('signal_peptide')
        
        # Add enzyme class
        ec = str(row.get('EC number', ''))
        if ec and ec != 'nan':
            dsl['tags'].append(f"EC_{ec.split('.')[0]}")
        
        return dsl
    
    def validate_and_clean(
        self,
        sequences_df: pd.DataFrame,
        annotations_df: pd.DataFrame,
        motifs_df: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Validate and clean the collected data"""
        
        logger.info("Validating and cleaning data...")
        
        # Merge all dataframes
        merged = sequences_df.merge(
            annotations_df, on='accession', how='inner'
        ).merge(
            motifs_df, on='accession', how='inner'
        )
        
        initial_count = len(merged)
        
        # Filter criteria
        # 1. Must have at least one key motif
        merged = merged[merged['num_motifs'] > 0]
        logger.info(f"After motif filter: {len(merged)} sequences")
        
        # 2. Prefer sequences with GXSXG motif
        gxsxg_sequences = merged[merged['has_gxsxg'] == True]
        non_gxsxg = merged[merged['has_gxsxg'] == False].sample(
            n=min(len(gxsxg_sequences) // 4, len(merged[merged['has_gxsxg'] == False])),
            random_state=42
        )
        merged = pd.concat([gxsxg_sequences, non_gxsxg])
        logger.info(f"After GXSXG preference: {len(merged)} sequences")
        
        # 3. Remove sequences with too many unknown residues
        def has_valid_sequence(seq):
            unknown_count = seq.count('X') + seq.count('U') + seq.count('B') + seq.count('Z')
            return unknown_count < len(seq) * 0.05  # Less than 5% unknown
        
        merged = merged[merged['sequence'].apply(has_valid_sequence)]
        logger.info(f"After sequence quality filter: {len(merged)} sequences")
        
        # 4. Remove redundant sequences (>95% identity)
        # This is simplified - in production, use CD-HIT
        merged = merged.drop_duplicates(subset=['sequence'])
        logger.info(f"After redundancy removal: {len(merged)} sequences")
        
        # Split back into separate dataframes
        sequences_clean = merged[['accession', 'full_id', 'description', 'sequence', 'length']]
        annotations_clean = merged.drop(columns=['sequence', 'full_id', 'description'])
        motifs_clean = merged[['accession', 'motifs', 'num_motifs', 'has_gxsxg', 'has_catalytic_triad']]
        
        # Save cleaned data
        sequences_clean.to_csv(self.output_dir / "ab_hydrolases_sequences_clean.csv", index=False)
        annotations_clean.to_csv(self.output_dir / "ab_hydrolases_annotations_clean.csv", index=False)
        motifs_clean.to_csv(self.output_dir / "ab_hydrolases_motifs_clean.csv", index=False)
        
        logger.info(f"Final dataset: {len(sequences_clean)} sequences (filtered from {initial_count})")
        
        return {
            'sequences': sequences_clean,
            'annotations': annotations_clean,
            'motifs': motifs_clean
        }

def main():
    """Main execution function"""
    
    collector = ABHydrolaseCollector()
    
    # Collect complete dataset
    dataset = collector.collect_complete_dataset()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    
    for name, df in dataset.items():
        print(f"{name}: {len(df)} entries")
    
    # Additional statistics
    if 'motifs' in dataset:
        motifs_df = dataset['motifs']
        print(f"\nMotif Statistics:")
        print(f"  - Sequences with GXSXG: {motifs_df['has_gxsxg'].sum()} ({motifs_df['has_gxsxg'].mean()*100:.1f}%)")
        print(f"  - Sequences with catalytic triad: {motifs_df['has_catalytic_triad'].sum()} ({motifs_df['has_catalytic_triad'].mean()*100:.1f}%)")
    
    print("\nData saved to: data/raw/")
    print("Ready for training pipeline!")

if __name__ == "__main__":
    main()
