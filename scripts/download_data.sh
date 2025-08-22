#!/bin/bash
# Script to download required data

set -euo pipefail

DATA_DIR="data/raw"
mkdir -p $DATA_DIR

echo "Downloading Swiss-Prot hydrolases..."
# Example - adjust to actual data source
wget -O $DATA_DIR/swissprot_hydrolases.fasta \
    "https://www.uniprot.org/uniprot/?query=family:hydrolase&format=fasta"

echo "Downloading Pfam annotations..."
wget -O $DATA_DIR/pfam_annotations.csv \
    "https://pfam.xfam.org/family/PF00561/alignment/seed/format?format=csv"

echo "Downloading motif patterns from PROSITE..."
wget -O $DATA_DIR/prosite_motifs.dat \
    "ftp://ftp.expasy.org/databases/prosite/prosite.dat"

echo "Installing required bioinformatics tools..."
conda install -y -c bioconda cd-hit=4.8.1
conda install -y -c bioconda hmmer=3.3.2
conda install -y -c bioconda muscle=3.8.31

echo "Data download complete!"
