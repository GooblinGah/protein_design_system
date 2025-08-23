#!/bin/bash
# Script to download required data

set -euo pipefail

DATA_DIR="data/raw"
mkdir -p $DATA_DIR

echo "Downloading Swiss-Prot hydrolases..."
wget -L -O "$DATA_DIR/swissprot_hydrolases.fasta" \
  "https://rest.uniprot.org/uniprotkb/stream?compressed=false&format=fasta&query=reviewed:true+AND+keyword:Hydrolase"

echo "Downloading Pfam PF00561 seed alignment..."
wget -L -O "$DATA_DIR/pfam_PF00561_seed.sto" \
  "https://pfam.xfam.org/family/PF00561/alignment/seed/format?format=stockholm"

echo "Downloading motif patterns from PROSITE..."
wget -O $DATA_DIR/prosite_motifs.dat \
    "ftp://ftp.expasy.org/databases/prosite/prosite.dat"

echo "Installing required bioinformatics tools..."
conda install -y -c bioconda cd-hit=4.8.1
conda install -y -c bioconda hmmer=3.3.2
conda install -y -c bioconda muscle=3.8.31

echo "Data download complete!"
