#!/usr/bin/env python3
"""
ingest_online.py
----------------
End-to-end data processor that:
  1) fetches protein sequences from UniProt REST
  2) builds lightweight annotations (CSV) from metadata/motifs
  3) runs the repo's ProteinDatasetBuilder to create parquet splits
  4) (optional) builds a FAISS retrieval index

Works out of the box for ~220–350 aa hydrolases, and can be customized via CLI flags.

Example:
  python scripts/ingest_online.py \
      --query "reviewed:true AND length:[220 TO 350] AND keyword:Hydrolase" \
      --out-root data \
      --build-retrieval

Requirements:
  pip install requests pandas pyarrow tqdm biopython
  (Your repo should already have pandas/pyarrow/biopython per requirements.txt)
"""

import io
import re
import sys
import json
import time
import gzip
import math
import shutil
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import requests
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO

# Import your internal builder/retriever
# Assumes your repo layout exposes these modules.
from data.preprocessing import ProteinDatasetBuilder
from data.retrieval import ExemplarRetriever

UNIPROT_STREAM = "https://rest.uniprot.org/uniprotkb/stream"

def setup_logger(verbosity:int=1)->logging.Logger:
    log = logging.getLogger("ingest_online")
    if not log.handlers:
        level = logging.INFO if verbosity>0 else logging.WARNING
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
    return logging.getLogger("ingest_online")


def fetch_uniprot_fasta(query: str, out_fasta: Path, batch_size: int = 500, decompress: bool = False, retries:int=3, sleep:float=1.5):
    """
    Stream FASTA from UniProt REST into a local file.

    Args:
      query: UniProt query string (e.g., 'reviewed:true AND keyword:Hydrolase')
      out_fasta: path to write .fasta
      batch_size: server-side paging hint (no strict guarantee)
      decompress: if UniProt returns compressed output (we request uncompressed)
    """
    params = {
        "compressed": "false",     # we want plain text FASTA
        "format": "fasta",
        "query": query
    }
    log = logging.getLogger("ingest_online")
    log.info(f"Fetching UniProt FASTA with query: {query}")
    with requests.get(UNIPROT_STREAM, params=params, stream=True, timeout=60) as r:
        if r.status_code != 200:
            raise RuntimeError(f"UniProt HTTP {r.status_code}: {r.text[:200]}")
        with open(out_fasta, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    n = sum(1 for _ in SeqIO.parse(str(out_fasta), "fasta"))
    log.info(f"Downloaded sequences: {n}")
    return n


def _extract_simple_annotations(fasta_path: Path) -> pd.DataFrame:
    """
    Build a minimal annotations dataframe from FASTA headers and simple heuristics.
    Columns:
      protein_id, motif, domain, localization, ec_class, function, properties
    """
    records = []
    # Heuristic motifs to seed labels (can expand as needed)
    # GxSxG is common serine hydrolase motif; HExH for metalloproteases; HxH etc.
    motif_candidates = [
        ("GXSXG", re.compile(r"G.S.G")),
        ("HExH",  re.compile(r"HE.H", re.IGNORECASE)),
        ("HxH",   re.compile(r"H.H", re.IGNORECASE)),
        ("CysHis",re.compile(r"C.{2,4}H", re.IGNORECASE)),
    ]

    for rec in SeqIO.parse(str(fasta_path), "fasta"):
        pid = rec.id
        desc = rec.description
        seq = str(rec.seq).upper()
        # simple localization guess from description
        loc = "cytoplasmic"
        if re.search(r"secreted|signal", desc, re.IGNORECASE):
            loc = "secreted"
        elif re.search(r"periplasm|extracellular", desc, re.IGNORECASE):
            loc = "extracellular"
        elif re.search(r"mitochond", desc, re.IGNORECASE):
            loc = "mitochondrial"

        ec = None
        m = re.search(r"EC=(\d+\.\d+\.\d+\.\d+)", desc)
        if m:
            ec = m.group(1)

        motif = None
        for motif_name, motif_rx in motif_candidates:
            if motif_rx.search(seq):
                motif = motif_name
                break

        records.append({
            "protein_id": pid,
            "motif": motif or "GXSXG",   # default to hydrolase motif
            "domain": "hydrolase",
            "localization": loc,
            "ec_class": ec or "hydrolase",
            "function": "catalytic",
            "properties": "stable at pH 7"
        })

    return pd.DataFrame(records)


def save_annotations_csv(df: pd.DataFrame, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


def prepare_datasets(fasta_path: Path, annotations_csv: Path, out_dir: Path,
                     min_len:int=220, max_len:int=350) -> Dict[str, str]:
    """
    Run the repo's ProteinDatasetBuilder → produce parquet splits.
    Returns dict of split → path.
    """
    builder = ProteinDatasetBuilder(min_length=min_len, max_length=max_len)
    paths = builder.build_dataset(
        fasta_file=str(fasta_path),
        annotation_file=str(annotations_csv),
        output_dir=str(out_dir)
    )
    return paths


def maybe_build_retrieval(train_parquet: Path, index_out: Path,
                          embedding_dim:int=1280, model_name:str="facebook/esm2_t33_650M_UR50D"):
    log = logging.getLogger("ingest_online")
    log.info("Building FAISS retrieval index from training set...")
    import pandas as pd
    df = pd.read_parquet(train_parquet)
    seqs = df["sequence"].dropna().astype(str).unique().tolist()
    retriever = ExemplarRetriever(embedding_dim=embedding_dim)
    # Let retriever choose HF model internally if not provided; otherwise compute with model_name
    try:
        retriever.build_index(seqs, save_path=str(index_out))
    except TypeError:
        # older signature without save_path string casting
        retriever.build_index(seqs, save_path=index_out)
    log.info(f"Saved retrieval index to: {index_out}")


def main():
    ap = argparse.ArgumentParser(description="Fetch sequences from UniProt and build training datasets.")
    ap.add_argument("--query", type=str,
                    default="reviewed:true AND length:[220 TO 350] AND keyword:Hydrolase",
                    help="UniProt query (see https://rest.uniprot.org)")
    ap.add_argument("--out-root", type=str, default="data", help="Root data dir")
    ap.add_argument("--min-len", type=int, default=220, help="Min sequence length")
    ap.add_argument("--max-len", type=int, default=350, help="Max sequence length")
    ap.add_argument("--build-retrieval", action="store_true", help="Also build FAISS retrieval index")
    ap.add_argument("--verbosity", type=int, default=1, help="0=warn, 1=info")
    args = ap.parse_args()

    log = setup_logger(args.verbosity)

    root = Path(args.out_root)
    raw_dir = root / "raw"
    proc_dir = root / "processed"
    ensure = lambda p: p.mkdir(parents=True, exist_ok=True)

    ensure(raw_dir); ensure(proc_dir)

    fasta_path = raw_dir / "uniprot_query.fasta"
    ann_csv = raw_dir / "annotations.csv"

    # 1) Download sequences
    n = fetch_uniprot_fasta(args.query, fasta_path)
    if n == 0:
        log.warning("No sequences downloaded from UniProt; exiting.")
        sys.exit(1)

    # 2) Build very simple annotations
    ann_df = _extract_simple_annotations(fasta_path)
    save_annotations_csv(ann_df, ann_csv)
    log.info(f"Annotations CSV written to {ann_csv} ({len(ann_df)} rows)")

    # 3) Build parquet datasets
    paths = prepare_datasets(fasta_path, ann_csv, proc_dir,
                             min_len=args.min_len, max_len=args.max_len)
    log.info(f"Dataset splits: {json.dumps(paths, indent=2)}")

    # 4) Optional retrieval index
    if args.build_retrieval:
        index_out = proc_dir / "retrieval_index"
        maybe_build_retrieval(Path(paths["train"]), index_out)

    log.info("Ingestion complete.")
    print(json.dumps(paths))
    

if __name__ == "__main__":
    main()
