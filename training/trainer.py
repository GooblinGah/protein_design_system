#!/usr/bin/env python3
"""Production trainer for protein design system."""

import os
import math
import json
import yaml
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm

# Repository imports
try:
    from models.decoder.pointer_generator import PointerGeneratorDecoder
except ImportError:
    from models.decoder import PointerGeneratorDecoder

from models.tokenizer import ProteinTokenizer
from data import ProteinDesignDataset, ExemplarRetriever


@dataclass
class TrainerConfig:
    # Data
    train_path: str = "data/processed/train.parquet"
    val_path: str = "data/processed/val.parquet"
    test_path: str = "data/processed/test.parquet"
    max_length: int = 350
    batch_size: int = 4
    num_workers: int = 0

    # Retrieval
    use_retrieval: bool = False
    retrieval_index_path: str = "data/processed/retrieval_index"
    embedding_model: str = "facebook/esm2_t33_650M_UR50D"
    embedding_dim: int = 1280

    # Model
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    vocab_size: Optional[int] = None  # if None, take from tokenizer

    # Optimizer
    epochs: int = 2
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    warmup_steps: int = 200
    grad_clip_norm: float = 1.0

    # Schedules
    cosine_min_lr_ratio: float = 0.01  # eta_min = lr * ratio

    # Checkpointing / logs
    output_dir: str = "runs/experiment_1"
    save_every_steps: int = 500
    eval_every_steps: int = 500
    keep_last_n: int = 5

    # Misc
    seed: int = 42


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(p: Path):
    """Create directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)


class ProteinDesignTrainer:
    """Complete trainer for protein design models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.cfg = TrainerConfig(**config)
        set_seed(self.cfg.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # --- Tokenizer ---
        self.tokenizer = ProteinTokenizer()
        self.pad_id = self.tokenizer.pad_id
        self.bos_id = self.tokenizer.bos_id
        self.eos_id = self.tokenizer.eos_id
        print(f"Tokenizer: vocab_size={self.tokenizer.vocab_size}, pad_id={self.pad_id}")

        # --- Model ---
        vocab_size = self.cfg.vocab_size or self.tokenizer.vocab_size
        print(f"Building model: d_model={self.cfg.d_model}, n_layers={self.cfg.n_layers}")

        self.model = PointerGeneratorDecoder(
            vocab_size=vocab_size,
            d_model=self.cfg.d_model,
            n_heads=self.cfg.n_heads,
            n_layers=self.cfg.n_layers,
            d_ff=self.cfg.d_ff,
            dropout=self.cfg.dropout,
            max_seq_length=self.cfg.max_length,  # Note: your decoder expects max_seq_length
        ).to(self.device)

        # --- Retrieval ---
        self.retriever = None
        if self.cfg.use_retrieval:
            print(f"Loading retrieval index: {self.cfg.retrieval_index_path}")
            self.retriever = ExemplarRetriever(
                embedding_dim=self.cfg.embedding_dim,
                model_name=self.cfg.embedding_model,
            )
            self.retriever.load_index(self.cfg.retrieval_index_path)

        # --- Data ---
        print(f"Loading datasets: {self.cfg.train_path}, {self.cfg.val_path}")
        self.train_ds = ProteinDesignDataset(
            data_path=self.cfg.train_path,
            tokenizer=self.tokenizer,
            retriever=self.retriever,
            max_length=self.cfg.max_length,
        )
        self.val_ds = ProteinDesignDataset(
            data_path=self.cfg.val_path,
            tokenizer=self.tokenizer,
            retriever=self.retriever,
            max_length=self.cfg.max_length,
        )

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=self.train_ds.collate_fn,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            collate_fn=self.val_ds.collate_fn,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

        print(f"Train: {len(self.train_ds)} samples, Val: {len(self.val_ds)} samples")
        print(f"Batch size: {self.cfg.batch_size}, Steps per epoch: {len(self.train_loader)}")

        # --- Optimizer & Scheduler ---
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            betas=self.cfg.betas,
            eps=self.cfg.eps
        )

        # Scheduler is built after we know steps_per_epoch
        self.scheduler = None

        # --- IO ---
        self.out_dir = Path(self.cfg.output_dir)
        self.ckpt_dir = self.out_dir / "checkpoints"
        self.log_path = self.out_dir / "training_log.jsonl"
        ensure_dir(self.out_dir)
        ensure_dir(self.ckpt_dir)

        self.global_step = 0
        self.best_val = float("inf")
        
        print(f"Output directory: {self.out_dir}")

    def adapt_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Map dataset batch to model inputs."""
        x = batch["sequences"].to(self.device)                 # [B,T]
        attn = batch.get("attention_mask")
        if attn is not None:
            attn = attn.to(self.device)

        # Teacher-forcing targets: next-token prediction
        # Shift left: target[t] = x[t+1], last becomes PAD
        target = x.clone()
        target[:, :-1] = x[:, 1:]
        target[:, -1] = self.pad_id

        # Handle exemplars
        ex = batch.get("exemplars")
        if isinstance(ex, dict) and "tokens" in ex:
            ex_tokens = ex["tokens"].to(self.device)  # [B,K,L]
            ex_dists = ex["distances"].to(self.device)
            exemplars = {"tokens": ex_tokens, "distances": ex_dists}
        else:
            exemplars = None

        return {
            "input_ids": x,
            "target_ids": target,
            "attention_mask": attn,
            "exemplars": exemplars,
            "dsl_specs": batch.get("dsl_specs"),
            "prompts": batch.get("prompts"),
        }

    def compute_loss(self, outputs: Dict[str, torch.Tensor], target_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute loss from model outputs.
        
        Prefer logits over probabilities. If only p_mixture is provided,
        switch to NLL on log probabilities.
        """
        if "logits_vocab" in outputs:
            logits = outputs["logits_vocab"]  # [B,T,V]
            ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=self.pad_id,
            )
            return ce
        elif "logits" in outputs:
            logits = outputs["logits"]
            ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=self.pad_id,
            )
            return ce
        elif "p_mixture" in outputs:
            logp = torch.log(outputs["p_mixture"].clamp_min(1e-12))
            ce = F.nll_loss(
                logp.view(-1, logp.size(-1)),
                target_ids.view(-1),
                ignore_index=self.pad_id,
            )
            return ce
        else:
            raise KeyError("Decoder outputs missing 'logits_vocab', 'logits', or 'p_mixture'.")

    def build_scheduler(self, steps_per_epoch: int):
        """Build learning rate scheduler with warmup + cosine annealing."""
        warmup = self.cfg.warmup_steps
        total = max(1, steps_per_epoch * self.cfg.epochs)
        main_steps = max(1, total - warmup)
        
        warm = LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup)
        cos = CosineAnnealingLR(
            self.optimizer,
            T_max=main_steps,
            eta_min=self.cfg.lr * self.cfg.cosine_min_lr_ratio
        )
        self.scheduler = SequentialLR(self.optimizer, [warm, cos], milestones=[warmup])
        
        print(f"LR Schedule: {warmup} warmup steps, {main_steps} cosine steps")

    def train_epoch(self, epoch_idx: int):
        """Train for one epoch."""
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_idx+1}/{self.cfg.epochs}")
        running_loss = 0.0

        for batch in pbar:
            self.global_step += 1
            batch = self.adapt_batch(batch)

            # Forward pass
            outputs = self.model(
                sequences=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                exemplars=batch["exemplars"],
                dsl_specs=batch["dsl_specs"],
            )

            # Compute loss
            loss = self.compute_loss(outputs, batch["target_ids"])

            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.cfg.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            # Update running loss and progress bar
            running_loss = 0.95 * running_loss + 0.05 * loss.item() if self.global_step > 1 else loss.item()
            lr = self.optimizer.param_groups[0]["lr"]
            pbar.set_postfix({"loss": f"{running_loss:.4f}", "lr": f"{lr:.2e}"})

            # Logging
            with open(self.log_path, "a") as f:
                f.write(json.dumps({
                    "step": self.global_step, 
                    "epoch": epoch_idx + 1,
                    "loss": loss.item(), 
                    "lr": lr,
                    "phase": "train"
                }) + "\n")

            # Periodic evaluation & checkpointing
            if self.global_step % self.cfg.eval_every_steps == 0:
                val_loss = self.evaluate()
                is_best = val_loss < self.best_val
                if is_best:
                    self.best_val = val_loss
                self.save_checkpoint(is_best=is_best, tag=f"step{self.global_step}")

            if self.global_step % self.cfg.save_every_steps == 0:
                self.save_checkpoint(is_best=False, tag=f"step{self.global_step}")

        return running_loss

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        losses = []
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            batch = self.adapt_batch(batch)
            outputs = self.model(
                sequences=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                exemplars=batch["exemplars"],
                dsl_specs=batch["dsl_specs"],
            )
            loss = self.compute_loss(outputs, batch["target_ids"])
            losses.append(loss.item())
        
        val_loss = float(sum(losses) / max(1, len(losses)))
        
        # Log validation loss
        with open(self.log_path, "a") as f:
            f.write(json.dumps({
                "step": self.global_step, 
                "val_loss": val_loss,
                "phase": "validation"
            }) + "\n")
        
        print(f"Validation Loss: {val_loss:.4f}")
        return val_loss

    def save_checkpoint(self, is_best: bool, tag: Optional[str] = None):
        """Save model checkpoint."""
        payload = {
            "config": asdict(self.cfg),
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": (self.scheduler.state_dict() if self.scheduler else None),
            "global_step": self.global_step,
            "best_val": self.best_val,
            "pad_id": self.pad_id,
            "tokenizer_info": {
                "vocab_size": self.tokenizer.vocab_size,
                "pad_id": self.tokenizer.pad_id,
                "bos_id": self.tokenizer.bos_id,
                "eos_id": self.tokenizer.eos_id,
            }
        }
        
        ensure_dir(self.ckpt_dir)
        name = "latest.pt" if tag is None else f"ckpt_{tag}.pt"
        path = self.ckpt_dir / name
        torch.save(payload, path)

        if is_best:
            torch.save(payload, self.ckpt_dir / "best.pt")
            print(f"New best model saved: {val_loss:.4f}")

        # Prune old checkpoints
        ckpts = sorted(
            self.ckpt_dir.glob("ckpt_step*.pt"),
            key=lambda p: int(p.stem.split("step")[-1])
        )
        if len(ckpts) > self.cfg.keep_last_n:
            for old in ckpts[:len(ckpts) - self.cfg.keep_last_n]:
                try:
                    old.unlink()
                except Exception:
                    pass

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        print(f"Loading checkpoint: {path}")
        payload = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        if payload.get("scheduler_state") and self.scheduler:
            self.scheduler.load_state_dict(payload["scheduler_state"])
        
        self.global_step = int(payload.get("global_step", 0))
        self.best_val = float(payload.get("best_val", float("inf")))
        
        # Verify tokenizer compatibility
        if "tokenizer_info" in payload:
            tokenizer_info = payload["tokenizer_info"]
            if (tokenizer_info["vocab_size"] != self.tokenizer.vocab_size or
                tokenizer_info["pad_id"] != self.tokenizer.pad_id):
                print(f"Warning: Tokenizer mismatch in checkpoint")
                print(f"   Checkpoint: vocab_size={tokenizer_info['vocab_size']}, pad_id={tokenizer_info['pad_id']}")
                print(f"   Current: vocab_size={self.tokenizer.vocab_size}, pad_id={self.tokenizer.pad_id}")

    def fit(self, resume_ckpt: Optional[str] = None):
        """Main training loop."""
        print(f"Starting training for {self.cfg.epochs} epochs")
        print(f"Total steps: {len(self.train_loader) * self.cfg.epochs}")
        
        if resume_ckpt:
            # Build scheduler after loaders to have correct step counts
            if self.scheduler is None:
                self.build_scheduler(len(self.train_loader))
            self.load_checkpoint(resume_ckpt)

        if self.scheduler is None:
            self.build_scheduler(len(self.train_loader))

        # Training loop
        for epoch in range(self.cfg.epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{self.cfg.epochs}")
            print(f"{'='*50}")
            
            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate()
            
            is_best = val_loss < self.best_val
            if is_best:
                self.best_val = val_loss
                print(f"New best validation loss: {val_loss:.4f}")
            
            self.save_checkpoint(is_best=is_best, tag=f"epoch{epoch+1}")
            
            print(f"Epoch {epoch+1} complete - Train: {train_loss:.4f}, Val: {val_loss:.4f}")

        # Save final checkpoint
        self.save_checkpoint(is_best=False, tag="final")
        print(f"\nTraining complete! Best validation loss: {self.best_val:.4f}")


def main():
    """Command-line interface."""
    import argparse
    ap = argparse.ArgumentParser(description="Train protein design model")
    ap.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    ap.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    args = ap.parse_args()

    print(f"Loading config: {args.config}")
    cfg_file = load_yaml_config(args.config)

    # Merge top-level sections commonly used in your repo's config.yaml
    merged = {}
    
    # Data section
    if "data" in cfg_file:
        merged.update({
            "train_path": cfg_file["data"].get("train_path", TrainerConfig.train_path),
            "val_path": cfg_file["data"].get("val_path", TrainerConfig.val_path),
            "test_path": cfg_file["data"].get("test_path", TrainerConfig.test_path),
            "max_length": cfg_file["data"].get("max_length", TrainerConfig.max_length),
            "batch_size": cfg_file.get("training", {}).get("batch_size", TrainerConfig.batch_size),
            "num_workers": cfg_file.get("training", {}).get("num_workers", TrainerConfig.num_workers),
        })
    
    # Model section
    if "model" in cfg_file:
        merged.update({
            "d_model": cfg_file["model"].get("d_model", TrainerConfig.d_model),
            "n_heads": cfg_file["model"].get("n_heads", TrainerConfig.n_heads),
            "n_layers": cfg_file["model"].get("n_layers", TrainerConfig.n_layers),
            "d_ff": cfg_file["model"].get("d_ff", TrainerConfig.d_ff),
            "dropout": cfg_file["model"].get("dropout", TrainerConfig.dropout),
        })
    
    # Retrieval section
    if "retrieval" in cfg_file:
        merged.update({
            "use_retrieval": cfg_file["retrieval"].get("use_retrieval", TrainerConfig.use_retrieval),
            "retrieval_index_path": cfg_file["retrieval"].get("index_path", TrainerConfig.retrieval_index_path),
            "embedding_model": cfg_file["retrieval"].get("embedding_model", TrainerConfig.embedding_model),
            "embedding_dim": cfg_file["retrieval"].get("embedding_dim", TrainerConfig.embedding_dim),
        })
    
    # Training section
    if "training" in cfg_file:
        merged.update({
            "epochs": cfg_file["training"].get("epochs", TrainerConfig.epochs),
            "lr": cfg_file["training"].get("lr", TrainerConfig.lr),
            "weight_decay": cfg_file["training"].get("weight_decay", TrainerConfig.weight_decay),
            "warmup_steps": cfg_file["training"].get("warmup_steps", TrainerConfig.warmup_steps),
            "grad_clip_norm": cfg_file["training"].get("grad_clip_norm", TrainerConfig.grad_clip_norm),
            "output_dir": cfg_file.get("output_dir", TrainerConfig.output_dir),
            "save_every_steps": cfg_file["training"].get("save_every_steps", TrainerConfig.save_every_steps),
            "eval_every_steps": cfg_file["training"].get("eval_every_steps", TrainerConfig.eval_every_steps),
            "keep_last_n": cfg_file["training"].get("keep_last_n", TrainerConfig.keep_last_n),
        })

    print(f"Merged config: {merged}")
    
    trainer = ProteinDesignTrainer(merged)
    trainer.fit(resume_ckpt=args.resume)


if __name__ == "__main__":
    main()
