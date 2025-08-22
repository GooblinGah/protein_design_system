from torch.utils.data import DataLoader
from typing import Optional
from .dataset import ProteinDesignDataset

def get_protein_dataloader(
    dataset: ProteinDesignDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False
) -> DataLoader:
    """
    Create a DataLoader for protein design data with proper collation.
    
    Args:
        dataset: ProteinDesignDataset instance
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU training
        drop_last: Whether to drop incomplete batches
        
    Returns:
        Configured DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=dataset.collate_fn
    )

def get_protein_dataloaders(
    train_dataset: ProteinDesignDataset,
    val_dataset: Optional[ProteinDesignDataset] = None,
    test_dataset: Optional[ProteinDesignDataset] = None,
    batch_size: int = 32,
    val_batch_size: Optional[int] = None,
    test_batch_size: Optional[int] = None,
    num_workers: int = 4,
    pin_memory: bool = True
):
    """
    Create multiple DataLoaders for train/val/test splits.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        test_dataset: Test dataset (optional)
        batch_size: Training batch size
        val_batch_size: Validation batch size (defaults to batch_size)
        test_batch_size: Test batch size (defaults to batch_size)
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU training
        
    Returns:
        Dictionary of DataLoaders
    """
    val_batch_size = val_batch_size or batch_size
    test_batch_size = test_batch_size or batch_size
    
    loaders = {
        'train': get_protein_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    }
    
    if val_dataset:
        loaders['val'] = get_protein_dataloader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    if test_dataset:
        loaders['test'] = get_protein_dataloader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    return loaders
