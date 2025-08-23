import faiss
import numpy as np
import torch
import pickle
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ExemplarRetriever:
    """Retrieves similar sequences for pointer mechanism"""
    
    def __init__(
        self,
        embedding_dim: int = 1280,
        index_type: str = 'flat',
        device: str = 'cpu',
        encoder_model: Optional[torch.nn.Module] = None,
        encoder_tokenizer: Optional[object] = None,
        model_name: Optional[str] = None
    ):
        self.embedding_dim = embedding_dim
        self.device = device
        self.encoder_model = encoder_model
        self.encoder_tokenizer = encoder_tokenizer
        self.model_name = model_name or "facebook/esm2_t33_650M_UR50D"
        
        # Initialize FAISS index
        if index_type == 'flat':
            self.index = faiss.IndexFlatL2(embedding_dim)
        else:
            # Can add other index types
            self.index = faiss.IndexFlatL2(embedding_dim)
        
        self.sequences = []
        self.metadata = []
        
    def build_index(
        self,
        sequences: List[str],
        encoder_model: Optional[torch.nn.Module] = None,
        save_path: Optional[str] = None
    ):
        """Build index from sequences"""
        
        logger.info(f"Building index for {len(sequences)} sequences...")
        
        # Get embeddings
        if encoder_model is None:
            # Use ESM or other pretrained model
            from transformers import AutoModel, AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            encoder = AutoModel.from_pretrained(self.model_name)
            
            embeddings = self._compute_embeddings(sequences, encoder, tokenizer)
        else:
            embeddings = self._compute_embeddings_custom(sequences, encoder_model)
        
        # Add to index
        self.index.add(embeddings)
        self.sequences = sequences
        
        # Save if path provided
        if save_path:
            self.save_index(save_path)
    
    def _compute_embeddings(self, sequences, model, tokenizer):
        """Compute embeddings using pretrained model"""
        embeddings = []
        
        batch_size = 32
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=350
            )
            
            with torch.no_grad():
                outputs = model(**inputs)
                # Use CLS token or mean pooling
                batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def _compute_embeddings_custom(self, sequences, model):
        """Compute embeddings using custom model"""
        # Implementation for custom encoder
        raise NotImplementedError(
            "Custom encoder not implemented. Use the default ESM2 model or implement "
            "this method to handle your custom encoder architecture."
        )
    
    def retrieve_exemplars(
        self,
        query: str,
        k: int = 10
    ) -> Tuple[List[str], np.ndarray]:
        """Retrieve k most similar sequences"""
        
        # Compute query embedding
        query_embedding = self._compute_query_embedding(query)
        
        # Search
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        
        # Get sequences
        retrieved_sequences = [self.sequences[idx] for idx in indices[0]]
        
        return retrieved_sequences, distances[0]
    
    def _compute_query_embedding(self, query: str) -> np.ndarray:
        """Compute embedding for query sequence"""
        if self.encoder_model is None:
            from transformers import AutoModel, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(self.model_name)
            mdl = AutoModel.from_pretrained(self.model_name)
        else:
            tok, mdl = self.encoder_tokenizer, self.encoder_model
        
        with torch.no_grad():
            inputs = tok([query], return_tensors="pt", padding=True, truncation=True, max_length=350)
            out = mdl(**inputs).last_hidden_state[:, 0, :].cpu().numpy()
        return out[0]
    
    def save_index(self, path: str):
        """Save index and metadata"""
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Save sequences and metadata
        with open(path / "sequences.pkl", 'wb') as f:
            pickle.dump({
                'sequences': self.sequences,
                'metadata': self.metadata,
                'model_name': self.model_name
            }, f)
    
    def load_index(self, path: str):
        """Load saved index"""
        path = Path(path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(path / "index.faiss"))
        
        # Load sequences
        with open(path / "sequences.pkl", 'rb') as f:
            data = pickle.load(f)
            self.sequences = data['sequences']
            self.metadata = data['metadata']
            self.model_name = data.get('model_name', self.model_name)
