"""
Embedding Management for Insurance Classification
Handles sentence transformer embeddings and caching
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages sentence transformer embeddings with caching capabilities
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', cache_dir: str = 'data/cache'):
        """
        Initialize embedding manager
        
        Args:
            model_name: SentenceTransformer model name
            cache_dir: Directory to cache embeddings
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize sentence transformer
        print(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Embedding caches
        self.label_embeddings = None
        self.label_cache_path = self.cache_dir / f'label_embeddings_{model_name.replace("/", "_")}.pkl'
        
    def encode_texts(self, texts: List[str], convert_to_tensor: bool = True) -> torch.Tensor:
        """
        Encode list of texts to embeddings
        
        Args:
            texts: List of texts to encode
            convert_to_tensor: Whether to return torch tensor
            
        Returns:
            Embeddings tensor or numpy array
        """
        return self.model.encode(texts, convert_to_tensor=convert_to_tensor)
    
    def encode_single_text(self, text: str, convert_to_tensor: bool = True) -> torch.Tensor:
        """
        Encode single text to embedding
        
        Args:
            text: Text to encode
            convert_to_tensor: Whether to return torch tensor
            
        Returns:
            Embedding tensor or numpy array
        """
        return self.model.encode(text, convert_to_tensor=convert_to_tensor)
    
    def compute_label_embeddings(self, labels: List[str], force_recompute: bool = False) -> torch.Tensor:
        """
        Compute and cache embeddings for insurance labels
        
        Args:
            labels: List of insurance labels
            force_recompute: Whether to force recomputation
            
        Returns:
            Label embeddings tensor
        """
        if not force_recompute and self.label_cache_path.exists():
            try:
                print("Loading cached label embeddings...")
                with open(self.label_cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    
                # Verify cache is for same labels
                if cached_data['labels'] == labels:
                    self.label_embeddings = torch.tensor(cached_data['embeddings'])
                    print(f"✅ Loaded {len(labels)} cached label embeddings")
                    return self.label_embeddings
                else:
                    print("Label list changed, recomputing embeddings...")
            except Exception as e:
                logger.warning(f"Failed to load cached embeddings: {e}")
        
        # Compute new embeddings
        print(f"Computing embeddings for {len(labels)} labels...")
        self.label_embeddings = self.model.encode(labels, convert_to_tensor=True)
        
        # Cache the embeddings
        try:
            cache_data = {
                'labels': labels,
                'embeddings': self.label_embeddings.cpu().numpy(),
                'model_name': self.model_name
            }
            with open(self.label_cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            print("✅ Cached label embeddings for future use")
        except Exception as e:
            logger.warning(f"Failed to cache embeddings: {e}")
            
        return self.label_embeddings
    
    def get_label_embeddings(self) -> Optional[torch.Tensor]:
        """Get cached label embeddings"""
        return self.label_embeddings
    
    def cosine_similarity(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between embeddings
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            
        Returns:
            Similarity scores
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Convert to numpy if needed
        if isinstance(embeddings1, torch.Tensor):
            embeddings1 = embeddings1.cpu().numpy()
        if isinstance(embeddings2, torch.Tensor):
            embeddings2 = embeddings2.cpu().numpy()
            
        # Ensure proper shape
        if embeddings1.ndim == 1:
            embeddings1 = embeddings1.reshape(1, -1)
        if embeddings2.ndim == 1:
            embeddings2 = embeddings2.reshape(1, -1)
            
        return cosine_similarity(embeddings1, embeddings2)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings from this model"""
        return self.model.get_sentence_embedding_dimension()
    
    def clear_cache(self):
        """Clear all cached embeddings"""
        try:
            if self.label_cache_path.exists():
                self.label_cache_path.unlink()
            print("✅ Cleared embedding cache")
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}") 