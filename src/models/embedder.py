"""
Embedding model for generating sentence transformer embeddings
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class EmbeddingModel:
    """Handles text embedding generation using sentence transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            NumPy array of embeddings
        """
        if not self.model:
            raise ValueError("Model not loaded")
            
        if not texts:
            return np.array([])
            
        try:
            # Filter out empty texts
            valid_texts = [text for text in texts if text and text.strip()]
            
            if not valid_texts:
                logger.warning("No valid texts to embed")
                return np.array([])
                
            # Generate embeddings
            embeddings = self.model.encode(valid_texts, convert_to_numpy=True)
            
            logger.info(f"Generated embeddings for {len(valid_texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def embed_single_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Text string to embed
            
        Returns:
            NumPy array of embedding
        """
        if not text or not text.strip():
            return np.array([])
            
        embeddings = self.embed_texts([text])
        return embeddings[0] if len(embeddings) > 0 else np.array([])
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        try:
            # Calculate cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0
    
    def find_most_similar(self, query_embedding: np.ndarray, 
                         candidate_embeddings: np.ndarray, 
                         top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Find most similar embeddings to a query embedding
        
        Args:
            query_embedding: Query embedding to match against
            candidate_embeddings: Array of candidate embeddings
            top_k: Number of top matches to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        if len(candidate_embeddings) == 0:
            return []
            
        try:
            # Calculate similarities with all candidates
            similarities = []
            for i, candidate_embedding in enumerate(candidate_embeddings):
                similarity = self.calculate_similarity(query_embedding, candidate_embedding)
                similarities.append((i, similarity))
            
            # Sort by similarity score (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top k results
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Failed to find most similar embeddings: {e}")
            return []
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        if not self.model:
            return {"model_name": self.model_name, "loaded": False}
            
        try:
            # Get embedding dimension
            test_embedding = self.embed_single_text("test")
            embedding_dim = len(test_embedding) if len(test_embedding) > 0 else 0
            
            return {
                "model_name": self.model_name,
                "loaded": True,
                "embedding_dimension": embedding_dim,
                "max_sequence_length": getattr(self.model, 'max_seq_length', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"model_name": self.model_name, "loaded": False, "error": str(e)} 