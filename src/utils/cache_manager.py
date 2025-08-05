"""
Cache management utilities for storing and retrieving processed data
"""

import pickle
import os
from pathlib import Path
from typing import Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    """Handles caching of embeddings and processed data"""
    
    @staticmethod
    def save_to_cache(data: Any, cache_path: Path) -> bool:
        """
        Save data to cache file
        
        Args:
            data: Data to cache
            cache_path: Path to cache file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save data using pickle
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
                
            logger.info(f"Data cached successfully at {cache_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache data at {cache_path}: {e}")
            return False
    
    @staticmethod
    def load_from_cache(cache_path: Path) -> Optional[Any]:
        """
        Load data from cache file
        
        Args:
            cache_path: Path to cache file
            
        Returns:
            Cached data if successful, None otherwise
        """
        try:
            if not cache_path.exists():
                logger.info(f"Cache file not found: {cache_path}")
                return None
                
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                
            logger.info(f"Data loaded from cache: {cache_path}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load cache from {cache_path}: {e}")
            return None
    
    @staticmethod
    def cache_exists(cache_path: Path) -> bool:
        """
        Check if cache file exists
        
        Args:
            cache_path: Path to cache file
            
        Returns:
            True if cache exists, False otherwise
        """
        return cache_path.exists()
    
    @staticmethod
    def clear_cache(cache_path: Path) -> bool:
        """
        Clear/delete cache file
        
        Args:
            cache_path: Path to cache file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if cache_path.exists():
                os.remove(cache_path)
                logger.info(f"Cache cleared: {cache_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear cache {cache_path}: {e}")
            return False
    
    @staticmethod
    def get_cache_size(cache_path: Path) -> int:
        """
        Get size of cache file in bytes
        
        Args:
            cache_path: Path to cache file
            
        Returns:
            Size in bytes, -1 if error
        """
        try:
            if cache_path.exists():
                return cache_path.stat().st_size
            return 0
            
        except Exception as e:
            logger.error(f"Failed to get cache size for {cache_path}: {e}")
            return -1
    
    @staticmethod
    def save_embeddings(embeddings: np.ndarray, labels: list, cache_path: Path) -> bool:
        """
        Save embeddings with their corresponding labels
        
        Args:
            embeddings: Numpy array of embeddings
            labels: List of labels corresponding to embeddings
            cache_path: Path to cache file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cache_data = {
                'embeddings': embeddings,
                'labels': labels,
                'embedding_shape': embeddings.shape,
                'num_labels': len(labels)
            }
            
            return CacheManager.save_to_cache(cache_data, cache_path)
            
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
            return False
    
    @staticmethod
    def load_embeddings(cache_path: Path) -> tuple:
        """
        Load embeddings and labels from cache
        
        Args:
            cache_path: Path to cache file
            
        Returns:
            Tuple of (embeddings, labels) or (None, None) if failed
        """
        try:
            cache_data = CacheManager.load_from_cache(cache_path)
            
            if cache_data is None:
                return None, None
                
            embeddings = cache_data.get('embeddings')
            labels = cache_data.get('labels')
            
            if embeddings is None or labels is None:
                logger.error("Invalid cache data structure")
                return None, None
                
            logger.info(f"Loaded {len(labels)} embeddings from cache")
            return embeddings, labels
            
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            return None, None 