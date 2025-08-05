"""
Embedding model management for company classification.
"""

import numpy as np
import pandas as pd
import pickle
import hashlib
import os
from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    """Manages embedding models and generation."""
    
    def __init__(self, models_config, device, use_cache=False, cache_dir=None):
        self.models_config = models_config
        self.device = device
        self.loaded_models = {}
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        
        # Ensure cache directory exists if caching is enabled
        if self.use_cache and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
    def _generate_cache_key(self, df, text_column, model_key):
        """Generate a unique cache key based on dataframe content and model."""
        # Create hash from text content + model key
        texts = df[text_column].fillna('').astype(str).tolist()
        content_str = '|'.join(texts) + f'_model_{model_key}'
        return hashlib.md5(content_str.encode()).hexdigest()[:16]
    
    def _get_cache_path(self, cache_key, model_key):
        """Get the full cache file path."""
        if not self.cache_dir:
            return None
        filename = f"embeddings_{model_key}_{cache_key}.pkl"
        return os.path.join(self.cache_dir, filename)
    
    def _save_embeddings_to_cache(self, embeddings, cache_path):
        """Save embeddings to cache file."""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embeddings, f)
            print(f"    💾 Cached embeddings to: {os.path.basename(cache_path)}")
        except Exception as e:
            print(f"    Warning: Failed to cache embeddings: {e}")
    
    def _load_embeddings_from_cache(self, cache_path):
        """Load embeddings from cache file."""
        try:
            with open(cache_path, 'rb') as f:
                embeddings = pickle.load(f)
            print(f"    ⚡ Loaded embeddings from cache: {os.path.basename(cache_path)}")
            return embeddings
        except Exception as e:
            print(f"    Warning: Failed to load cached embeddings: {e}")
            return None
        
    def load_models(self):
        """Load all embedding models."""
        print(f"Loading embedding models...")
        for model_key, model_name in self.models_config.items():
            print(f"  - Loading Embedding Model '{model_key}': {model_name}")
            try:
                if model_name == 'BAAI/bge-m3':
                    self.loaded_models[model_key] = SentenceTransformer(
                        model_name, device=self.device, trust_remote_code=True
                    )
                else:
                    self.loaded_models[model_key] = SentenceTransformer(
                        model_name, device=self.device
                    )
                print(f"    Embedding model '{model_key} ({model_name})' loaded successfully.")
            except Exception as e:
                print(f"    Error loading embedding model '{model_key} ({model_name})': {e}")
        return self.loaded_models
    
    def get_embeddings(self, texts_list, model_obj, batch_size=32):
        """Generate embeddings for a list of texts."""
        if not texts_list or not all(isinstance(t, str) for t in texts_list):
            return np.array([])
        try:
            all_embeddings = model_obj.encode(
                texts_list, 
                convert_to_tensor=True, 
                show_progress_bar=False, 
                batch_size=batch_size
            )
            return all_embeddings.cpu().numpy()
        except Exception as e:
            print(f"Error during get_embeddings: {e}")
            return np.array([])
    
    def generate_embeddings_for_dataframe(self, df, text_column, model_key):
        """Generate embeddings for a dataframe column with caching support."""
        if model_key not in self.loaded_models:
            print(f"Model {model_key} not loaded.")
            return df
        
        # Check cache first if enabled
        if self.use_cache and self.cache_dir:
            cache_key = self._generate_cache_key(df, text_column, model_key)
            cache_path = self._get_cache_path(cache_key, model_key)
            
            if cache_path and os.path.exists(cache_path):
                cached_embeddings = self._load_embeddings_from_cache(cache_path)
                if cached_embeddings is not None and len(cached_embeddings) == len(df):
                    df[f'{model_key}_embedding'] = list(cached_embeddings)
                    print(f"    Stored cached embeddings in column '{model_key}_embedding'. Shape example: {cached_embeddings[0].shape if len(cached_embeddings) > 0 and hasattr(cached_embeddings[0], 'shape') else 'N/A'}")
                    return df
        
        # Generate new embeddings if not cached
        model_obj = self.loaded_models[model_key]
        embeddings = self.get_embeddings(df[text_column].tolist(), model_obj)
        
        if len(embeddings) == len(df):
            df[f'{model_key}_embedding'] = list(embeddings)
            print(f"    Stored embeddings in column '{model_key}_embedding'. Shape example: {embeddings[0].shape if len(embeddings) > 0 and hasattr(embeddings[0], 'shape') else 'N/A'}")
            
            # Save to cache if enabled
            if self.use_cache and self.cache_dir:
                cache_key = self._generate_cache_key(df, text_column, model_key)
                cache_path = self._get_cache_path(cache_key, model_key)
                if cache_path:
                    self._save_embeddings_to_cache(embeddings, cache_path)
        else:
            print(f"    Warning: Mismatch in length between dataframe ({len(df)}) and generated embeddings ({len(embeddings)}) for {model_key}.")
        
        return df 