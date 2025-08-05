"""
Efficient matrix-based similarity calculations for company classification.
"""

import numpy as np
import pandas as pd
import time
from sklearn.metrics.pairwise import cosine_similarity


class MatrixSimilarityCalculator:
    """Efficient matrix-based similarity calculations."""
    
    def __init__(self):
        self.company_embeddings_matrix = None
        self.taxonomy_embeddings_matrix = None
        self.company_ids = None
        self.taxonomy_ids = None
        self.similarity_matrix = None
    
    def prepare_embedding_matrices(self, companies_df, taxonomy_df, company_embedding_col, taxonomy_embedding_col):
        """Convert dataframe embeddings to matrices for efficient computation."""
        print(f"    📊 Preparing embedding matrices...")
        start_time = time.time()
        
        # Extract company embeddings
        company_embeddings = []
        company_ids = []
        
        for idx, row in companies_df.iterrows():
            if company_embedding_col in row and row[company_embedding_col] is not None:
                embedding = row[company_embedding_col]
                if hasattr(embedding, 'shape') and len(embedding.shape) > 0:
                    company_embeddings.append(embedding)
                    company_ids.append(row.get('company_id', idx))
        
        # Extract taxonomy embeddings  
        taxonomy_embeddings = []
        taxonomy_ids = []
        
        for idx, row in taxonomy_df.iterrows():
            if taxonomy_embedding_col in row and row[taxonomy_embedding_col] is not None:
                embedding = row[taxonomy_embedding_col]
                if hasattr(embedding, 'shape') and len(embedding.shape) > 0:
                    taxonomy_embeddings.append(embedding)
                    taxonomy_ids.append(row.get('label', f'label_{idx}'))
        
        if not company_embeddings or not taxonomy_embeddings:
            print(f"    ❌ No valid embeddings found!")
            return False
        
        # Convert to matrices
        self.company_embeddings_matrix = np.vstack(company_embeddings)
        self.taxonomy_embeddings_matrix = np.vstack(taxonomy_embeddings) 
        self.company_ids = company_ids
        self.taxonomy_ids = taxonomy_ids
        
        prep_time = time.time() - start_time
        print(f"    ✅ Matrix preparation complete in {prep_time:.2f}s")
        print(f"       Companies matrix shape: {self.company_embeddings_matrix.shape}")
        print(f"       Taxonomy matrix shape: {self.taxonomy_embeddings_matrix.shape}")
        
        return True
    
    def calculate_similarity_matrix(self):
        """Calculate full similarity matrix using efficient matrix operations."""
        if self.company_embeddings_matrix is None or self.taxonomy_embeddings_matrix is None:
            print(f"    ❌ Embedding matrices not prepared!")
            return False
        
        print(f"    🚀 Computing similarity matrix...")
        start_time = time.time()
        
        # Use sklearn's optimized cosine similarity
        # Result shape: (n_companies, n_taxonomy_labels)
        self.similarity_matrix = cosine_similarity(
            self.company_embeddings_matrix, 
            self.taxonomy_embeddings_matrix
        )
        
        calc_time = time.time() - start_time
        total_similarities = self.similarity_matrix.shape[0] * self.similarity_matrix.shape[1]
        
        print(f"    ✅ Similarity matrix computed in {calc_time:.2f}s")
        print(f"       Matrix shape: {self.similarity_matrix.shape}")
        print(f"       Total similarities: {total_similarities:,}")
        print(f"       Rate: {total_similarities/calc_time:,.0f} similarities/second")
        
        return True
    
    def get_top_k_similarities(self, k=10, model_key='matrix_model'):
        """Extract top-k similarities for each company efficiently."""
        if self.similarity_matrix is None:
            print(f"    ❌ Similarity matrix not computed!")
            return pd.DataFrame()
        
        print(f"    📋 Extracting top-{k} similarities per company...")
        start_time = time.time()
        
        results = []
        
        # For each company (row in similarity matrix)
        for company_idx in range(self.similarity_matrix.shape[0]):
            company_id = self.company_ids[company_idx]
            similarities = self.similarity_matrix[company_idx]
            
            # Get top-k indices and scores
            top_k_indices = np.argpartition(similarities, -k)[-k:]
            top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]
            
            # Extract results
            for rank, taxonomy_idx in enumerate(top_k_indices, 1):
                similarity_score = similarities[taxonomy_idx]
                label_name = self.taxonomy_ids[taxonomy_idx]
                
                results.append({
                    'company_id': company_id,
                    'label_name': label_name,
                    'similarity_score': similarity_score,
                    'embedding_model': model_key,  # Add model key for compatibility
                    'rank': rank
                })
        
        extract_time = time.time() - start_time
        results_df = pd.DataFrame(results)
        
        print(f"    ✅ Top-k extraction complete in {extract_time:.2f}s")
        print(f"       Total results: {len(results_df):,}")
        
        return results_df
    
    def get_all_similarities_above_threshold(self, threshold=0.3):
        """Get all similarities above threshold for each company."""
        if self.similarity_matrix is None:
            print(f"    ❌ Similarity matrix not computed!")
            return pd.DataFrame()
        
        print(f"    📋 Extracting similarities above {threshold:.3f}...")
        start_time = time.time()
        
        results = []
        
        # Find all positions above threshold
        company_indices, taxonomy_indices = np.where(self.similarity_matrix >= threshold)
        
        for company_idx, taxonomy_idx in zip(company_indices, taxonomy_indices):
            company_id = self.company_ids[company_idx]
            label_name = self.taxonomy_ids[taxonomy_idx]
            similarity_score = self.similarity_matrix[company_idx, taxonomy_idx]
            
            results.append({
                'company_id': company_id,
                'label_name': label_name,
                'similarity_score': similarity_score
            })
        
        extract_time = time.time() - start_time
        results_df = pd.DataFrame(results)
        
        if not results_df.empty:
            # Add ranking within each company
            results_df['rank'] = results_df.groupby('company_id')['similarity_score'].rank(
                method='dense', ascending=False
            ).astype(int)
            results_df = results_df.sort_values(['company_id', 'rank'])
        
        print(f"    ✅ Threshold extraction complete in {extract_time:.2f}s")
        print(f"       Total results: {len(results_df):,}")
        
        return results_df
    
    def get_performance_stats(self):
        """Get performance statistics about the similarity calculation."""
        if self.similarity_matrix is None:
            return {}
        
        return {
            'n_companies': self.similarity_matrix.shape[0],
            'n_taxonomy_labels': self.similarity_matrix.shape[1], 
            'total_similarities_computed': self.similarity_matrix.size,
            'matrix_memory_mb': self.similarity_matrix.nbytes / (1024 * 1024),
            'avg_similarity': np.mean(self.similarity_matrix),
            'max_similarity': np.max(self.similarity_matrix),
            'min_similarity': np.min(self.similarity_matrix)
        } 