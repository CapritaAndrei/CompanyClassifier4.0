"""
Similarity-based classification for company classification.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityClassifier:
    """Handles embedding-based similarity classification."""
    
    def __init__(self, embedding_manager=None, taxonomy_labels=None):
        self.embedding_manager = embedding_manager
        self.taxonomy_labels = taxonomy_labels
        self.taxonomy_embeddings = None
        
        if self.embedding_manager and self.taxonomy_labels:
            self._create_taxonomy_embeddings()
    
    def _create_taxonomy_embeddings(self):
        """Create embeddings for all taxonomy labels."""
        print("Creating taxonomy embeddings for classification...")
        
        # Combine label, definition, and keywords for better embeddings
        taxonomy_descriptions = []
        for label in self.taxonomy_labels:
            # For now, just use the label itself
            # TODO: If you have taxonomy definitions/keywords, combine them here
            taxonomy_descriptions.append(label)
        
        # Create embeddings using the BGE model
        self.taxonomy_embeddings = self.embedding_manager.get_bge_embeddings(taxonomy_descriptions)
        print(f"Created embeddings for {len(self.taxonomy_labels)} taxonomy labels")
    
    def classify_single_top_k(self, description, top_k=3):
        """Classify a single description and return top-K labels."""
        if self.taxonomy_embeddings is None:
            raise ValueError("Taxonomy embeddings not initialized")
        
        # Get embedding for the description
        description_embedding = self.embedding_manager.get_bge_embeddings([description])[0]
        
        # Calculate similarities
        similarities = cosine_similarity([description_embedding], self.taxonomy_embeddings)[0]
        
        # Get top-K indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return top labels and scores
        top_labels = [self.taxonomy_labels[idx] for idx in top_indices]
        top_scores = [similarities[idx] for idx in top_indices]
        
        return {
            'top_labels': top_labels,
            'top_scores': top_scores
        }
    
    def classify_batch_top_k(self, descriptions, top_k=3, batch_size=50):
        """Classify a batch of descriptions and return top-K labels for each."""
        if self.taxonomy_embeddings is None:
            raise ValueError("Taxonomy embeddings not initialized")
        
        results = []
        
        # Process in smaller batches to manage memory
        for i in range(0, len(descriptions), batch_size):
            batch_descriptions = descriptions[i:i+batch_size]
            
            # Get embeddings for the batch
            batch_embeddings = self.embedding_manager.get_bge_embeddings(batch_descriptions)
            
            # Calculate similarities for the entire batch
            batch_similarities = cosine_similarity(batch_embeddings, self.taxonomy_embeddings)
            
            # Process each description in the batch
            for j, similarities in enumerate(batch_similarities):
                # Get top-K indices
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                
                # Get top labels and scores
                top_labels = [self.taxonomy_labels[idx] for idx in top_indices]
                top_scores = [similarities[idx] for idx in top_indices]
                
                results.append({
                    'top_labels': top_labels,
                    'top_scores': top_scores
                })
        
        return results
    
    def classify_all_scores(self, companies_df, taxonomy_df, model_key):
        """Calculate all similarity scores between companies and taxonomy labels."""
        print(f"\n--- Calculating Embedding-based Similarities with '{model_key}' ---")
        all_company_label_scores = []

        company_emb_col = f'{model_key}_embedding'
        taxonomy_emb_col = f'{model_key}_embedding'

        if company_emb_col not in companies_df.columns or taxonomy_emb_col not in taxonomy_df.columns:
            print(f"  Error: Embedding columns ('{company_emb_col}' or '{taxonomy_emb_col}') not found. Skipping model '{model_key}'.")
            return []
        
        if companies_df.empty or taxonomy_df.empty:
            print("  Error: Companies or taxonomy dataframe is empty. Skipping.")
            return []
        
        valid_company_mask = companies_df[company_emb_col].apply(
            lambda x: isinstance(x, np.ndarray) and x.ndim == 1 and x.size > 0
        )
        valid_taxonomy_mask = taxonomy_df[taxonomy_emb_col].apply(
            lambda x: isinstance(x, np.ndarray) and x.ndim == 1 and x.size > 0
        )

        temp_companies_df = companies_df[valid_company_mask].copy()
        temp_taxonomy_df = taxonomy_df[valid_taxonomy_mask].copy()

        if temp_companies_df.empty or temp_taxonomy_df.empty:
            print(f"  Warning: No valid embeddings found for '{model_key}' after filtering. Skipping.")
            return []

        try:
            company_embeddings_matrix = np.vstack(temp_companies_df[company_emb_col].tolist())
            taxonomy_embeddings_matrix = np.vstack(temp_taxonomy_df[taxonomy_emb_col].tolist())
        except Exception as e_stack:
            print(f"  Error stacking embeddings for model '{model_key}': {e_stack}. Skipping.")
            return []

        if company_embeddings_matrix.ndim != 2 or taxonomy_embeddings_matrix.ndim != 2 or \
           company_embeddings_matrix.shape[0] == 0 or taxonomy_embeddings_matrix.shape[0] == 0:
            print(f"  Error: Stacked embeddings are not valid 2D matrices or are empty for model '{model_key}'. Skipping.")
            return []

        similarity_matrix = cosine_similarity(company_embeddings_matrix, taxonomy_embeddings_matrix)

        for comp_matrix_idx, comp_actual_idx in enumerate(temp_companies_df.index):
            comp_id = temp_companies_df.loc[comp_actual_idx].get('company_id', f'comp_idx_{comp_actual_idx}')
            
            for label_matrix_idx, label_actual_idx in enumerate(temp_taxonomy_df.index):
                score = similarity_matrix[comp_matrix_idx, label_matrix_idx]
                label_name_val = temp_taxonomy_df.loc[label_actual_idx]['label']
                all_company_label_scores.append({
                    'company_id': comp_id,
                    'label_name': label_name_val,
                    'embedding_model': model_key,
                    'similarity_score': score
                })
        
        print(f"  Calculated {len(all_company_label_scores)} company-label similarity scores with '{model_key}'.")
        return all_company_label_scores
    
    def classify_with_threshold(self, companies_df, taxonomy_df, model_key, sim_threshold):
        """Classify using a similarity threshold."""
        all_scores = self.classify_all_scores(companies_df, taxonomy_df, model_key)
        return [score for score in all_scores if score['similarity_score'] >= sim_threshold]

    def classify_all_scores_custom(self, companies_df, taxonomy_df, company_emb_col, taxonomy_emb_col):
        """
        Compute similarity scores for all company-taxonomy pairs using custom embedding columns.
        """
        all_company_label_scores = []

        if company_emb_col not in companies_df.columns or taxonomy_emb_col not in taxonomy_df.columns:
            print(f"  Error: Embedding columns ('{company_emb_col}' or '{taxonomy_emb_col}') not found. Skipping.")
            return []
        
        if companies_df.empty or taxonomy_df.empty:
            print("  Error: Companies or taxonomy dataframe is empty. Skipping.")
            return []
        
        valid_company_mask = companies_df[company_emb_col].apply(
            lambda x: isinstance(x, np.ndarray) and x.ndim == 1 and x.size > 0
        )
        valid_taxonomy_mask = taxonomy_df[taxonomy_emb_col].apply(
            lambda x: isinstance(x, np.ndarray) and x.ndim == 1 and x.size > 0
        )

        temp_companies_df = companies_df[valid_company_mask].copy()
        temp_taxonomy_df = taxonomy_df[valid_taxonomy_mask].copy()

        if temp_companies_df.empty or temp_taxonomy_df.empty:
            print(f"  Warning: No valid embeddings found for custom columns after filtering. Skipping.")
            return []

        try:
            company_embeddings_matrix = np.vstack(temp_companies_df[company_emb_col].tolist())
            taxonomy_embeddings_matrix = np.vstack(temp_taxonomy_df[taxonomy_emb_col].tolist())
        except Exception as e_stack:
            print(f"  Error stacking embeddings for custom columns: {e_stack}. Skipping.")
            return []

        if company_embeddings_matrix.ndim != 2 or taxonomy_embeddings_matrix.ndim != 2 or \
           company_embeddings_matrix.shape[0] == 0 or taxonomy_embeddings_matrix.shape[0] == 0:
            print(f"  Error: Stacked embeddings are not valid 2D matrices or are empty for custom columns. Skipping.")
            return []

        similarity_matrix = cosine_similarity(company_embeddings_matrix, taxonomy_embeddings_matrix)

        for comp_matrix_idx, comp_actual_idx in enumerate(temp_companies_df.index):
            comp_id = temp_companies_df.loc[comp_actual_idx].get('company_id', f'comp_idx_{comp_actual_idx}')
            for label_matrix_idx, label_actual_idx in enumerate(temp_taxonomy_df.index):
                score = similarity_matrix[comp_matrix_idx, label_matrix_idx]
                label_name_val = temp_taxonomy_df.loc[label_actual_idx]['label']
                all_company_label_scores.append({
                    'company_id': comp_id,
                    'label_name': label_name_val,
                    'embedding_model': company_emb_col,
                    'similarity_score': score
                })
        print(f"  Calculated {len(all_company_label_scores)} company-label similarity scores with custom columns '{company_emb_col}' and '{taxonomy_emb_col}'.")
        return all_company_label_scores 