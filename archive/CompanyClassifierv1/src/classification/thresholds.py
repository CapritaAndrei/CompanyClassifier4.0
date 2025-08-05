"""
Threshold and selection strategies for classification.
"""

import pandas as pd


class TopKSelector:
    """Handles top-K selection for classification results."""
    
    def __init__(self, k=5):
        self.k = k
    
    def select_top_k_per_company(self, scores_df, k=None):
        """Select top K labels per company based on similarity scores."""
        if k is None:
            k = self.k
        
        if scores_df.empty:
            return pd.DataFrame()
        
        # Group by company and select top K
        top_k_results = scores_df.groupby('company_id', group_keys=False).apply(
            lambda x: x.nlargest(k, 'similarity_score')
        )
        
        return top_k_results
    
    def select_top_k_per_company_per_model(self, scores_df, k=None):
        """Select top K labels per company per model."""
        if k is None:
            k = self.k
        
        if scores_df.empty:
            return pd.DataFrame()
        
        # Group by company and model, then select top K
        top_k_results = scores_df.groupby(['company_id', 'embedding_model'], group_keys=False).apply(
            lambda x: x.nlargest(k, 'similarity_score')
        )
        
        return top_k_results
    
    def apply_threshold_after_topk(self, top_k_df, threshold):
        """Apply threshold filtering after top-K selection."""
        return top_k_df[top_k_df['similarity_score'] >= threshold]


class AdaptiveThresholdSelector:
    """
    Intelligent label selection using adaptive gap thresholding.
    Stops when similarity gaps become too large, with diminishing tolerance.
    """
    
    def __init__(self, initial_gap_threshold=0.15, decay_factor=0.8, max_labels=5, min_score=0.3):
        """
        Initialize adaptive threshold selector.
        
        Args:
            initial_gap_threshold: Starting gap threshold (0.15 = 15% relative gap)
            decay_factor: How much to reduce threshold each iteration (0.8 = 20% reduction)
            max_labels: Hard limit on number of labels per company
            min_score: Minimum absolute similarity score to consider
        """
        self.initial_gap_threshold = initial_gap_threshold
        self.decay_factor = decay_factor
        self.max_labels = max_labels
        self.min_score = min_score
    
    def select_adaptive_labels_per_company(self, scores_df):
        """Apply adaptive threshold selection per company."""
        if scores_df.empty:
            return pd.DataFrame()
        
        results = []
        
        # Group by company and model
        for (company_id, model), group in scores_df.groupby(['company_id', 'embedding_model']):
            # Sort by similarity score descending
            sorted_scores = group.sort_values('similarity_score', ascending=False)
            
            # Apply adaptive selection
            selected = self._adaptive_select_for_group(sorted_scores)
            if not selected.empty:
                results.append(selected)
        
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _adaptive_select_for_group(self, sorted_group):
        """Apply adaptive selection logic to a single company-model group."""
        selected_rows = []
        current_threshold = self.initial_gap_threshold
        
        for i, (_, row) in enumerate(sorted_group.iterrows()):
            # Always take the first (best) match if it meets minimum score
            if i == 0:
                if row['similarity_score'] >= self.min_score:
                    selected_rows.append(row)
                continue
            
            # Check hard limits (only if max_labels is set)
            if self.max_labels is not None and len(selected_rows) >= self.max_labels:
                break
            
            # Check minimum score
            if row['similarity_score'] < self.min_score:
                break
            
            # Calculate gap from previous score
            prev_score = selected_rows[-1]['similarity_score']
            current_score = row['similarity_score']
            
            # Calculate relative gap (percentage drop)
            gap = prev_score - current_score
            relative_gap = gap / prev_score if prev_score > 0 else 1.0
            
            # Check if gap is acceptable
            if relative_gap <= current_threshold:
                selected_rows.append(row)
                # Reduce threshold for next iteration (diminishing returns)
                current_threshold *= self.decay_factor
            else:
                # Gap too large, stop here
                break
        
        # Convert back to DataFrame
        if selected_rows:
            result_df = pd.DataFrame(selected_rows)
            # Add selection metadata
            result_df['selection_method'] = 'adaptive_threshold'
            result_df['num_selected'] = len(selected_rows)
            return result_df
        else:
            return pd.DataFrame()
    
    def get_selection_summary(self, results_df):
        """Get summary statistics of the selection process."""
        if results_df.empty:
            return {}
        
        summary = {
            'total_companies': results_df['company_id'].nunique(),
            'avg_labels_per_company': results_df.groupby('company_id').size().mean(),
            'min_labels_per_company': results_df.groupby('company_id').size().min(),
            'max_labels_per_company': results_df.groupby('company_id').size().max(),
            'companies_with_1_label': (results_df.groupby('company_id').size() == 1).sum(),
        }
        
        # Only include max_labels statistics if max_labels is set
        if self.max_labels is not None:
            summary['companies_with_max_labels'] = (results_df.groupby('company_id').size() == self.max_labels).sum()
        
        return summary 