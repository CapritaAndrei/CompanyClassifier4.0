"""
Clustering-based validation for company classification results.
Groups similar companies and detects outlier label assignments.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')


class ClusterBasedValidator:
    """
    Validates classification results by clustering companies and detecting
    label assignments that don't fit the typical pattern for their cluster.
    """
    
    def __init__(self, n_clusters=None, outlier_threshold=0.1, min_cluster_size=2):
        """
        Initialize the cluster-based validator.
        
        Args:
            n_clusters: Number of clusters (auto-determined if None)
            outlier_threshold: Labels appearing in <threshold% of cluster = outliers
            min_cluster_size: Minimum companies per cluster
        """
        self.n_clusters = n_clusters
        self.outlier_threshold = outlier_threshold
        self.min_cluster_size = min_cluster_size
        self.cluster_model = None
        self.cluster_label_profiles = {}
        self.company_clusters = None
    
    def find_optimal_clusters(self, embeddings, max_clusters=8):
        """Find optimal number of clusters using silhouette score."""
        if len(embeddings) < 4:  # Need at least 4 points for meaningful clustering
            return 2
        
        max_clusters = min(max_clusters, len(embeddings) - 1)
        best_score = -1
        best_k = 2
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Check if all clusters have minimum size
            cluster_sizes = Counter(cluster_labels)
            if min(cluster_sizes.values()) < self.min_cluster_size:
                continue
                
            score = silhouette_score(embeddings, cluster_labels)
            if score > best_score:
                best_score = score
                best_k = k
        
        return best_k
    
    def cluster_companies(self, companies_df, embedding_column):
        """
        Cluster companies based on their embeddings.
        
        Args:
            companies_df: DataFrame with companies and embeddings
            embedding_column: Name of column containing embeddings
        
        Returns:
            Updated DataFrame with cluster assignments
        """
        print("\n--- Clustering Companies for Validation ---")
        
        # Extract embeddings
        embeddings = np.vstack(companies_df[embedding_column].values)
        
        # Determine optimal number of clusters
        if self.n_clusters is None:
            self.n_clusters = self.find_optimal_clusters(embeddings)
        
        print(f"  Clustering {len(companies_df)} companies into {self.n_clusters} groups...")
        
        # Perform clustering
        self.cluster_model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster_labels = self.cluster_model.fit_predict(embeddings)
        
        # Add cluster assignments to dataframe
        companies_df = companies_df.copy()
        companies_df['cluster'] = cluster_labels
        self.company_clusters = companies_df
        
        # Print cluster summary
        for cluster_id in range(self.n_clusters):
            cluster_companies = companies_df[companies_df['cluster'] == cluster_id]
            print(f"    Cluster {cluster_id}: {len(cluster_companies)} companies")
            
            # Show sample company descriptions for context
            if len(cluster_companies) > 0:
                sample_desc = cluster_companies.iloc[0].get('description', 'N/A')
                if len(sample_desc) > 80:
                    sample_desc = sample_desc[:80] + "..."
                print(f"      Sample: {sample_desc}")
        
        return companies_df
    
    def analyze_cluster_label_patterns(self, classification_results):
        """
        Analyze what labels are typical for each cluster.
        
        Args:
            classification_results: DataFrame with company_id, label_name, similarity_score, cluster
        """
        print("\n--- Analyzing Label Patterns per Cluster ---")
        
        self.cluster_label_profiles = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_results = classification_results[
                classification_results['company_id'].isin(
                    self.company_clusters[self.company_clusters['cluster'] == cluster_id]['company_id']
                )
            ]
            
            if len(cluster_results) == 0:
                continue
            
            # Count label frequency in this cluster
            label_counts = cluster_results['label_name'].value_counts()
            total_assignments = len(cluster_results)
            
            # Calculate label prevalence (what % of assignments in this cluster)
            label_prevalence = (label_counts / total_assignments).to_dict()
            
            # Identify common vs rare labels
            common_labels = {label: prev for label, prev in label_prevalence.items() 
                           if prev >= self.outlier_threshold}
            rare_labels = {label: prev for label, prev in label_prevalence.items() 
                         if prev < self.outlier_threshold}
            
            self.cluster_label_profiles[cluster_id] = {
                'common_labels': common_labels,
                'rare_labels': rare_labels,
                'total_companies': len(self.company_clusters[self.company_clusters['cluster'] == cluster_id]),
                'total_label_assignments': total_assignments
            }
            
            print(f"  Cluster {cluster_id} Profile:")
            print(f"    Companies: {self.cluster_label_profiles[cluster_id]['total_companies']}")
            print(f"    Common labels ({len(common_labels)}): {list(common_labels.keys())[:3]}...")
            print(f"    Rare labels ({len(rare_labels)}): {list(rare_labels.keys())[:3]}...")
    
    def detect_outlier_assignments(self, classification_results):
        """
        Detect label assignments that are outliers for their cluster.
        
        Args:
            classification_results: DataFrame with company_id, label_name, similarity_score
        
        Returns:
            DataFrame with outlier_score column (higher = more suspicious)
        """
        print("\n--- Detecting Outlier Label Assignments ---")
        
        # Add cluster info to results
        company_cluster_map = dict(zip(self.company_clusters['company_id'], self.company_clusters['cluster']))
        classification_results = classification_results.copy()
        classification_results['cluster'] = classification_results['company_id'].map(company_cluster_map)
        
        # Calculate outlier scores
        outlier_scores = []
        flagged_outliers = []
        
        for _, row in classification_results.iterrows():
            cluster_id = row['cluster']
            label_name = row['label_name']
            
            if cluster_id in self.cluster_label_profiles:
                # Check if this label is rare in this cluster
                cluster_profile = self.cluster_label_profiles[cluster_id]
                
                if label_name in cluster_profile['rare_labels']:
                    # This is a rare label for this cluster - potential outlier
                    rarity_score = 1.0 - cluster_profile['rare_labels'][label_name]
                    outlier_scores.append(rarity_score)
                    
                    if rarity_score > 0.8:  # Very rare = high suspicion
                        flagged_outliers.append({
                            'company_id': row['company_id'],
                            'label_name': label_name,
                            'similarity_score': row['similarity_score'],
                            'cluster': cluster_id,
                            'outlier_score': rarity_score
                        })
                else:
                    # Common label for this cluster
                    outlier_scores.append(0.0)
            else:
                outlier_scores.append(0.0)
        
        classification_results['outlier_score'] = outlier_scores
        
        print(f"  Detected {len(flagged_outliers)} potential outlier assignments")
        
        # Show examples of flagged outliers
        if flagged_outliers:
            print("  Top outlier examples:")
            for outlier in sorted(flagged_outliers, key=lambda x: x['outlier_score'], reverse=True)[:3]:
                print(f"    Company {outlier['company_id']} -> {outlier['label_name']} "
                      f"(score: {outlier['similarity_score']:.3f}, outlier: {outlier['outlier_score']:.3f})")
        
        return classification_results
    
    def validate_classifications(self, companies_df, classification_results, embedding_column):
        """
        Full validation pipeline: cluster companies and detect outlier label assignments.
        
        Args:
            companies_df: DataFrame with companies and embeddings
            classification_results: DataFrame with classification results
            embedding_column: Name of embedding column in companies_df
        
        Returns:
            Tuple of (clustered_companies_df, validated_results_df)
        """
        # Step 1: Cluster companies
        clustered_companies = self.cluster_companies(companies_df, embedding_column)
        
        # Step 2: Analyze label patterns per cluster
        self.analyze_cluster_label_patterns(classification_results)
        
        # Step 3: Detect outlier assignments
        validated_results = self.detect_outlier_assignments(classification_results)
        
        return clustered_companies, validated_results 