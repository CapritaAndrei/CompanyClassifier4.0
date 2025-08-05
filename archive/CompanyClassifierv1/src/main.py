import sys
import os
import time
import torch
import pandas as pd

# Path setup
src_dir = os.path.dirname(os.path.abspath(__file__))
project_root_for_path = os.path.dirname(src_dir)
if project_root_for_path not in sys.path:
    sys.path.insert(0, project_root_for_path)

# NLTK setup
import nltk
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
project_nltk_data_path = os.path.join(project_root, 'nltk_data')
nltk.data.path = [project_nltk_data_path]

# Import configurations
from .config import (
    COMPANY_DATA_FILE, TAXONOMY_FILE, SAMPLE_SIZE,
    EMBEDDING_MODELS_CONFIG,
    USE_CACHE, PREPROCESSED_COMPANIES_CACHE_FILE, PREPROCESSED_TAXONOMY_CACHE_FILE,
    COMPANIES_WITH_EMBEDDINGS_CACHE_FILE, TAXONOMY_WITH_EMBEDDINGS_CACHE_FILE,
    EMBEDDINGS_CACHE_DIR
)

FORCE_CPU = True
TOP_N_EMBEDDING_MATCHES = 10

# Import modules
from .data.loader import load_data
from .data.preprocessor import DataPreprocessor
from .models.embeddings import EmbeddingManager
from .classification.similarity import SimilarityClassifier
from .classification.matrix_similarity import MatrixSimilarityCalculator
from .classification.thresholds import TopKSelector, AdaptiveThresholdSelector
from .classification.cluster_validation import ClusterBasedValidator
from .data.multi_field_preprocessor import MultiFieldPreprocessor


def main():
    """Main execution function with production strategy."""
    print("--- Starting Company Classification with Production Strategy ---")
    print("🏆 STRATEGY: tags_focused + balanced (Optimized)")
    overall_start_time = time.time()
    
    # Initialize components
    preprocessor = DataPreprocessor(use_cache=USE_CACHE)
    multi_preprocessor = MultiFieldPreprocessor()
    similarity_classifier = SimilarityClassifier()
    matrix_similarity_calc = MatrixSimilarityCalculator()
    adaptive_selector = AdaptiveThresholdSelector(
        initial_gap_threshold=0.15,
        decay_factor=0.8,
        max_labels=None,  # No limit - let adaptive threshold decide
        min_score=0.25
    )
    cluster_validator = ClusterBasedValidator(
        outlier_threshold=0.1,  # Labels in <10% of cluster assignments = outliers
        min_cluster_size=3      # Allow small clusters for this test dataset
    )
    
    # Device setup
    device = torch.device("cpu" if FORCE_CPU else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Step 1: Load raw data
    companies_df, taxonomy_df = load_data(COMPANY_DATA_FILE, TAXONOMY_FILE)
    
    if companies_df.empty or taxonomy_df.empty:
        print("Critical error: No data available. Exiting.")
        sys.exit(1)

    # Step 2: Apply sampling
    if SAMPLE_SIZE is not None:
        companies_df = companies_df.head(SAMPLE_SIZE).copy()
        print(f"Using sample of {len(companies_df)} companies")
    
    # Step 3: Production preprocessing with winning strategy
    print("\n--- Applying Production Preprocessing Strategy ---")
    companies_df = multi_preprocessor.preprocess_companies_production(companies_df, 'tags_focused')
    taxonomy_df = multi_preprocessor.preprocess_taxonomy_production(taxonomy_df, 'balanced')
    
    # Step 4: Load embedding model
    embedding_manager = EmbeddingManager(EMBEDDING_MODELS_CONFIG, device, USE_CACHE, EMBEDDINGS_CACHE_DIR)
    loaded_models = embedding_manager.load_models()
    
    if not loaded_models:
        print("No embedding models loaded. Exiting.")
        sys.exit(1)

    # Step 5: Apply production strategy
    run_production_classification(
        companies_df, taxonomy_df, embedding_manager, loaded_models, 
        multi_preprocessor, similarity_classifier, matrix_similarity_calc, adaptive_selector, cluster_validator
    )
    
    total_time = time.time() - overall_start_time
    print(f"\n--- TOTAL EXECUTION TIME: {total_time:.2f} seconds ---")


def run_production_classification(companies_df, taxonomy_df, embedding_manager, loaded_models, 
                                 multi_preprocessor, similarity_classifier, matrix_similarity_calc, adaptive_selector, cluster_validator):
    """Run classification with the production-optimized strategy."""
    
    # Production strategy: tags_focused + balanced
    comp_strategy = 'tags_focused'
    tax_strategy = 'balanced'
    
    print(f"\n--- Applying Production Strategy: {comp_strategy} + {tax_strategy} ---")
    
    comp_text_col = f'company_text_{comp_strategy}'
    tax_text_col = f'taxonomy_text_{tax_strategy}'
    
    if comp_text_col not in companies_df.columns or tax_text_col not in taxonomy_df.columns:
        print(f"Error: Missing preprocessing columns")
        return
    
    # Generate embeddings
    companies_with_emb = companies_df.copy()
    taxonomy_with_emb = taxonomy_df.copy()
    
    for model_key in loaded_models.keys():
        print(f"  Generating embeddings with {model_key}...")
        
        companies_with_emb = embedding_manager.generate_embeddings_for_dataframe(
            companies_with_emb, comp_text_col, model_key
        )
        taxonomy_with_emb = embedding_manager.generate_embeddings_for_dataframe(
            taxonomy_with_emb, tax_text_col, model_key
        )
        
        # Rename columns for consistency
        if f'{model_key}_embedding' in companies_with_emb.columns:
            companies_with_emb[f'{model_key}_{comp_strategy}'] = companies_with_emb[f'{model_key}_embedding']
            companies_with_emb = companies_with_emb.drop(columns=[f'{model_key}_embedding'])
        
        if f'{model_key}_embedding' in taxonomy_with_emb.columns:
            taxonomy_with_emb[f'{model_key}_{tax_strategy}'] = taxonomy_with_emb[f'{model_key}_embedding']
            taxonomy_with_emb = taxonomy_with_emb.drop(columns=[f'{model_key}_embedding'])
    
    # Perform efficient matrix-based classification
    print(f"  Running efficient matrix similarity classification...")
    classification_start_time = time.time()
    
    # Use the first model for matrix calculation (can extend to multiple models later)
    primary_model_key = list(loaded_models.keys())[0]
    company_emb_col = f'{primary_model_key}_{comp_strategy}'
    taxonomy_emb_col = f'{primary_model_key}_{tax_strategy}'
    
    # Prepare embedding matrices
    if matrix_similarity_calc.prepare_embedding_matrices(
        companies_with_emb, taxonomy_with_emb, company_emb_col, taxonomy_emb_col
    ):
        # Calculate similarity matrix
        if matrix_similarity_calc.calculate_similarity_matrix():
            # Get performance stats
            perf_stats = matrix_similarity_calc.get_performance_stats()
            print(f"    📊 Performance: {perf_stats['total_similarities_computed']:,} similarities, "
                  f"{perf_stats['matrix_memory_mb']:.1f}MB matrix")
            
            # Extract top similarities using matrix approach
            matrix_results = matrix_similarity_calc.get_top_k_similarities(k=TOP_N_EMBEDDING_MATCHES, model_key=primary_model_key)
            
            if not matrix_results.empty:
                adaptive_matches = adaptive_selector.select_adaptive_labels_per_company(matrix_results)
                classification_time = time.time() - classification_start_time
                print(f"    ⚡ Matrix classification completed in {classification_time:.2f}s")
            else:
                print("    ❌ No matrix results generated")
                adaptive_matches = pd.DataFrame()
        else:
            print("    ❌ Failed to calculate similarity matrix")
            adaptive_matches = pd.DataFrame()
    else:
        print("    ❌ Failed to prepare embedding matrices")
        adaptive_matches = pd.DataFrame()
    
    # Fallback to old method if matrix approach fails
    if adaptive_matches.empty:
        print("    🔄 Falling back to pairwise similarity calculation...")
        all_scores = []
        for model_key in loaded_models.keys():
            scores = similarity_classifier.classify_all_scores_custom(
                companies_with_emb, taxonomy_with_emb, 
                f'{model_key}_{comp_strategy}', f'{model_key}_{tax_strategy}'
            )
            all_scores.extend(scores)
        
        if all_scores:
            scores_df = pd.DataFrame(all_scores)
            adaptive_matches = adaptive_selector.select_adaptive_labels_per_company(scores_df)
    
    if not adaptive_matches.empty:
        # Cluster validation disabled per user request
        validated_results = adaptive_matches.copy()
        validated_results['outlier_score'] = 0.0
        clustered_companies = companies_with_emb.copy()
        clustered_companies['cluster'] = 0
        outlier_count = 0
        outlier_percentage = 0.0
        
        # Calculate summary statistics
        summary = adaptive_selector.get_selection_summary(adaptive_matches)
        avg_labels = summary.get('avg_labels_per_company', 0)
        avg_score = adaptive_matches['similarity_score'].mean()
        
        print(f"\n--- CLASSIFICATION RESULTS ---")
        print(f"  Total companies processed: {len(companies_with_emb)}")
        print(f"  Average labels per company: {avg_labels:.1f}")
        print(f"  Average similarity score: {avg_score:.3f}")
        print(f"  Total label assignments: {len(adaptive_matches)}")
        
        # Show detailed results for first few companies
        display_detailed_results(validated_results, clustered_companies)
    else:
        print("  No classification results generated.")


def display_detailed_results(results_df, companies_df):
    """Display detailed classification results for debugging."""
    print(f"\n--- DETAILED RESULTS ---")
    
    # Show results for first few companies
    unique_companies = sorted(results_df['company_id'].unique())
    display_companies = unique_companies[:3]  # Show first 3 companies
    
    for company_id in display_companies:
        company_results = results_df[results_df['company_id'] == company_id].sort_values(
            'similarity_score', ascending=False
        )
        
        if not company_results.empty:
            print(f"\n🏢 COMPANY: {company_id}")
            
            # Show company description if available
            company_info = companies_df[companies_df['company_id'] == company_id]
            if not company_info.empty and 'description' in company_info.columns:
                desc = company_info.iloc[0]['description']
                if len(desc) > 100:
                    desc = desc[:100] + "..."
                print(f"   Description: {desc}")
            
            print("-" * 60)
            
            # Show top matches
            for idx, (_, row) in enumerate(company_results.head(7).iterrows(), 1):
                score = row['similarity_score']
                label_name = row['label_name']
                outlier_score = row.get('outlier_score', 0.0)
                outlier_flag = " ⚠️" if outlier_score > 0.5 else ""
                print(f"   {idx:2d}. {score:.3f} - {label_name}{outlier_flag}")
                if outlier_score > 0.5:
                    print(f"       ^ Outlier score: {outlier_score:.3f}")


if __name__ == "__main__":
    main() 