"""
Main orchestration script for company classification.
Clean, modular implementation following clean code principles.
"""

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

# Now, the relative imports should work when main.py is run as part of the src package
# e.g., python -m src.main from the project root
# or when other modules try to import from src (e.g. from src.data_utils)

import nltk
import os # Ensure os is imported if any os.path operations are ever added back

# Set specific NLTK data path UNCONDITIONALLY
# Ensure this path exists and contains the unzipped NLTK resources
# project_nltk_data_path = '/home/capri/Projects/Veridion/nltk_data'
# nltk.data.path = [project_nltk_data_path]

# Project root is one level up from the 'src' directory where main.py is
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
project_nltk_data_path = os.path.join(project_root, 'nltk_data')

# Set specific NLTK data path
# Ensure this path exists and contains the unzipped NLTK resources
nltk.data.path = [project_nltk_data_path]

import pandas as pd
import numpy as np
import re
# from nltk.corpus import stopwords # Imported within text_processing_utils
# from nltk.stem import WordNetLemmatizer # Imported within text_processing_utils
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import torch
# import os # Already imported above
import pickle
import json # Though not immediately used, good to have for potential JSON handling
# import ast # Imported within text_processing_utils
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm # Import tqdm

# Initialize tqdm for pandas operations (like progress_apply)
tqdm.pandas(desc="Pandas Processing")

# Import configurations
from .config import (
    COMPANY_DATA_FILE,
    TAXONOMY_FILE,
    OUTPUT_BASE_NAME,
    SAMPLE_SIZE,
    NLI_MODEL_NAME,
    NLI_ENTAILMENT_THRESHOLD,
    EMBEDDING_MODELS_CONFIG,
    EMBEDDING_SIMILARITY_THRESHOLD,
    USE_CACHE,  # New cache flag
    PREPROCESSED_COMPANIES_CACHE_FILE, # New cache file path
    PREPROCESSED_TAXONOMY_CACHE_FILE,  # New cache file path
    COMPANIES_WITH_EMBEDDINGS_CACHE_FILE, # New cache file path
    TAXONOMY_WITH_EMBEDDINGS_CACHE_FILE   # New cache file path
)

# Import from utility files
from .data_utils import load_data
from .text_processing_utils import (
    preprocess_text,
    parse_keywords_string,
    parse_tag_string,
    create_company_representation,
    create_taxonomy_representation
)
from .model_management_utils import (
    load_models,
    get_embeddings
)
from .classification_strategies import (
    classify_with_nli,
    classify_companies_with_embeddings
)

# --- Configuration ---
# ... (other configurations) ...
FORCE_CPU = True # <<< ADDED: Set to True to force CPU, False to use CUDA if available
TOP_N_EMBEDDING_MATCHES = 15 # <<< ADDED: How many top matches to show for embedding similarity

# --- Embedding-based Classification Function (Modified to return all scores for a company) ---
def classify_companies_with_embeddings(companies_dataframe, taxonomy_dataframe, model_key, sim_threshold): # sim_threshold can be kept for potential later use
    print(f"\n--- Calculating Embedding-based Similarities with '{model_key}' ---") # Changed title slightly
    all_company_label_scores = [] # Will store all scores for all companies

    company_emb_col = f'{model_key}_embedding'
    taxonomy_emb_col = f'{model_key}_embedding'

    if company_emb_col not in companies_dataframe.columns or taxonomy_emb_col not in taxonomy_dataframe.columns:
        print(f"  Error: Embedding columns ('{company_emb_col}' or '{taxonomy_emb_col}') not found. Skipping model '{model_key}'.")
        return []
    if companies_dataframe.empty or taxonomy_dataframe.empty:
        print("  Error: Companies or taxonomy dataframe is empty. Skipping.")
        return []
    
    valid_company_mask = companies_dataframe[company_emb_col].apply(lambda x: isinstance(x, np.ndarray) and x.ndim == 1 and x.size > 0)
    valid_taxonomy_mask = taxonomy_dataframe[taxonomy_emb_col].apply(lambda x: isinstance(x, np.ndarray) and x.ndim == 1 and x.size > 0)

    temp_companies_df = companies_dataframe[valid_company_mask].copy()
    temp_taxonomy_df = taxonomy_dataframe[valid_taxonomy_mask].copy()

    if temp_companies_df.empty or temp_taxonomy_df.empty:
        print(f"  Warning: No valid embeddings found for '{model_key}' after filtering. Skipping.")
        return []

    try:
        company_embeddings_matrix = np.vstack(temp_companies_df[company_emb_col].tolist())
        taxonomy_embeddings_matrix = np.vstack(temp_taxonomy_df[taxonomy_emb_col].tolist())
    except Exception as e_stack: # Catching general exception for stacking
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
            all_company_label_scores.append({ # Store all scores
                'company_id': comp_id,
                'label_name': label_name_val,
                'embedding_model': model_key,
                'similarity_score': score
            })
    
    print(f"  Calculated {len(all_company_label_scores)} company-label similarity scores with '{model_key}'.")
    return all_company_label_scores

# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Starting Company Classification Process ---")
    
    # Start overall timing
    overall_start_time = time.time()
    
    companies_df_full = None
    taxonomy_df_full = None
    loaded_from_embedding_cache = False
    loaded_from_preprocessing_cache = False

    # Attempt to load from the most processed cache first (embeddings cache)
    if USE_CACHE:
        print("\n--- Attempting to load data with embeddings from cache ---")
        if os.path.exists(COMPANIES_WITH_EMBEDDINGS_CACHE_FILE) and os.path.exists(TAXONOMY_WITH_EMBEDDINGS_CACHE_FILE):
            try:
                with open(COMPANIES_WITH_EMBEDDINGS_CACHE_FILE, 'rb') as f:
                    companies_df_full = pickle.load(f)
                with open(TAXONOMY_WITH_EMBEDDINGS_CACHE_FILE, 'rb') as f:
                    taxonomy_df_full = pickle.load(f)
                
                if companies_df_full is not None and not companies_df_full.empty and \
                   taxonomy_df_full is not None and not taxonomy_df_full.empty:
                    print(f"Successfully loaded companies ({len(companies_df_full)}) and taxonomy ({len(taxonomy_df_full)}) with embeddings from cache.")
                    loaded_from_embedding_cache = True
                else:
                    print("Warning: Embedding cache files loaded empty or problematic data. Will try reprocessing.")
                    loaded_from_embedding_cache = False # Treat as not loaded
                    companies_df_full, taxonomy_df_full = None, None # Reset
            except Exception as e:
                print(f"Error loading from embeddings cache: {e}. Proceeding with other methods.")
                companies_df_full, taxonomy_df_full = None, None
                loaded_from_embedding_cache = False # Ensure this is set to False on error
        else:
            print("Embeddings cache files not found.")

    # If not loaded from embedding cache, try preprocessed data cache
    if not loaded_from_embedding_cache and USE_CACHE:
        print("\n--- Attempting to load preprocessed data from cache ---")
        if os.path.exists(PREPROCESSED_COMPANIES_CACHE_FILE) and os.path.exists(PREPROCESSED_TAXONOMY_CACHE_FILE):
            try:
                with open(PREPROCESSED_COMPANIES_CACHE_FILE, 'rb') as f:
                    companies_df_full = pickle.load(f)
                with open(PREPROCESSED_TAXONOMY_CACHE_FILE, 'rb') as f:
                    taxonomy_df_full = pickle.load(f)

                if companies_df_full is not None and not companies_df_full.empty and \
                   taxonomy_df_full is not None and not taxonomy_df_full.empty:
                    print(f"Successfully loaded preprocessed companies ({len(companies_df_full)}) and taxonomy ({len(taxonomy_df_full)} labels). Preprocessing step will be skipped. Embeddings will be generated. ---")
                    loaded_from_preprocessing_cache = True
                else:
                    print("Warning: Preprocessed cache files loaded empty or problematic data. Will load raw.")
                    loaded_from_preprocessing_cache = False
                    companies_df_full, taxonomy_df_full = None, None # Reset
            except Exception as e:
                print(f"Error loading from preprocessed data cache: {e}. Proceeding to load raw data.")
                companies_df_full, taxonomy_df_full = None, None
                loaded_from_preprocessing_cache = False # Ensure this is set to False on error
        else:
            print("Preprocessed data cache files not found.")

    # Step 1: Load Raw Data (if not loaded from any cache)
    if not loaded_from_embedding_cache and not loaded_from_preprocessing_cache:
        print("\n--- 1. Loading Raw Data ---")
        companies_df_full, taxonomy_df_full = load_data(COMPANY_DATA_FILE, TAXONOMY_FILE)
        if companies_df_full.empty or taxonomy_df_full.empty:
            print("Critical error: Raw data loading failed or returned empty dataframes. Exiting.")
            sys.exit(1)
        print(f"Initial load: {len(companies_df_full)} companies, {len(taxonomy_df_full)} taxonomy labels.")
        # This data is now raw, needs preprocessing (Step 3) and embeddings (Step 5)
    elif loaded_from_preprocessing_cache:
        print(f"\n--- Data loaded from PREPROCESSED cache ({len(companies_df_full)} companies, {len(taxonomy_df_full)} labels). Preprocessing step will be skipped. Embeddings will be generated. ---")
        # Data is preprocessed, needs embeddings (Step 5)
    elif loaded_from_embedding_cache:
        print(f"\n--- Data loaded from EMBEDDINGS cache ({len(companies_df_full)} companies, {len(taxonomy_df_full)} labels). Preprocessing and Embedding steps will be skipped. ---")
        # Data is preprocessed and has embeddings. Both Step 3 and Step 5 will be skipped.

    # Ensure dataframes are not None if cache loading failed and raw loading was also skipped (should not happen with current logic but good check)
    if companies_df_full is None or taxonomy_df_full is None:
        print("Critical error: Dataframes are None after cache/load attempts. Exiting.")
        sys.exit(1)

    # Step 2: Apply SAMPLE_SIZE
    if SAMPLE_SIZE is not None:
        print(f"\n--- Applying SAMPLE_SIZE: Processing first {SAMPLE_SIZE} companies against ALL {len(taxonomy_df_full)} taxonomy entries ---")
        companies_df_processed = companies_df_full.head(SAMPLE_SIZE).copy()
        taxonomy_df_processed = taxonomy_df_full.copy() # Use a copy of the full taxonomy
        print(f"After sampling: {len(companies_df_processed)} companies will be processed against {len(taxonomy_df_processed)} taxonomy labels.")
    else:
        print(f"\n--- Processing ALL {len(companies_df_full)} companies against ALL {len(taxonomy_df_full)} taxonomy entries ---")
        companies_df_processed = companies_df_full.copy()
        taxonomy_df_processed = taxonomy_df_full.copy()

    # Step 3: Preprocess Text Data (only if not loaded from a cache that already includes preprocessing)
    if not loaded_from_embedding_cache and not loaded_from_preprocessing_cache:
        print("\n--- 3. Preprocessing Text Data (Operating on potentially sampled data) ---")
        # Preprocess Taxonomy Data
        print("\nPreprocessing taxonomy definitions and keywords...")
        if 'Definition' in taxonomy_df_processed.columns:
            taxonomy_df_processed['processed_definition'] = taxonomy_df_processed['Definition'].apply(
                lambda x: preprocess_text(x, to_lower=True, punct_remove=True, stopword_remove=False, lemmatize=False))
        else: print("  - Warning: 'Definition' column not found in taxonomy.")

        if 'Keywords' in taxonomy_df_processed.columns:
            taxonomy_df_processed['keyword_list'] = taxonomy_df_processed['Keywords'].apply(parse_keywords_string)
            taxonomy_df_processed['processed_keyword_list'] = taxonomy_df_processed['keyword_list'].apply(
                lambda kws: [preprocess_text(kw, to_lower=True, punct_remove=True, stopword_remove=False, lemmatize=False) for kw in kws])
            taxonomy_df_processed['processed_keywords_string'] = taxonomy_df_processed['processed_keyword_list'].apply(lambda x: ' '.join(x))
        else: print("  - Warning: 'Keywords' column not found in taxonomy.")

        # Preprocess Company Data
        print("\nParsing and preprocessing company business tags...")
        if 'business_tags' in companies_df_processed.columns:
            companies_df_processed['parsed_tags_list'] = companies_df_processed['business_tags'].apply(parse_tag_string)
            companies_df_processed['processed_tags_list'] = companies_df_processed['parsed_tags_list'].apply(
                lambda tags: [preprocess_text(t, to_lower=True, punct_remove=True, stopword_remove=False, lemmatize=False) for t in tags])
            companies_df_processed['processed_tags_string'] = companies_df_processed['processed_tags_list'].apply(lambda x: ' '.join(x))
        else: print("  - Warning: 'business_tags' column not found in companies.")

        print("\nPreprocessing company descriptions...")
        if 'description' in companies_df_processed.columns:
            companies_df_processed['processed_description'] = companies_df_processed['description'].apply(
                lambda x: preprocess_text(x, to_lower=True, punct_remove=True, stopword_remove=False, lemmatize=False))
        else: print("  - Warning: 'description' column not found in companies.")

        # Create structured text representations
        print("\nCreating structured company text representation...")
        companies_df_processed['company_full_text_structured'] = companies_df_processed.apply(create_company_representation, axis=1)
        print("\nCreating structured taxonomy text representation...")
        taxonomy_df_processed['taxonomy_full_text_structured'] = taxonomy_df_processed.apply(create_taxonomy_representation, axis=1)
        
        if USE_CACHE:
            print("\n--- Saving PREPROCESSED data to cache ---")
            # Note: These cache files will contain the (potentially sampled) _processed dataframes.
            # If SAMPLE_SIZE is active, this means you're caching a small subset.
            # For caching the full dataset preprocessed, run with SAMPLE_SIZE = None once.
            try:
                with open(PREPROCESSED_COMPANIES_CACHE_FILE, 'wb') as f:
                    pickle.dump(companies_df_processed, f)
                with open(PREPROCESSED_TAXONOMY_CACHE_FILE, 'wb') as f:
                    pickle.dump(taxonomy_df_processed, f)
                print("Successfully saved preprocessed data to cache.")
            except Exception as e:
                print(f"Error saving preprocessed data to cache: {e}")
    else:
        # This else corresponds to (loaded_from_embedding_cache OR loaded_from_preprocessing_cache) being true
        print("\n--- Skipping Step 3 (Preprocessing Text Data): Loaded from cache. ---")
        # Ensure the _processed dataframes still have the necessary structured text columns if loaded from cache
        # (This should be guaranteed if cache was saved correctly after these columns were made)
        if 'company_full_text_structured' not in companies_df_processed.columns and 'description' in companies_df_processed.columns: # Check if already exists, if not, create it
            print("Re-creating 'company_full_text_structured' on cached data as it's missing...")
            companies_df_processed['company_full_text_structured'] = companies_df_processed.apply(create_company_representation, axis=1)
        if 'taxonomy_full_text_structured' not in taxonomy_df_processed.columns and 'label' in taxonomy_df_processed.columns:
            print("Re-creating 'taxonomy_full_text_structured' on cached data as it's missing...")
            taxonomy_df_processed['taxonomy_full_text_structured'] = taxonomy_df_processed.apply(create_taxonomy_representation, axis=1)

    # Print samples using the processed (potentially sampled and/or cached) dataframes
    print("\nSample of structured company text (post-cache/preprocessing):")
    if not companies_df_processed.empty and 'company_id' in companies_df_processed.columns and 'company_full_text_structured' in companies_df_processed.columns:
        print(companies_df_processed[['company_id', 'company_full_text_structured']].head(min(SAMPLE_SIZE if SAMPLE_SIZE else len(companies_df_processed), len(companies_df_processed))).to_string(index=False))
    else: print("  Company data or required columns for sample print are missing.")
    
    print("\nSample of structured taxonomy text (post-cache/preprocessing):")
    if not taxonomy_df_processed.empty and 'label_id' in taxonomy_df_processed.columns and 'label' in taxonomy_df_processed.columns and 'taxonomy_full_text_structured' in taxonomy_df_processed.columns:
        print(taxonomy_df_processed[['label_id','label', 'taxonomy_full_text_structured']].head(min(SAMPLE_SIZE if SAMPLE_SIZE else len(taxonomy_df_processed), len(taxonomy_df_processed))).to_string(index=False))
    else: print("  Taxonomy data or required columns for sample print are missing.")

    # Step 4: Load Models
    print("\n--- 4. Loading Hugging Face Models ---")
    # <<< MODIFIED: Device selection logic >>>
    if FORCE_CPU:
        current_device = torch.device("cpu")
        print("Forcing CPU usage.")
    else:
        current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # <<< END MODIFICATION >>>
    print(f"Using device: {current_device}")
    
    # Load models (including NLI for completeness, but we'll skip using it)
    print("--- Loading models (NLI will be skipped for runtime testing) ---")
    nli_tokenizer, nli_model, loaded_embedding_models = load_models(NLI_MODEL_NAME, EMBEDDING_MODELS_CONFIG, current_device)
    
    # We'll allow NLI loading to proceed but won't use it
    # if not nli_model or not nli_tokenizer:
    #     print("CRITICAL: NLI model or tokenizer failed to load. NLI-based strategies cannot proceed. Exiting.")
    #     sys.exit(1)
    if not loaded_embedding_models:
        print("WARNING: No embedding models were loaded. Embedding-based strategies will fail if attempted.")
        sys.exit(1)

    # Step 5: Generate Embeddings (on processed DFs)
    if not loaded_from_embedding_cache:
        print("\n--- 5. Generating Embeddings for Downstream Tasks ---")
        if loaded_embedding_models:
            # Ensure structured text columns exist before trying to embed them
            if 'company_full_text_structured' in companies_df_processed.columns and not companies_df_processed.empty:
                print("\nGenerating embeddings for company structured representations...")
                for model_key, model_obj in loaded_embedding_models.items():
                    print(f"  Generating company embeddings using {model_key} for {len(companies_df_processed)} companies...")
                    company_embeddings_list = get_embeddings(companies_df_processed['company_full_text_structured'].tolist(), model_obj, current_device)
                    if len(company_embeddings_list) == len(companies_df_processed):
                        companies_df_processed[f'{model_key}_embedding'] = list(company_embeddings_list)
                        print(f"    Stored company embeddings in column '{model_key}_embedding'. Shape example: {company_embeddings_list[0].shape if len(company_embeddings_list) > 0 and hasattr(company_embeddings_list[0], 'shape') else 'N/A'}")
                    else:
                        print(f"    Warning: Mismatch in length between companies ({len(companies_df_processed)}) and generated embeddings ({len(company_embeddings_list)}) for {model_key}. Skipping column assignment.")
            else: print("  Skipping company embedding generation: 'company_full_text_structured' column not found or DataFrame empty.")

            if 'taxonomy_full_text_structured' in taxonomy_df_processed.columns and not taxonomy_df_processed.empty:
                print("\nGenerating embeddings for taxonomy structured representations...")
                for model_key, model_obj in loaded_embedding_models.items():
                    print(f"  Generating taxonomy embeddings using {model_key} for {len(taxonomy_df_processed)} labels...")
                    taxonomy_embeddings_list = get_embeddings(taxonomy_df_processed['taxonomy_full_text_structured'].tolist(), model_obj, current_device)
                    if len(taxonomy_embeddings_list) == len(taxonomy_df_processed):
                        taxonomy_df_processed[f'{model_key}_embedding'] = list(taxonomy_embeddings_list)
                        print(f"    Stored taxonomy embeddings in column '{model_key}_embedding'. Shape example: {taxonomy_embeddings_list[0].shape if len(taxonomy_embeddings_list) > 0 and hasattr(taxonomy_embeddings_list[0], 'shape') else 'N/A'}")
                    else:
                        print(f"    Warning: Mismatch in length between taxonomy labels ({len(taxonomy_df_processed)}) and generated embeddings ({len(taxonomy_embeddings_list)}) for {model_key}. Skipping column assignment.")
            else: print("  Skipping taxonomy embedding generation: 'taxonomy_full_text_structured' column not found or DataFrame empty.")
            
            if USE_CACHE: # Save the DFs that now contain embeddings
                print("\n--- Saving data WITH EMBEDDINGS to cache ---")
                # These cache files will contain the (possibly sampled) _processed dataframes WITH embedding columns
                try:
                    with open(COMPANIES_WITH_EMBEDDINGS_CACHE_FILE, 'wb') as f:
                        pickle.dump(companies_df_processed, f)
                    with open(TAXONOMY_WITH_EMBEDDINGS_CACHE_FILE, 'wb') as f:
                        pickle.dump(taxonomy_df_processed, f)
                    print("Successfully saved data with embeddings to cache.")
                except Exception as e:
                    print(f"Error saving data with embeddings to cache: {e}")
        else:
            print("No embedding models loaded. Skipping embedding generation step and subsequent caching.")
    else:
        print("\n--- Skipping Step 5 (Embedding Generation): Data with embeddings loaded from cache. ---")
    
    # Print samples of data with embeddings
    if not companies_df_processed.empty and loaded_embedding_models: # Only print if embeddings were expected
        print("\nSample of company data with new embedding columns (post-embedding/cache load, presence check):")
        comp_embedding_cols_present = [col for col in companies_df_processed.columns if '_embedding' in col]
        if comp_embedding_cols_present:
            sample_comp_emb_df = companies_df_processed[['company_id'] + comp_embedding_cols_present].copy()
            for col in comp_embedding_cols_present: sample_comp_emb_df[col] = sample_comp_emb_df[col].apply(lambda x: isinstance(x, np.ndarray) and x.size > 0)
            print(sample_comp_emb_df.head(min(SAMPLE_SIZE if SAMPLE_SIZE else len(companies_df_processed), len(companies_df_processed))).to_string(index=False))
        else: print("  No embedding columns found in company_df for sample print.")
    if not taxonomy_df_processed.empty and loaded_embedding_models: # Only print if embeddings were expected
        print("\nSample of taxonomy data with new embedding columns (post-embedding/cache load, presence check):")
        tax_embedding_cols_present = [col for col in taxonomy_df_processed.columns if '_embedding' in col]
        if tax_embedding_cols_present:
            sample_tax_emb_df = taxonomy_df_processed[['label_id', 'label'] + tax_embedding_cols_present].copy()
            for col in tax_embedding_cols_present: sample_tax_emb_df[col] = sample_tax_emb_df[col].apply(lambda x: isinstance(x, np.ndarray) and x.size > 0)
            print(sample_tax_emb_df.head(min(SAMPLE_SIZE if SAMPLE_SIZE else len(taxonomy_df_processed), len(taxonomy_df_processed))).to_string(index=False))
        else: print("  No embedding columns found in taxonomy_df for sample print.")

    # Step 6: NLI Classification - COMMENTED OUT FOR RUNTIME TESTING
    # print(f"\n--- 6. Performing NLI-based Classification ({len(companies_df_processed)} companies vs {len(taxonomy_df_processed)} labels) ---")
    # nli_results_all_companies = []

    # if not companies_df_processed.empty and not taxonomy_df_processed.empty:
    #     for _, company_row in tqdm(companies_df_processed.iterrows(), total=len(companies_df_processed), desc="NLI Classification (Companies)"):
    #         company_id_val = company_row.get('company_id')
    #         premise = company_row['company_full_text_structured']
    #         for _, label_row in taxonomy_df_processed.iterrows():
    #             label_name = label_row['label']
    #             nli_scores = classify_with_nli(premise, label_name, nli_tokenizer, nli_model, current_device)
    #             nli_results_all_companies.append({
    #                 'company_id': company_id_val, 'label_name': label_name, 
    #                 'entailment_score': nli_scores['entailment'],
    #                 'neutral_score': nli_scores['neutral'],
    #                 'contradiction_score': nli_scores['contradiction']
    #             })
    # else:
    #     print("Skipping NLI classification as processed companies or taxonomy is empty.")

    # if nli_results_all_companies:
    #     nli_results_df = pd.DataFrame(nli_results_all_companies)
    #     print(f"\n--- Top {TOP_N_EMBEDDING_MATCHES} NLI Results (per company, sorted by Entailment Score) ---")
    #     # Group by company_id and get top N for each
    #     top_n_nli_matches = nli_results_df.groupby('company_id', group_keys=False)\
    #                                 .apply(lambda x: x.nlargest(TOP_N_EMBEDDING_MATCHES, 'entailment_score'))
    #     if not top_n_nli_matches.empty:
    #         print(top_n_nli_matches.to_string(index=False))
    #     else:
    #         print("No NLI results to display after attempting to get top N (potentially all scores were low or data was empty).")
    # else: 
    #     print("\nNo NLI results generated.")

    print("\n--- Step 6: NLI Classification SKIPPED for runtime testing ---")

    # Step 7: Embedding-based Similarity Classification
    print("\n--- 7. Embedding-based Similarity Classification ---")
    
    # Start timing for embedding classification
    embedding_start_time = time.time()
    
    all_embedding_scores_df = pd.DataFrame() # To store all scores from all models before filtering top N
    
    if loaded_embedding_models:
        if not companies_df_processed.empty and not taxonomy_df_processed.empty:
            for model_key_iter in loaded_embedding_models.keys():
                model_start_time = time.time()
                print(f"\n--- Processing with model: {model_key_iter} ---")
                
                company_emb_col_name = f'{model_key_iter}_embedding'
                taxonomy_emb_col_name = f'{model_key_iter}_embedding'
                if company_emb_col_name in companies_df_processed.columns and taxonomy_emb_col_name in taxonomy_df_processed.columns:
                    # Get all scores from the function
                    current_model_scores = classify_companies_with_embeddings(
                        companies_df_processed,
                        taxonomy_df_processed,
                        model_key_iter,
                        EMBEDDING_SIMILARITY_THRESHOLD # Threshold is still passed but not used for filtering inside for now
                    )
                    if current_model_scores: # If scores were returned
                        all_embedding_scores_df = pd.concat([all_embedding_scores_df, pd.DataFrame(current_model_scores)], ignore_index=True)
                else:
                    print(f"Skipping embedding classification for '{model_key_iter}': embedding columns not found in processed DFs.")
                
                model_end_time = time.time()
                model_duration = model_end_time - model_start_time
                print(f"--- Model {model_key_iter} processing time: {model_duration:.2f} seconds ---")
        else:
            print("Skipping embedding classification as processed companies or taxonomy is empty.")
        
        if not all_embedding_scores_df.empty:
            print(f"\n--- Top {TOP_N_EMBEDDING_MATCHES} Embedding-based Matches (per company, per model) ---")
            # Group by company and model, then get top N for each group
            top_n_matches = all_embedding_scores_df.groupby(['company_id', 'embedding_model'], group_keys=False)\
                                            .apply(lambda x: x.nlargest(TOP_N_EMBEDDING_MATCHES, 'similarity_score'))
            if not top_n_matches.empty:
                print(top_n_matches.to_string(index=False))
            else:
                print("No embedding similarity results to display after attempting to get top N (potentially all scores were low or data was empty).")
            
            # Optional: Still show matches above original threshold if needed for comparison
            # high_similarity_matches = all_embedding_scores_df[all_embedding_scores_df['similarity_score'] >= EMBEDDING_SIMILARITY_THRESHOLD]
            # print(f"\n--- Embedding Matches with Similarity >= {EMBEDDING_SIMILARITY_THRESHOLD} (All Models) ---")
            # if not high_similarity_matches.empty:
            #     print(high_similarity_matches.sort_values(by=['company_id', 'embedding_model', 'similarity_score'], ascending=[True, True, False]).to_string(index=False))
            # else:
            #     print("No matches found above the similarity threshold.")
        else:
            print("\nNo similarity scores were generated from embedding-based classification.")
    else:
        print("No embedding models loaded, so embedding-based classification cannot proceed.")
    
    # End timing for embedding classification
    embedding_end_time = time.time()
    total_embedding_duration = embedding_end_time - embedding_start_time
    print(f"\n--- TOTAL EMBEDDING CLASSIFICATION TIME: {total_embedding_duration:.2f} seconds ---")

    print("\n--- Main execution finished. ---")
    
    # End overall timing
    overall_end_time = time.time()
    total_execution_duration = overall_end_time - overall_start_time
    print(f"\n--- TOTAL EXECUTION TIME: {total_execution_duration:.2f} seconds ---")
