# Import necessary libraries
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util # util for cosine_similarity if needed later
import torch
import os
import pickle # For saving/loading objects if needed later
import sys
import json # Though not immediately used, good to have for potential JSON handling
import ast  # For safely evaluating string literals, e.g., business_tags
from sklearn.metrics.pairwise import cosine_similarity # For embedding comparison

# --- Configuration ---
COMPANY_DATA_FILE = 'ml_insurance_challenge.csv'
TAXONOMY_FILE = 'insurance_taxonomy.xlsx'
OUTPUT_BASE_NAME = 'classified_companies_main'

# For development, let's process a small sample first
# Set SAMPLE_SIZE = None to process all data
SAMPLE_SIZE = 3 # Start with a very small number for quick testing

# NLI Model Configuration
NLI_MODEL_NAME = 'facebook/bart-large-mnli'
NLI_ENTAILMENT_THRESHOLD = 0.8

# Sentence Transformer Model Configuration
EMBEDDING_MODELS_CONFIG = {
    'mini_lm': 'all-MiniLM-L6-v2',
    'bge_m3': 'BAAI/bge-m3' # The new model you found
}
EMBEDDING_SIMILARITY_THRESHOLD = 0.7 # Example threshold, can be adjusted

# --- NLTK Resource Download & Initialization ---
nltk_stopwords = None
wordnet_lemmatizer = None

def download_nltk_resources():
    """Checks for necessary NLTK resources and attempts to download if missing."""
    print(f"NLTK default data path(s) being used: {nltk.data.path}")
    nltk_resources = {
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4", # Open Multilingual Wordnet, often a dependency for WordNet
        "tokenizers/punkt": "punkt"
    }
    all_resources_available = True
    for resource_path_segment, resource_name in nltk_resources.items():
        try:
            nltk.data.find(resource_path_segment)
            print(f"NLTK resource '{resource_name}' found.")
        except LookupError:
            print(f"NLTK resource '{resource_name}' not found. Attempting download...")
            try:
                nltk.download(resource_name, quiet=False)
                nltk.data.find(resource_path_segment) # Verify after download
                print(f"NLTK resource '{resource_name}' successfully downloaded/verified.")
            except Exception as e_download:
                print(f"ERROR: Failed to download or verify NLTK resource '{resource_name}'. Error: {e_download}")
                print(f"Please try running 'python -m nltk.downloader {resource_name}' manually.")
                all_resources_available = False
        except Exception as e_find_initial:
            print(f"Error initially checking for NLTK resource '{resource_name}': {e_find_initial}")
            all_resources_available = False
    if not all_resources_available:
        print("CRITICAL: One or more NLTK resources could not be made available. Please check errors above.")
    else:
        print("All necessary NLTK resources appear to be available.")
    return all_resources_available

def initialize_nlp_resources():
    """Ensures NLTK resources are downloaded and initializes stopwords set and lemmatizer globally."""
    global nltk_stopwords, wordnet_lemmatizer # Declare intent to modify globals
    print("\n--- 0. Initializing NLTK Resources ---")
    resources_ok = download_nltk_resources()
    if not resources_ok:
        print("Critical: Failed to ensure all NLTK resources during initialize_nlp_resources. Exiting.")
        sys.exit(1)
    try:
        nltk_stopwords = set(stopwords.words('english'))
        wordnet_lemmatizer = WordNetLemmatizer()
        print("NLTK stopwords and lemmatizer initialized.")
    except Exception as e:
        print(f"Error initializing NLTK stopwords/lemmatizer: {e}")
        print("This usually means wordnet or omw-1.4 are still not correctly found/loaded despite download attempts.")
        print("Please ensure NLTK data path is correct and resources are accessible.")
        sys.exit(1)

# --- Data Loading Function ---
def load_data(company_file_path, taxonomy_file_path):
    """Loads company and taxonomy data, ensuring essential ID columns exist."""
    print(f"\nLoading company data from: {company_file_path}")
    companies_df = None
    try:
        companies_df = pd.read_csv(company_file_path)
        print(f"Successfully loaded {len(companies_df)} companies.")
        if 'company_id' not in companies_df.columns:
            print("Warning: 'company_id' column not found in companies. Creating one from index: 'comp_idx_N'.")
            companies_df['company_id'] = [f'comp_idx_{i}' for i in range(len(companies_df))]
        else:
            if companies_df['company_id'].isnull().any():
                print("Warning: 'company_id' column in companies contains NaN values. Filling with unique IDs.")
                nan_ids_mask = companies_df['company_id'].isnull()
                companies_df.loc[nan_ids_mask, 'company_id'] = [f'comp_idx_nan_{i}' for i in range(nan_ids_mask.sum())]
            companies_df['company_id'] = companies_df['company_id'].astype(str)
    except FileNotFoundError:
        print(f"Error: Company data file not found at '{company_file_path}'.")
        return pd.DataFrame(), pd.DataFrame() # Return empty DFs if critical file missing
    except Exception as e:
        print(f"An unexpected error occurred during company data loading: {e}")
        return pd.DataFrame(), pd.DataFrame()

    print(f"Loading taxonomy data from: {taxonomy_file_path}")
    taxonomy_df = None
    try:
        taxonomy_df = pd.read_excel(taxonomy_file_path)
        print(f"Successfully loaded {len(taxonomy_df)} taxonomy entries.")
        expected_taxonomy_cols = ['label', 'Definition', 'Keywords']
        if not all(col in taxonomy_df.columns for col in expected_taxonomy_cols):
            print(f"Warning: Taxonomy file {taxonomy_file_path} does not seem to contain all expected columns: {expected_taxonomy_cols}.")
        if 'label' not in taxonomy_df.columns:
            print("CRITICAL ERROR: 'label' column is missing from the loaded taxonomy data. This column is essential. Exiting.")
            sys.exit(1)
        if 'label_id' not in taxonomy_df.columns:
            print("Warning: 'label_id' column not found in taxonomy. Creating one from index: 'label_idx_N'.")
            taxonomy_df['label_id'] = [f'label_idx_{i}' for i in range(len(taxonomy_df))]
        else:
            if taxonomy_df['label_id'].isnull().any():
                print("Warning: 'label_id' column in taxonomy contains NaN values. Filling with unique IDs.")
                nan_ids_mask = taxonomy_df['label_id'].isnull()
                taxonomy_df.loc[nan_ids_mask, 'label_id'] = [f'label_idx_nan_{i}' for i in range(nan_ids_mask.sum())]
            taxonomy_df['label_id'] = taxonomy_df['label_id'].astype(str)
    except FileNotFoundError:
        print(f"Error: Taxonomy data file not found at '{taxonomy_file_path}'.")
        return companies_df if companies_df is not None else pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred during taxonomy data loading: {e}")
        return companies_df if companies_df is not None else pd.DataFrame(), pd.DataFrame()

    # Basic cleaning: Fill NA in crucial text columns
    text_cols_to_clean_company = ['description', 'business_tags', 'sector', 'category', 'niche']
    if companies_df is not None:
        for col in text_cols_to_clean_company:
            if col in companies_df.columns:
                companies_df[col] = companies_df[col].fillna('')
            else:
                print(f"Warning: Expected column '{col}' not found in company data (will be treated as empty string).")
                companies_df[col] = '' # Ensure column exists for downstream processing
    
    text_cols_to_clean_taxonomy = ['Definition', 'Keywords', 'label']
    if taxonomy_df is not None:
        for col in text_cols_to_clean_taxonomy:
            if col in taxonomy_df.columns:
                taxonomy_df[col] = taxonomy_df[col].fillna('')
            else:
                print(f"Warning: Expected column '{col}' not found in taxonomy data (will be treated as empty string).")
                taxonomy_df[col] = '' # Ensure column exists
    return companies_df, taxonomy_df

# --- Text Cleaning Utilities ---
def text_to_lower(text):
    return text.lower()

def remove_punctuation(text):
    return re.sub(r'[^\\w\\s]', '', text) # Keeps alphanumeric and spaces

def remove_stopwords_from_text(text, stop_words_set):
    if not stop_words_set: return text
    return " ".join([word for word in text.split() if word not in stop_words_set])

def lemmatize_words_in_text(text, lemmatizer_obj):
    if not lemmatizer_obj: return text
    return " ".join([lemmatizer_obj.lemmatize(word) for word in text.split()])

# --- Master Preprocessing Function ---
def preprocess_text(text,
                    to_lower=True,
                    punct_remove=True,
                    stopword_remove=True,
                    lemmatize=True):
    global nltk_stopwords, wordnet_lemmatizer # Use globally initialized NLTK components
    if pd.isna(text) or not isinstance(text, str): return ""
    processed_text = text
    if to_lower: processed_text = text_to_lower(processed_text)
    if punct_remove: processed_text = remove_punctuation(processed_text)
    if stopword_remove:
        if nltk_stopwords: processed_text = remove_stopwords_from_text(processed_text, nltk_stopwords)
        else: print("Skipping stopword removal: nltk_stopwords not initialized.")
    if lemmatize:
        if wordnet_lemmatizer: processed_text = lemmatize_words_in_text(processed_text, wordnet_lemmatizer)
        else: print("Skipping lemmatization: wordnet_lemmatizer not initialized.")
    return re.sub(r'\\s+', ' ', processed_text).strip()

# --- Keyword and Tag Parsing/Processing Utilities ---
def parse_keywords_string(kw_string):
    if pd.isna(kw_string) or not isinstance(kw_string, str): return []
    return [keyword.strip() for keyword in kw_string.split(',') if keyword.strip()]

def parse_tag_string(tag_string):
    if pd.isna(tag_string) or not isinstance(tag_string, str) or not tag_string.strip(): return []
    try:
        evaluated_tags = ast.literal_eval(tag_string)
        if isinstance(evaluated_tags, (list, tuple)):
            return [str(tag).strip() for tag in evaluated_tags if str(tag).strip()]
        elif isinstance(evaluated_tags, str):
            if any(c in evaluated_tags for c in ['|', ',', ';']): # Heuristic for splittable single string
                return [t.strip() for t in re.split(r'[|;,]', evaluated_tags) if t.strip()]
            return [evaluated_tags.strip()] if evaluated_tags.strip() else []
        return [str(evaluated_tags).strip()] if str(evaluated_tags).strip() else [] # Catch other eval types
    except (ValueError, SyntaxError):
        return [tag.strip() for tag in re.split(r'\\s*[,|;]\\s*', tag_string) if tag.strip()]
    except Exception as e:
        print(f"Warning: Could not parse tag string '{tag_string}' due to {e}. Returning as empty list.")
        return []

# --- Rich Representation Creation ---
def create_company_representation(row):
    sector = str(row.get('sector', '')).strip()
    category = str(row.get('category', '')).strip()
    niche = str(row.get('niche', '')).strip()
    tags_str = str(row.get('processed_tags_string', '')).strip()
    desc_str = str(row.get('processed_description', '')).strip()
    representation = f"[SECTOR] {sector} [CATEGORY] {category} [NICHE] {niche} [TAGS] {tags_str} [DESCRIPTION] {desc_str}"
    return re.sub(r'\\s+', ' ', representation).strip()

def create_taxonomy_representation(row):
    label_name = str(row.get('label', '')).strip()
    definition_str = str(row.get('processed_definition', '')).strip()
    keywords_str = str(row.get('processed_keywords_string', '')).strip()
    representation = f"[LABEL] {label_name} [DEFINITION] {definition_str} [KEYWORDS] {keywords_str}"
    return re.sub(r'\\s+', ' ', representation).strip()

# --- Model Loading, Embedding, Classification Functions ---
def load_models(nli_model_name_config, embedding_models_dict_config, device_to_use):
    print(f"\nLoading models...")
    nli_tokenizer_obj, nli_model_obj = None, None
    loaded_embedding_models_dict = {}
    print(f"  - Loading NLI Model: {nli_model_name_config}")
    try:
        nli_tokenizer_obj = AutoTokenizer.from_pretrained(nli_model_name_config)
        nli_model_obj = AutoModelForSequenceClassification.from_pretrained(nli_model_name_config).to(device_to_use)
        nli_model_obj.eval()
        print(f"    NLI model '{nli_model_name_config}' loaded successfully.")
    except Exception as e:
        print(f"    Error loading NLI model '{nli_model_name_config}': {e}")

    for model_key, model_name_val in embedding_models_dict_config.items():
        print(f"  - Loading Embedding Model '{model_key}': {model_name_val}")
        try:
            if model_name_val == 'BAAI/bge-m3':
                loaded_embedding_models_dict[model_key] = SentenceTransformer(model_name_val, device=device_to_use, trust_remote_code=True)
            else:
                loaded_embedding_models_dict[model_key] = SentenceTransformer(model_name_val, device=device_to_use)
            print(f"    Embedding model '{model_key} ({model_name_val})' loaded successfully.")
        except Exception as e:
            print(f"    Error loading embedding model '{model_key} ({model_name_val})': {e}")
    return nli_tokenizer_obj, nli_model_obj, loaded_embedding_models_dict

def get_embeddings(texts_list, embedding_model_obj, device_to_use, batch_size=32):
    if not texts_list or not all(isinstance(t, str) for t in texts_list):
        # print("Warning: get_embeddings received an empty list or non-string elements.")
        return np.array([]) # Return empty numpy array for consistency
    try:
        all_embeddings = embedding_model_obj.encode(texts_list, convert_to_tensor=True, show_progress_bar=False, batch_size=batch_size)
        return all_embeddings.cpu().numpy()
    except Exception as e:
        print(f"Error during get_embeddings: {e}")
        return np.array([])

def classify_with_nli(premise_text, label_name_text, nli_tokenizer_obj, nli_model_obj, device_to_use):
    hypothesis = f"The company's operations and main activities are related to '{label_name_text}'."
    try:
        premise_text_str = str(premise_text) if premise_text is not None else ""
        label_name_str = str(label_name_text) if label_name_text is not None else ""
        inputs = nli_tokenizer_obj(premise_text_str, hypothesis, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device_to_use)
        with torch.no_grad():
            outputs = nli_model_obj(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
        
        ent_id = nli_model_obj.config.label2id.get('entailment', -1)
        neut_id = nli_model_obj.config.label2id.get('neutral', -1)
        contr_id = nli_model_obj.config.label2id.get('contradiction', -1)
        
        res = {'entailment': 0.0, 'neutral': 0.0, 'contradiction': 1.0}
        if ent_id != -1: res['entailment'] = probabilities[0][ent_id].item()
        if neut_id != -1: res['neutral'] = probabilities[0][neut_id].item()
        if contr_id != -1: res['contradiction'] = probabilities[0][contr_id].item()
        if ent_id == -1 or neut_id == -1 or contr_id == -1:
            print(f"Warning: Standard NLI labels (entailment, neutral, contradiction) not found in model config. Got: {nli_model_obj.config.label2id}")
        return res
    except Exception as e:
        print(f"Error during NLI classification for premise '{str(premise_text_str)[:50]}...' and label '{str(label_name_str)}': {e}")
        return {'entailment': 0.0, 'neutral': 0.0, 'contradiction': 1.0}

def classify_companies_with_embeddings(companies_dataframe, taxonomy_dataframe, model_key, sim_threshold):
    print(f"\n--- Running Embedding-based Classification with '{model_key}' (Threshold: {sim_threshold}) ---")
    results_list = []
    company_emb_col = f'{model_key}_embedding'
    taxonomy_emb_col = f'{model_key}_embedding'

    if company_emb_col not in companies_dataframe.columns or taxonomy_emb_col not in taxonomy_dataframe.columns:
        print(f"  Error: Embedding columns ('{company_emb_col}' or '{taxonomy_emb_col}') not found. Skipping '{model_key}'.")
        return []
    if companies_dataframe.empty or taxonomy_dataframe.empty:
        print(f"  Error: Companies or taxonomy dataframe is empty for '{model_key}'. Skipping.")
        return []
    
    # Ensure embeddings are valid 1D numpy arrays before trying to stack
    valid_company_mask = companies_dataframe[company_emb_col].apply(lambda x: isinstance(x, np.ndarray) and x.ndim == 1 and x.size > 0)
    valid_taxonomy_mask = taxonomy_dataframe[taxonomy_emb_col].apply(lambda x: isinstance(x, np.ndarray) and x.ndim == 1 and x.size > 0)

    temp_companies_df = companies_dataframe[valid_company_mask].copy()
    temp_taxonomy_df = taxonomy_dataframe[valid_taxonomy_mask].copy()

    if temp_companies_df.empty or temp_taxonomy_df.empty:
        print(f"  Warning: No valid embeddings found for '{model_key}' after filtering NaN/empty. Companies: {len(temp_companies_df)}, Labels: {len(temp_taxonomy_df)}. Skipping.")
        return []

    try:
        company_embeddings_matrix = np.vstack(temp_companies_df[company_emb_col].tolist())
        taxonomy_embeddings_matrix = np.vstack(temp_taxonomy_df[taxonomy_emb_col].tolist())
    except ValueError as e:
        print(f"  Error stacking embeddings for model '{model_key}': {e}. This might be due to inconsistent embedding dimensions or empty lists. Skipping.")
        return []
    except Exception as e_stack:
        print(f"  Unexpected error during embedding stacking for '{model_key}': {e_stack}. Skipping.")
        return []

    if company_embeddings_matrix.ndim != 2 or taxonomy_embeddings_matrix.ndim != 2 or \
       company_embeddings_matrix.shape[0] == 0 or taxonomy_embeddings_matrix.shape[0] == 0:
        print(f"  Error: Stacked embeddings are not valid 2D matrices or are empty for model '{model_key}'. Skipping.")
        return []        

    similarity_matrix = cosine_similarity(company_embeddings_matrix, taxonomy_embeddings_matrix)

    for comp_matrix_idx, comp_actual_idx in enumerate(temp_companies_df.index):
        comp_id = temp_companies_df.loc[comp_actual_idx].get('company_id', f'comp_idx_{comp_actual_idx}')
        # comp_desc = str(temp_companies_df.loc[comp_actual_idx].get('description', 'N/A'))[:50]
        # if comp_matrix_idx < 2: print(f"  Processing company ID {comp_id} ('{comp_desc}...') with '{model_key}'")

        for label_matrix_idx, label_actual_idx in enumerate(temp_taxonomy_df.index):
            score = similarity_matrix[comp_matrix_idx, label_matrix_idx]
            if score >= sim_threshold:
                label_name_val = temp_taxonomy_df.loc[label_actual_idx]['label']
                results_list.append({
                    'company_id': comp_id,
                    'label_name': label_name_val,
                    'embedding_model': model_key,
                    'similarity_score': score
                })
                # if comp_matrix_idx < 2 and sum(1 for r_item in results_list if r_item['company_id'] == comp_id) <=3 :
                #     print(f"    MATCH: Label '{label_name_val}' - Score: {score:.4f}")
    
    print(f"  Found {len(results_list)} potential matches with '{model_key}' above threshold {sim_threshold}.")
    return results_list

# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Starting Company Classification Process ---")

    # Step 0: Initialize NLTK (resources and lemmatizer/stopwords)
    initialize_nlp_resources() # This will set global nltk_stopwords and wordnet_lemmatizer

    # Step 1: Load Data
    print("\n--- 1. Loading Data ---")
    companies_df, taxonomy_df = load_data(COMPANY_DATA_FILE, TAXONOMY_FILE)

    if companies_df.empty or taxonomy_df.empty:
        print("Critical error: Data loading failed or returned empty dataframes. Exiting.")
        sys.exit(1)
    print(f"Initial load: {len(companies_df)} companies, {len(taxonomy_df)} taxonomy labels.")

    # Step 2: Apply SAMPLE_SIZE for faster iteration during development
    if SAMPLE_SIZE is not None:
        print(f"\n--- Applying SAMPLE_SIZE: Processing first {SAMPLE_SIZE} companies and taxonomy entries ---")
        companies_df = companies_df.head(SAMPLE_SIZE).copy() # Use .copy() to avoid SettingWithCopyWarning on slices
        taxonomy_df = taxonomy_df.head(SAMPLE_SIZE).copy()
        print(f"After sampling: {len(companies_df)} companies, {len(taxonomy_df)} taxonomy labels.")

    # Step 3: Preprocess Text Data (on sampled or full data)
    print("\n--- 3. Preprocessing Text Data ---")
    # Preprocess Taxonomy Data
    print("\nPreprocessing taxonomy definitions and keywords...")
    if 'Definition' in taxonomy_df.columns:
        taxonomy_df['processed_definition'] = taxonomy_df['Definition'].apply(
            lambda x: preprocess_text(x, to_lower=True, punct_remove=True, stopword_remove=False, lemmatize=False))
    else: print("  - Warning: 'Definition' column not found in taxonomy.")

    if 'Keywords' in taxonomy_df.columns:
        taxonomy_df['keyword_list'] = taxonomy_df['Keywords'].apply(parse_keywords_string)
        taxonomy_df['processed_keyword_list'] = taxonomy_df['keyword_list'].apply(
            lambda kws: [preprocess_text(kw, to_lower=True, punct_remove=True, stopword_remove=False, lemmatize=False) for kw in kws])
        taxonomy_df['processed_keywords_string'] = taxonomy_df['processed_keyword_list'].apply(lambda x: ' '.join(x))
    else: print("  - Warning: 'Keywords' column not found in taxonomy.")

    # Preprocess Company Data
    print("\nParsing and preprocessing company business tags...")
    if 'business_tags' in companies_df.columns:
        companies_df['parsed_tags_list'] = companies_df['business_tags'].apply(parse_tag_string)
        companies_df['processed_tags_list'] = companies_df['parsed_tags_list'].apply(
            lambda tags: [preprocess_text(t, to_lower=True, punct_remove=True, stopword_remove=False, lemmatize=False) for t in tags])
        companies_df['processed_tags_string'] = companies_df['processed_tags_list'].apply(lambda x: ' '.join(x))
    else: print("  - Warning: 'business_tags' column not found in companies.")

    print("\nPreprocessing company descriptions...")
    if 'description' in companies_df.columns:
        companies_df['processed_description'] = companies_df['description'].apply(
            lambda x: preprocess_text(x, to_lower=True, punct_remove=True, stopword_remove=False, lemmatize=False))
    else: print("  - Warning: 'description' column not found in companies.")

    # Create structured text representations (these will be used for embeddings)
    print("\nCreating structured company text representation...")
    companies_df['company_full_text_structured'] = companies_df.apply(create_company_representation, axis=1)
    print("\nCreating structured taxonomy text representation...")
    taxonomy_df['taxonomy_full_text_structured'] = taxonomy_df.apply(create_taxonomy_representation, axis=1)

    print("\nSample of structured company text:")
    if not companies_df.empty:
        print(companies_df[['company_id', 'company_full_text_structured']].head(min(SAMPLE_SIZE if SAMPLE_SIZE else len(companies_df), len(companies_df))).to_string(index=False))
    print("\nSample of structured taxonomy text:")
    if not taxonomy_df.empty:
        print(taxonomy_df[['label_id','label', 'taxonomy_full_text_structured']].head(min(SAMPLE_SIZE if SAMPLE_SIZE else len(taxonomy_df), len(taxonomy_df))).to_string(index=False))

    # Step 4: Load Models
    print("\n--- 4. Loading Hugging Face Models ---")
    current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {current_device}")
    nli_tokenizer, nli_model, loaded_embedding_models = load_models(NLI_MODEL_NAME, EMBEDDING_MODELS_CONFIG, current_device)

    if not nli_model or not nli_tokenizer:
        print("CRITICAL: NLI model or tokenizer failed to load. NLI-based strategies cannot proceed. Exiting.")
        sys.exit(1)
    if not loaded_embedding_models:
        print("WARNING: No embedding models were loaded. Embedding-based strategies will fail if attempted.")

    # Step 5: Generate Embeddings
    print("\n--- 5. Generating Embeddings for Downstream Tasks ---")
    if loaded_embedding_models:
        if 'company_full_text_structured' in companies_df.columns and not companies_df.empty:
            print("\nGenerating embeddings for company structured representations...")
            for model_key, model_obj in loaded_embedding_models.items():
                print(f"  Generating company embeddings using {model_key} for {len(companies_df)} companies...")
                company_embeddings_list = get_embeddings(companies_df['company_full_text_structured'].tolist(), model_obj, current_device)
                if len(company_embeddings_list) == len(companies_df):
                    companies_df[f'{model_key}_embedding'] = list(company_embeddings_list)
                    print(f"    Stored company embeddings in column '{model_key}_embedding'. Shape example: {company_embeddings_list[0].shape if len(company_embeddings_list) > 0 and hasattr(company_embeddings_list[0], 'shape') else 'N/A'}")
                else:
                    print(f"    Warning: Mismatch in length between companies ({len(companies_df)}) and generated embeddings ({len(company_embeddings_list)}) for {model_key}. Skipping column assignment.")
        else: print("  Skipping company embedding generation: 'company_full_text_structured' column not found or DataFrame empty.")

        if 'taxonomy_full_text_structured' in taxonomy_df.columns and not taxonomy_df.empty:
            print("\nGenerating embeddings for taxonomy structured representations...")
            for model_key, model_obj in loaded_embedding_models.items():
                print(f"  Generating taxonomy embeddings using {model_key} for {len(taxonomy_df)} labels...")
                taxonomy_embeddings_list = get_embeddings(taxonomy_df['taxonomy_full_text_structured'].tolist(), model_obj, current_device)
                if len(taxonomy_embeddings_list) == len(taxonomy_df):
                    taxonomy_df[f'{model_key}_embedding'] = list(taxonomy_embeddings_list)
                    print(f"    Stored taxonomy embeddings in column '{model_key}_embedding'. Shape example: {taxonomy_embeddings_list[0].shape if len(taxonomy_embeddings_list) > 0 and hasattr(taxonomy_embeddings_list[0], 'shape') else 'N/A'}")
                else:
                    print(f"    Warning: Mismatch in length between taxonomy labels ({len(taxonomy_df)}) and generated embeddings ({len(taxonomy_embeddings_list)}) for {model_key}. Skipping column assignment.")
        else: print("  Skipping taxonomy embedding generation: 'taxonomy_full_text_structured' column not found or DataFrame empty.")
        
        if not companies_df.empty:
            print("\nSample of company data with new embedding columns (presence check):")
            comp_embedding_cols_present = [col for col in companies_df.columns if '_embedding' in col]
            if comp_embedding_cols_present:
                sample_comp_emb_df = companies_df[['company_id'] + comp_embedding_cols_present].copy()
                for col in comp_embedding_cols_present: sample_comp_emb_df[col] = sample_comp_emb_df[col].apply(lambda x: isinstance(x, np.ndarray) and x.size > 0) # Check size instead of shape[0]
                print(sample_comp_emb_df.head(min(SAMPLE_SIZE if SAMPLE_SIZE else len(companies_df), len(companies_df))).to_string(index=False))
            else: print("  No embedding columns found in company_df for sample print.")
        if not taxonomy_df.empty:
            print("\nSample of taxonomy data with new embedding columns (presence check):")
            tax_embedding_cols_present = [col for col in taxonomy_df.columns if '_embedding' in col]
            if tax_embedding_cols_present:
                sample_tax_emb_df = taxonomy_df[['label_id', 'label'] + tax_embedding_cols_present].copy()
                for col in tax_embedding_cols_present: sample_tax_emb_df[col] = sample_tax_emb_df[col].apply(lambda x: isinstance(x, np.ndarray) and x.size > 0) # Check size instead of shape[0]
                print(sample_tax_emb_df.head(min(SAMPLE_SIZE if SAMPLE_SIZE else len(taxonomy_df), len(taxonomy_df))).to_string(index=False))
            else: print("  No embedding columns found in taxonomy_df for sample print.")
    else:
        print("No embedding models loaded. Skipping embedding generation step.")

    # Step 6: NLI Classification
    print(f"\n--- 6. Performing NLI-based Classification (Sample of {min(len(companies_df), SAMPLE_SIZE if SAMPLE_SIZE else len(companies_df))} companies vs {min(len(taxonomy_df), SAMPLE_SIZE if SAMPLE_SIZE else len(taxonomy_df))} labels) ---")
    nli_results_all_companies = []
    num_companies_to_sample_nli = min(len(companies_df), SAMPLE_SIZE if SAMPLE_SIZE is not None else len(companies_df))
    num_labels_to_sample_nli = min(len(taxonomy_df), SAMPLE_SIZE if SAMPLE_SIZE is not None else len(taxonomy_df))
    
    sampled_companies_for_nli = companies_df.iloc[:num_companies_to_sample_nli]
    sampled_taxonomy_for_nli = taxonomy_df.iloc[:num_labels_to_sample_nli]

    if not sampled_companies_for_nli.empty and not sampled_taxonomy_for_nli.empty:
        for _, company_row in sampled_companies_for_nli.iterrows():
            company_id_val = company_row.get('company_id')
            premise = company_row['company_full_text_structured']
            # print(f"\nClassifying Company ID (NLI): {company_id_val} ('{str(company_row.get('description', 'N/A'))[:30]}...')")
            for _, label_row in sampled_taxonomy_for_nli.iterrows():
                label_name = label_row['label']
                nli_scores = classify_with_nli(premise, label_name, nli_tokenizer, nli_model, current_device)
                nli_results_all_companies.append({
                    'company_id': company_id_val, 'label_name': label_name,
                    'entailment_score': nli_scores['entailment'],
                    'neutral_score': nli_scores['neutral'],
                    'contradiction_score': nli_scores['contradiction']
                })
    else:
        print("Skipping NLI classification as sampled companies or taxonomy is empty.")

    if nli_results_all_companies:
        nli_results_df = pd.DataFrame(nli_results_all_companies)
        print("\n--- Sample NLI Results (DataFrame) ---")
        print(nli_results_df.head(min(15, len(nli_results_df))).to_string(index=False))
        high_entailment_df = nli_results_df[nli_results_df['entailment_score'] >= NLI_ENTAILMENT_THRESHOLD]
        print(f"\n--- NLI Matches with Entailment >= {NLI_ENTAILMENT_THRESHOLD} (Sample) ---")
        print(high_entailment_df.head(min(15, len(high_entailment_df))).to_string(index=False))
    else: print("\nNo NLI results generated.")

    # Step 7: Embedding-based Similarity Classification
    print("\n--- 7. Embedding-based Similarity Classification ---")
    all_embedding_matches = []
    if loaded_embedding_models:
        # Using the already sampled (or full if SAMPLE_SIZE=None) companies_df and taxonomy_df from earlier
        if not companies_df.empty and not taxonomy_df.empty:
            for model_key_iter in loaded_embedding_models.keys():
                # Check if embedding columns exist before proceeding (they should have been created in Step 5)
                company_emb_col_name = f'{model_key_iter}_embedding'
                taxonomy_emb_col_name = f'{model_key_iter}_embedding'
                if company_emb_col_name in companies_df.columns and taxonomy_emb_col_name in taxonomy_df.columns:
                    embedding_matches = classify_companies_with_embeddings(
                        companies_df, taxonomy_df, model_key_iter, EMBEDDING_SIMILARITY_THRESHOLD
                    )
                    all_embedding_matches.extend(embedding_matches)
                else:
                    print(f"Skipping embedding classification for '{model_key_iter}': embedding columns ('{company_emb_col_name}' or '{taxonomy_emb_col_name}') not found.")
        else:
            print("Skipping embedding classification as companies_df or taxonomy_df is empty before this step.")

        if all_embedding_matches:
            embedding_matches_df = pd.DataFrame(all_embedding_matches)
            print("\n--- Sample Embedding-based Matches (Combined from all models) ---")
            print(embedding_matches_df.head(min(15, len(embedding_matches_df))).to_string(index=False))
            print("\n--- Top Embedding-based Matches (Sorted by Score) ---")
            print(embedding_matches_df.sort_values(by='similarity_score', ascending=False).head(min(15, len(embedding_matches_df))).to_string(index=False))
        else: print("\nNo matches found from embedding-based classification.")
    else:
        print("No embedding models loaded, so embedding-based classification cannot proceed.")

    print("\n--- Main execution finished. ---")