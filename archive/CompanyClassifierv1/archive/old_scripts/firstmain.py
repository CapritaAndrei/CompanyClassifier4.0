import pandas as pd
import numpy as np
# from sentence_transformers import SentenceTransformer # Keep if Bi-Encoder might be reintroduced later
from transformers import AutoTokenizer, AutoModelForSequenceClassification # Changed import
# from sentence_transformers import CrossEncoder # Removed import
import torch # Added for softmax and device handling
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
from tqdm.auto import tqdm
import time
import math
import pickle  # <<< Added import
import os     # <<< Added import

# --- Configuration ---
warnings.filterwarnings("ignore")
try: # NLTK Download
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading NLTK resources...")
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

# --- Model & Parameters ---
SAMPLE_SIZE = 5  # Number of companies to process for development
# BI_ENCODER_MODEL = 'all-MiniLM-L6-v2' # For embedding sectors and maybe Stage 1
NLI_MODEL = 'facebook/bart-large-mnli' # NLI model for Zero-Shot
# CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2' # Previous relevance model
# K_CANDIDATES = 30  # No longer needed for filtering stage
FINAL_THRESHOLD = 0.75 # Threshold for final label selection (if re-enabled)
N_FINAL_MATCHES = 10 # Max number of final labels per company
TAXONOMY_MAP_FILE = 'taxonomy_sector_map_multilabel.pkl' # <<< Updated filename for new structure
SECTOR_MAPPING_THRESHOLD = 0.1 # <<< LOWERED Threshold for associating label with a sector

# Sector definitions for mapping (REFINED Descriptions)
SECTOR_DESCRIPTIONS = {
    'Education': "Businesses providing instruction and training, often operating for profit or as non-profit institutions.", # Acknowledged potential overlap
    'Manufacturing': "Businesses primarily operating for profit, involved in the physical or chemical transformation of materials or components into new products, including assembly.", # Added profit focus
    'Services': "Businesses primarily operating for profit by providing professional, scientific, technical, administrative, repair, installation, maintenance, consulting, or management services to clients.", # Added profit focus
    'Retail': "Businesses primarily operating for profit by selling merchandise directly to consumers, often from stores or online.", # Added profit focus
    'Wholesale': "Businesses primarily operating for profit by selling goods in large quantities to retailers or other businesses.", # Added profit focus
    'Government': "Public administration entities including municipalities, schools, and special districts operating without a commercial profit motive, requiring coverage for public official liability, municipal vehicles, public property, infrastructure projects, and employee benefits.", # Added non-profit motive clarification
    'Non Profit': "Organizations operating without a primary profit motive, typically for charitable, educational, religious, or social causes. Needs insurance for event liability, volunteer workers, directors and officers (D&O) liability related to board governance, property, fundraising activities, and professional services offered to beneficiaries.", # Emphasized no profit motive & D&O
    # 'Construction': "Businesses engaged in constructing, altering, or repairing buildings, infrastructure, or other structures.", # Removed
    # 'Transportation': "Businesses providing transportation of passengers or cargo, warehousing, or support activities.", # Removed
    # 'Healthcare': "Businesses providing medical, dental, diagnostic, therapeutic, or social assistance services.", # Removed
    # 'Agriculture': "Businesses growing crops, raising animals, harvesting timber or fish.", # Removed
    # 'Finance': "Businesses engaged in financial transactions, insurance, real estate, or holding assets.", # Removed
    # 'Technology': "Businesses developing or providing software, hardware, IT services, data processing, or telecommunications.", # Removed
    # 'Food & Beverage': "Businesses preparing meals, snacks, or drinks for immediate consumption, or selling specialty food items.", # Removed
    # 'Utilities': "Businesses providing electric power, natural gas, water supply, or sewage removal.", # Removed
    # 'General': "General business operations applicable across multiple sectors." # Removed Fallback
}

# --- Utility Functions ---
def preprocess_text(text):
    """Clean and standardize text."""
    if pd.isnull(text): return ""
    text = re.sub(r'[^\w\s]', ' ', str(text).lower())
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 1]
    return " ".join(filtered_words)

def get_taxonomy_phrases(label):
    """Extract meaningful single words and 2-word phrases from taxonomy labels."""
    # (Using the previous good version here)
    label = label.lower()
    stop_words = set(stopwords.words('english')) | {'and', 'of', 'for', 'the', 'in', 'to'}
    words = [w for w in re.findall(r'\b\w+\b', label) if w not in stop_words]
    phrases = set(words)
    if len(words) >= 2:
        for i in range(len(words) - 1): phrases.add(words[i] + " " + words[i+1])
    known_phrases = ["general liability", "professional liability", "property insurance", "workers compensation", "cyber liability"]
    for kp in known_phrases:
        if kp in label: phrases.add(kp)
    return list(phrases)

def create_full_company_text(row):
    """Create a single detailed text representation for Stage 2 Cross-Encoder."""
    desc = preprocess_text(row['description'])
    tags = preprocess_text(row['business_tags'])
    sector = row.get('sector', 'N/A') # Use raw sector name here for clarity
    category = row.get('category', 'N/A')
    niche = row.get('niche', 'N/A')
    # Structure for clarity
    return (
        f"Sector: {sector}. Category: {category}. Niche: {niche}. "
        f"Business Tags: {tags}. Description: {desc}."
    )

# --- NEW Function: TF-IDF Keyword Extraction ---
def extract_characteristic_keywords_tfidf(df, grouping_column, text_columns=['description', 'business_tags'], top_n=20):
    """
    Extracts characteristic keywords for groups using TF-IDF per document, then averaging.

    Args:
        df (pd.DataFrame): DataFrame containing company data.
        grouping_column (str): Column name to group by (e.g., 'sector', 'category').
        text_columns (list): List of column names containing text to analyze.
        top_n (int): Number of top keywords to return per group.

    Returns:
        dict: Dictionary where keys are group names and values are lists of top keywords.
    """
    print(f"\nExtracting keywords based on TF-IDF aggregation for column: '{grouping_column}'")

    # 1. Prepare text data
    df_copy = df.copy()
    # Combine specified text columns, handling potential NaNs
    df_copy['combined_text'] = ''
    for col in text_columns:
        if col in df_copy.columns:
            # Ensure column is string type before fillna and concatenation
            df_copy[col] = df_copy[col].astype(str)
            df_copy['combined_text'] += df_copy[col].fillna('') + ' '
        else:
            print(f"Warning: Text column '{col}' not found in DataFrame.")

    # Drop rows where the grouping column itself is NaN
    df_copy = df_copy.dropna(subset=[grouping_column])
    if df_copy.empty:
        print(f"Warning: No data left after dropping NaNs in '{grouping_column}'. Cannot extract keywords.")
        return {}

    # Ensure grouping column is suitable type (e.g., string) for grouping
    df_copy[grouping_column] = df_copy[grouping_column].astype(str)

    # Preprocess the combined text for each company (document)
    print("Preprocessing text for TF-IDF...")
    preprocessed_texts = [preprocess_text(text) for text in tqdm(df_copy['combined_text'], desc=f"Preprocessing {grouping_column}")]

    # Link preprocessed text back to original df rows for filtering and grouping
    df_copy['preprocessed_text'] = preprocessed_texts

    # Filter out companies whose text became empty after preprocessing
    df_filtered = df_copy[df_copy['preprocessed_text'] != ''].copy()

    if df_filtered.empty:
        print("Warning: All texts became empty after preprocessing. Cannot extract keywords.")
        return {}

    # 2. Calculate TF-IDF
    print("Calculating TF-IDF scores...")
    vectorizer = TfidfVectorizer(max_features=5000, # Limit dimensionality
                                 ngram_range=(1, 2)) # Consider uni- and bi-grams
                                 # Removed stop_words='english' as preprocess_text handles it
    tfidf_matrix = vectorizer.fit_transform(df_filtered['preprocessed_text'])
    feature_names = vectorizer.get_feature_names_out()

    # 3. Map Scores back to DataFrame (aligns with df_filtered index)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names, index=df_filtered.index)

    # Add the grouping column for the aggregation step
    tfidf_df[grouping_column] = df_filtered[grouping_column]

    # 4. Group and Aggregate (Calculate Mean TF-IDF per group)
    print(f"Aggregating TF-IDF scores by {grouping_column}...")
    grouped_tfidf = tfidf_df.groupby(grouping_column).mean()

    # 5. Extract Top Keywords
    characteristic_keywords = {}
    print(f"Extracting top {top_n} keywords per {grouping_column}...")
    for group_name, scores in grouped_tfidf.iterrows():
        # Sort scores and get indices of top N
        top_indices = scores.argsort()[::-1][:top_n]
        # Get corresponding feature names, ensure score > 0
        top_keywords = [feature_names[i] for i in top_indices if scores.iloc[i] > 0]
        characteristic_keywords[group_name] = top_keywords

    print(f"Finished extracting keywords for '{grouping_column}'.")
    return characteristic_keywords

# --- NEW Function: Map Taxonomy Labels to Sectors (Multi-Label) ---
def map_taxonomy_to_sectors_multilabel(taxonomy_df, sector_descriptions, nli_model, tokenizer, device, nli_score_threshold):
    """
    Maps each taxonomy label to potentially MULTIPLE relevant sectors using NLI,
    based on exceeding a score threshold.

    Args:
        taxonomy_df (pd.DataFrame): DataFrame containing taxonomy labels.
        sector_descriptions (dict): Dictionary mapping sector names to descriptions.
        nli_model: The trained NLI model.
        tokenizer: The tokenizer for the NLI model.
        device: The device to run inference on ('cuda' or 'cpu').
        nli_score_threshold (float): Minimum entailment probability to associate a label with a sector.

    Returns:
        dict: A dictionary mapping taxonomy label index to a LIST of matching sector names.
              Returns an empty list if no sector meets the threshold.
    """
    print(f"\nMapping taxonomy labels to multiple sectors (Threshold: {nli_score_threshold})...")
    taxonomy_sector_map = {} # {label_idx: [sector1, sector2, ...]}
    sector_names = list(sector_descriptions.keys())
    sector_premises = [sector_descriptions[name] for name in sector_names]

    with torch.no_grad():
        for idx, label in tqdm(taxonomy_df['label'].items(), total=len(taxonomy_df), desc="Mapping Labels"):
            if pd.isnull(label):
                taxonomy_sector_map[idx] = [] # Assign empty list for NaN labels
                continue

            label_hypothesis = f"This insurance label relates to activities in a specific business sector. The label is: {label}."
            matched_sectors = [] # List to store sectors meeting the threshold for this label

            for i, sector_premise in enumerate(sector_premises):
                inputs = tokenizer.encode_plus(sector_premise, label_hypothesis, return_tensors='pt',
                                               truncation='only_first', max_length=tokenizer.model_max_length).to(device)
                logits = nli_model(**inputs)[0]
                entail_contradiction_logits = logits[:, [0, 2]]
                probs = entail_contradiction_logits.softmax(dim=1)
                entailment_prob = probs[:, 1].item()

                if entailment_prob >= nli_score_threshold:
                    matched_sectors.append(sector_names[i]) # Add sector if score is high enough

            taxonomy_sector_map[idx] = matched_sectors # Assign the list of matched sectors

    # --- Optional: Print a summary of the mapping ---
    print("Finished mapping labels to sectors (multi-label).")
    # Calculate summary stats (e.g., avg sectors per label, count of labels per sector)
    all_mapped_sectors = [sector for sectors in taxonomy_sector_map.values() for sector in sectors]
    sector_counts = pd.Series(all_mapped_sectors).value_counts()
    avg_sectors = len(all_mapped_sectors) / len(taxonomy_df) if len(taxonomy_df) > 0 else 0
    labels_no_sector = sum(1 for sectors in taxonomy_sector_map.values() if not sectors)
    print(f"Avg sectors per label: {avg_sectors:.2f}")
    print(f"Labels with no sector assigned (below threshold {nli_score_threshold}): {labels_no_sector}")
    print("Sector assignment counts (labels can be in multiple):")
    print(sector_counts)
    # ---

    return taxonomy_sector_map


# --- Data Loading ---
def load_data(company_file, taxonomy_file):
    """Load company and taxonomy data."""
    print("Loading data...")
    try:
        companies_df = pd.read_csv(company_file)
        taxonomy_df = pd.read_excel(taxonomy_file)
        print(f"Loaded {len(companies_df)} companies and {len(taxonomy_df)} taxonomy labels.")
        # Basic cleaning: Fill NA in crucial columns for representation
        for col in ['description', 'business_tags', 'sector', 'category', 'niche']:
            if col in companies_df.columns:
                companies_df[col] = companies_df[col].fillna('')
        return companies_df, taxonomy_df
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None

# --- Taxonomy Processing Function ---
# Function REMOVED/Commented Out: process_taxonomy (not needed without Stage 1)

# --- Candidate Generation (Stage 1) ---
# Stage 1 is currently removed

# --- Zero-Shot Classification (Modified from Re-ranking) ---
def classify_zero_shot(company_row, all_taxonomy_indices, taxonomy_df, nli_model, tokenizer, device,
                       taxonomy_sector_map=None): # Map is now {idx: [sector1, ...]}
    """Classify company against relevant labels using NLI zero-shot approach,
       filtering labels based on multi-sector mapping."""

    relevant_taxonomy_indices = all_taxonomy_indices # Default to all
    company_sector = company_row.get('sector', '')

    # --- Filter based on multi-label map ---
    if taxonomy_sector_map and company_sector and company_sector in SECTOR_DESCRIPTIONS:
        relevant_taxonomy_indices = [
            idx for idx in all_taxonomy_indices
            # Check if the company's sector is IN the list of sectors mapped to this label
            if company_sector in taxonomy_sector_map.get(idx, [])
        ]
        print(f"  Company sector: {company_sector}. Filtering {len(all_taxonomy_indices)} labels down to {len(relevant_taxonomy_indices)} relevant labels (multi-map).")
    elif taxonomy_sector_map and company_sector == '':
         print(f"  Company sector missing/unknown. Comparing against all {len(all_taxonomy_indices)} labels.")
         # Still compare against all if company sector is unknown
    # No need for a separate 'else' if taxonomy_sector_map is None


    # --- Early exit if filtering resulted in no relevant labels ---
    if not relevant_taxonomy_indices:
        # This is more likely now if a company sector has few associated labels
        print(f"  No relevant labels identified for company {company_row.name} based on sector '{company_sector}' and multi-map. Skipping NLI.")
        return []

    company_premise = create_full_company_text(company_row) # Use the detailed text as premise
    results = []
    model_device = next(nli_model.parameters()).device

    with torch.no_grad(): # Ensure no gradients are computed
        # Process ONLY the relevant_taxonomy_indices
        for i in tqdm(relevant_taxonomy_indices, desc=f"  Labels Cmp {company_row.name}", leave=False):
            label = taxonomy_df['label'].iloc[i]
            if pd.isnull(label): continue

            hypothesis = f"Is this company's activity related to {label}?"
            inputs = tokenizer.encode_plus(company_premise, hypothesis, return_tensors='pt',
                                           truncation='only_first', max_length=tokenizer.model_max_length).to(device)
            logits = nli_model(**inputs)[0]
            entail_contradiction_logits = logits[:, [0, 2]]
            probs = entail_contradiction_logits.softmax(dim=1)
            prob_label_is_true = probs[:, 1].item()

            results.append({
                'label_idx': i,
                'label': label,
                'score': prob_label_is_true
            })

    results.sort(key=lambda x: x['score'], reverse=True)
    return results

# --- Final Selection & Output ---
def select_and_format_results(classification_results):
    """Select top N matches (ignoring threshold for now)."""
    final_matches = []
    # Iterate through the top results up to N_FINAL_MATCHES
    for result in classification_results[:N_FINAL_MATCHES]:
        # --- MODIFIED: Removed score threshold check ---
        # Original check:
        # if result['score'] >= FINAL_THRESHOLD and len(final_matches) < N_FINAL_MATCHES:
        # Simplified check (just limit by count):
        if len(final_matches) < N_FINAL_MATCHES:
            final_matches.append((result['label'], result['score']))
        else:
            # Should not happen if we slice with [:N_FINAL_MATCHES], but good practice
            break

    return final_matches

def generate_output(companies_df, all_final_matches, output_base_name):
    """Generate the final CSV output files."""
    # (Similar to previous version, but takes all_final_matches dict)
    print("Generating output files...")
    output_df = companies_df.copy()
    output_df['insurance_label'] = ""
    output_df['insurance_label_with_scores'] = ""

    matched_count = 0
    total_labels_assigned = 0

    for idx, row in output_df.iterrows():
        matches = all_final_matches.get(idx, []) # Get matches using original index
        if matches:
            matched_count += 1
            total_labels_assigned += len(matches)
            output_df.at[idx, 'insurance_label'] = '; '.join([m[0] for m in matches])
            output_df.at[idx, 'insurance_label_with_scores'] = '; '.join([f"{m[0]} ({m[1]:.2f})" for m in matches])
        else: # Ensure empty strings if no matches
             output_df.at[idx, 'insurance_label'] = ""
             output_df.at[idx, 'insurance_label_with_scores'] = ""


    # Reorder columns
    readable_cols = [
        'description', 'business_tags', 'insurance_label', 'insurance_label_with_scores',
        'sector', 'category', 'niche'
    ]
    # Ensure columns exist before selecting
    readable_cols = [col for col in readable_cols if col in output_df.columns]
    readable_df = output_df[readable_cols]

    readable_output_file = f"{output_base_name}_readable.csv"
    readable_df.to_csv(readable_output_file, index=False)
    print(f"Readable results saved to '{readable_output_file}'")

    # Summary
    if len(companies_df) > 0:
         percent_matched = (matched_count / len(companies_df)) * 100
         avg_labels = total_labels_assigned / matched_count if matched_count > 0 else 0
    else:
         percent_matched = 0
         avg_labels = 0

    summary = pd.DataFrame([{
        'total_companies': len(companies_df),
        'companies_with_labels': matched_count,
        'percentage_matched': f"{percent_matched:.1f}%",
        'avg_labels_per_matched_company': f"{avg_labels:.2f}",
        # Add sector counts if needed
    }])
    summary_output_file = f"{output_base_name}_summary.csv"
    summary.to_csv(summary_output_file, index=False)
    print(f"Summary statistics saved to '{summary_output_file}'")

    # Main output (as requested by task)
    main_output_df = companies_df.copy() # Start from original again
    main_output_df['insurance_label'] = ""
    for idx, row in main_output_df.iterrows():
         matches = all_final_matches.get(idx, [])
         if matches:
              main_output_df.at[idx, 'insurance_label'] = '; '.join([m[0] for m in matches])
         else:
              main_output_df.at[idx, 'insurance_label'] = ""

    main_output_file = f"{output_base_name}.csv"
    main_output_df.to_csv(main_output_file, index=False)
    print(f"Main results (with insurance_label column) saved to '{main_output_file}'")


# --- Main Execution ---
if __name__ == "__main__":
    start_time = time.time()

    # 1. Load Data
    companies_df, taxonomy_df = load_data('ml_insurance_challenge.csv', 'insurance_taxonomy.xlsx')

    # Determine device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Only proceed if data loaded successfully ---
    if companies_df is not None and taxonomy_df is not None:

        # --- Keyword Extraction (Optional - can keep for insights) ---
        # sector_keywords = extract_characteristic_keywords_tfidf(companies_df, 'sector')
        # ... (print keywords etc.) ...

        # 2. Initialize NLI Model (needed for mapping if file doesn't exist)
        print(f"\nInitializing NLI Model: {NLI_MODEL}")
        nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
        nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(device)
        nli_model.eval()
        print(f"Model loaded onto device: {next(nli_model.parameters()).device}")

        # --- Load or Calculate Taxonomy-Sector Map (Multi-Label) ---
        taxonomy_sector_map = None
        if os.path.exists(TAXONOMY_MAP_FILE):
            print(f"\nLoading existing multi-label taxonomy map from '{TAXONOMY_MAP_FILE}'...")
            try:
                with open(TAXONOMY_MAP_FILE, 'rb') as f:
                    taxonomy_sector_map = pickle.load(f)
                print("Multi-label taxonomy map loaded successfully.")
                # Add more robust validation if needed (e.g., check if values are lists)
                if not isinstance(taxonomy_sector_map, dict) or len(taxonomy_sector_map) != len(taxonomy_df):
                     print("Warning: Loaded map seems invalid (type or length mismatch). Recalculating.")
                     taxonomy_sector_map = None
                # Check a sample value to ensure it's a list (or None/empty list for NaN labels)
                elif len(taxonomy_df)>0 and not isinstance(taxonomy_sector_map.get(0, []), list):
                     print("Warning: Loaded map values do not appear to be lists. Recalculating.")
                     taxonomy_sector_map = None


            except Exception as e:
                print(f"Error loading taxonomy map: {e}. Will recalculate.")
                taxonomy_sector_map = None

        if taxonomy_sector_map is None:
            print("\nCalculating multi-label taxonomy-sector map (this may take a while)...")
            # Call mapping function with refined SECTOR_DESCRIPTIONS
            taxonomy_sector_map = map_taxonomy_to_sectors_multilabel(
                taxonomy_df, SECTOR_DESCRIPTIONS, nli_model, nli_tokenizer, device,
                nli_score_threshold=SECTOR_MAPPING_THRESHOLD
            )
            print(f"Saving multi-label taxonomy map to '{TAXONOMY_MAP_FILE}'...")
            try:
                with open(TAXONOMY_MAP_FILE, 'wb') as f:
                    pickle.dump(taxonomy_sector_map, f)
                print("Multi-label taxonomy map saved successfully.")
            except Exception as e:
                print(f"Error saving multi-label taxonomy map: {e}")


        # 4. Select Sample
        companies_sample_df = companies_df.head(SAMPLE_SIZE).copy()
        print(f"\nProcessing sample of {SAMPLE_SIZE} companies using Zero-Shot Classification with Sector Filtering...")

        # 5. Process Companies Sample (Using NLI Zero-Shot)
        all_final_matches = {}
        all_taxonomy_indices = list(range(len(taxonomy_df)))

        print("Running Zero-Shot Classification (filtered labels vs company where possible)...")
        companies_processed_for_debug = 0
        # Check if map is valid before proceeding with classification that depends on it
        if taxonomy_sector_map is not None:
            for i, (original_idx, row) in enumerate(tqdm(companies_sample_df.iterrows(), desc="Processing Sample")):
                classification_results = classify_zero_shot(
                    row,
                    all_taxonomy_indices,
                    taxonomy_df,
                    nli_model,
                    nli_tokenizer,
                    device,
                    taxonomy_sector_map=taxonomy_sector_map # Pass the map
                )

                # --- DEBUG Print ---
                if classification_results:
                    print(f"\nDEBUG: Top 5 Zero-Shot Scores for Company {original_idx} ({row.get('sector', 'Unknown')} - {row['description'][:50]}...)")
                    for k, res in enumerate(classification_results[:5]):
                         # Display list of mapped sectors
                         mapped_sectors = taxonomy_sector_map.get(res['label_idx'], []) # Should return list
                         print(f"  {k+1}. Label: {res['label']} (Mapped Sectors: {mapped_sectors}), Score: {res['score']:.4f}")
                else:
                    print(f"\nDEBUG: No relevant labels found or scored for Company {original_idx} ({row.get('sector', 'Unknown')} - {row['description'][:50]}...)")

                companies_processed_for_debug += 1
                final_matches = select_and_format_results(classification_results)
                all_final_matches[original_idx] = final_matches

            # 6. Generate Output
            generate_output(companies_sample_df, all_final_matches, output_base_name=f"zeroshot_sector_filtered_companies_sample{SAMPLE_SIZE}")

            end_time = time.time()
            print(f"\nTotal processing time for sample (zero-shot with sector filtering): {end_time - start_time:.2f} seconds")
        else:
             print("\nError: Taxonomy map could not be loaded or generated. Cannot proceed with classification.")


    else:
        print("Failed to load data. Exiting.")