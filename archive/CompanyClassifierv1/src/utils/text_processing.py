"""
Text processing utilities for company classification.
"""

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import re
import ast
import sys

# Global variables for NLTK resources
nltk_stopwords = None
wordnet_lemmatizer = None


def download_nltk_resources():
    """Checks for necessary NLTK resources and attempts to download if missing."""
    print(f"NLTK data path(s) being used for resource check: {nltk.data.path}")
    nltk_resources = {
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
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
                nltk.data.find(resource_path_segment)
                print(f"NLTK resource '{resource_name}' successfully downloaded/verified.")
            except Exception as e_download:
                print(f"ERROR: Failed to download or verify NLTK resource '{resource_name}'. Error: {e_download}")
                all_resources_available = False
        except Exception as e_find_initial:
            print(f"Error initially checking for NLTK resource '{resource_name}': {e_find_initial}")
            all_resources_available = False
    if not all_resources_available:
        print("CRITICAL: One or more NLTK resources could not be made available during text_processing_utils setup.")
    else:
        print("All necessary NLTK resources appear to be available for text_processing_utils.")
    return all_resources_available


def initialize_nlp_resources():
    """Ensures NLTK resources are downloaded and initializes stopwords set and lemmatizer globally."""
    global nltk_stopwords, wordnet_lemmatizer
    print("\n--- Initializing NLP Resources (within text_processing_utils) ---")
    resources_ok = download_nltk_resources()
    if not resources_ok:
        print("Critical: Failed to ensure all NLTK resources during NLP initialization. Exiting.")
        sys.exit(1)
    try:
        nltk_stopwords = set(stopwords.words('english'))
        wordnet_lemmatizer = WordNetLemmatizer()
        print("NLTK stopwords and lemmatizer initialized (within text_processing_utils).")
    except Exception as e:
        print(f"Error initializing NLTK stopwords/lemmatizer: {e}")
        sys.exit(1)


def text_to_lower(text):
    """Convert text to lowercase."""
    return text.lower()


def remove_punctuation(text):
    """Remove punctuation from text."""
    return re.sub(r'[^\w\s]', '', text)


def remove_stopwords_from_text(text, stop_words_set):
    """Remove stopwords from text."""
    if not stop_words_set:
        return text
    return " ".join([word for word in text.split() if word not in stop_words_set])


def lemmatize_words_in_text(text, lemmatizer_obj):
    """Lemmatize words in text."""
    if not lemmatizer_obj:
        return text
    return " ".join([lemmatizer_obj.lemmatize(word) for word in text.split()])


def preprocess_text(text, to_lower=True, punct_remove=True, stopword_remove=True, lemmatize=True):
    """Master preprocessing function for text."""
    global nltk_stopwords, wordnet_lemmatizer
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    processed_text = text
    if to_lower:
        processed_text = text_to_lower(processed_text)
    if punct_remove:
        processed_text = remove_punctuation(processed_text)
    if stopword_remove:
        if nltk_stopwords:
            processed_text = remove_stopwords_from_text(processed_text, nltk_stopwords)
        else:
            print("Skipping stopword removal: nltk_stopwords not initialized in text_processing_utils.")
    if lemmatize:
        if wordnet_lemmatizer:
            processed_text = lemmatize_words_in_text(processed_text, wordnet_lemmatizer)
        else:
            print("Skipping lemmatization: wordnet_lemmatizer not initialized in text_processing_utils.")
    return re.sub(r'\s+', ' ', processed_text).strip()


def parse_keywords_string(kw_string):
    """Parse keywords from string."""
    if pd.isna(kw_string) or not isinstance(kw_string, str):
        return []
    return [keyword.strip() for keyword in kw_string.split(',') if keyword.strip()]


def parse_tag_string(tag_string):
    """Parse tags from string."""
    if pd.isna(tag_string) or not isinstance(tag_string, str) or not tag_string.strip():
        return []
    try:
        evaluated_tags = ast.literal_eval(tag_string)
        if isinstance(evaluated_tags, (list, tuple)):
            return [str(tag).strip() for tag in evaluated_tags if str(tag).strip()]
        elif isinstance(evaluated_tags, str):
            if any(c in evaluated_tags for c in ['|', ',', ';']):
                return [t.strip() for t in re.split(r'[|;,]', evaluated_tags) if t.strip()]
            return [evaluated_tags.strip()] if evaluated_tags.strip() else []
        return [str(evaluated_tags).strip()] if str(evaluated_tags).strip() else []
    except (ValueError, SyntaxError):
        return [tag.strip() for tag in re.split(r'\s*[,|;]\s*', tag_string) if tag.strip()]
    except Exception as e:
        print(f"Warning: Could not parse tag string '{tag_string}' due to {e}. Returning as empty list.")
        return []


def create_company_representation(row):
    """Create structured text representation for company."""
    sector = str(row.get('sector', '')).strip()
    category = str(row.get('category', '')).strip()
    niche = str(row.get('niche', '')).strip()
    tags_str = str(row.get('processed_tags_string', '')).strip()
    desc_str = str(row.get('processed_description', '')).strip()
    representation = f"[SECTOR] {sector} [CATEGORY] {category} [NICHE] {niche} [TAGS] {tags_str} [DESCRIPTION] {desc_str}"
    return re.sub(r'\s+', ' ', representation).strip()


def create_taxonomy_representation(row):
    """Create structured text representation for taxonomy."""
    label_name = str(row.get('label', '')).strip()
    definition_str = str(row.get('processed_definition', '')).strip()
    keywords_str = str(row.get('processed_keywords_string', '')).strip()
    representation = f"[LABEL] {label_name} [DEFINITION] {definition_str} [KEYWORDS] {keywords_str}"
    return re.sub(r'\s+', ' ', representation).strip()


# Initialize NLP resources when module is imported
initialize_nlp_resources() 