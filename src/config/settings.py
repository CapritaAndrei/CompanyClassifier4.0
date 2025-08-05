"""
Configuration settings for the Insurance Company Classifier
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
DATA_INPUT_PATH = SRC_ROOT / "data" / "input"
DATA_PROCESSED_PATH = SRC_ROOT / "data" / "processed"
DATA_OUTPUT_PATH = SRC_ROOT / "data" / "output"

# Input files
COMPANY_DATA_FILE = DATA_INPUT_PATH / "ml_insurance_challenge.csv"
TAXONOMY_FILE = DATA_INPUT_PATH / "insurance_taxonomy.csv"

# Cache files
TAXONOMY_EMBEDDINGS_CACHE = DATA_PROCESSED_PATH / "taxonomy_embeddings.pkl"
PREPROCESSING_CACHE = DATA_PROCESSED_PATH / "preprocessed_data.pkl"

# Model settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence transformer model
SIMILARITY_THRESHOLD = 0.47  # Minimum similarity score for label assignment (updated for quality)
TOP_K_LABELS = 10 # Maximum number of labels to consider per company

# Classification settings
MIN_CONFIDENCE_SCORE = 0.47  # Minimum confidence to assign a label
MAX_LABELS_PER_COMPANY = 7 # Maximum labels to assign per company

# Text processing settings
MIN_TAG_LENGTH = 3  # Minimum length for business tags to be considered
KEYWORD_MATCH_BOOST = 0.2  # Boost for direct keyword matches

# Testing settings
TEST_SAMPLE_SIZE = 10  # Number of companies to test with initially

# Create directories if they don't exist
DATA_PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
DATA_OUTPUT_PATH.mkdir(parents=True, exist_ok=True) 