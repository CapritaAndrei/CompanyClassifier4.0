import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # src -> project_root

COMPANY_DATA_FILE = os.path.join(PROJECT_ROOT, 'data', 'input', 'ml_insurance_challenge.csv')
TAXONOMY_FILE = os.path.join(PROJECT_ROOT, 'data', 'input', 'insurance_taxonomy.xlsx')
OUTPUT_BASE_NAME = os.path.join(PROJECT_ROOT, 'output', 'classified_companies_main') # Assuming output goes to output dir

# --- Cache Configuration ---
USE_CACHE = True # Master switch for using cached intermediate data

# Cache directories
CACHE_ROOT = os.path.join(PROJECT_ROOT, 'data', 'cache')
PREPROCESSED_CACHE_DIR = os.path.join(CACHE_ROOT, 'preprocessed')
EMBEDDINGS_CACHE_DIR = os.path.join(CACHE_ROOT, 'embeddings')
MODELS_CACHE_DIR = os.path.join(CACHE_ROOT, 'models')

# Ensure cache directories exist
os.makedirs(PREPROCESSED_CACHE_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_CACHE_DIR, exist_ok=True)
os.makedirs(MODELS_CACHE_DIR, exist_ok=True)

# Cache file paths
PREPROCESSED_COMPANIES_CACHE_FILE = os.path.join(PREPROCESSED_CACHE_DIR, 'preprocessed_companies.pkl')
PREPROCESSED_TAXONOMY_CACHE_FILE = os.path.join(PREPROCESSED_CACHE_DIR, 'preprocessed_taxonomy.pkl')
COMPANIES_WITH_EMBEDDINGS_CACHE_FILE = os.path.join(EMBEDDINGS_CACHE_DIR, 'companies_with_embeddings.pkl')
TAXONOMY_WITH_EMBEDDINGS_CACHE_FILE = os.path.join(EMBEDDINGS_CACHE_DIR, 'taxonomy_with_embeddings.pkl')

SAMPLE_SIZE = None # Process all companies for full performance test

# --- Embedding Configuration ---
EMBEDDING_MODELS_CONFIG = {
    'mini_lm': 'all-MiniLM-L6-v2',
    #'bge_m3': 'BAAI/bge-m3'  // consuma prea multe resurse (nu am memorie destula, chiar si cu CUDA :)) )
}
EMBEDDING_SIMILARITY_THRESHOLD = 0.7

