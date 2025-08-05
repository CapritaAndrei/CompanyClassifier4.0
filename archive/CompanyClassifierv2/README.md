# SIC Classification System

A hierarchical SIC (Standard Industrial Classification) code classifier for companies using machine learning and semantic embeddings.

## 🏗️ Clean Architecture

```
Veridion2/
├── src/                     # 🧠 Application Code
│   ├── data/                # Data loading and management
│   │   ├── data_loader.py   # CSV reading, company data loading
│   │   └── sic_hierarchy.py # SIC hierarchy management
│   ├── preprocessing/       # Text preprocessing and features
│   │   ├── text_utils.py    # Text preprocessing utilities
│   │   └── company_features.py # Company representation creation
│   ├── classifiers/         # Classification algorithms
│   │   ├── base_classifier.py  # Abstract base class
│   │   ├── semantic_classifier.py    # Comprehensive semantic embeddings
│   │   └── weighted_classifier.py    # Tag-focused weighted approach
│   ├── utils/               # Utility functions
│   │   └── formatting.py    # SIC code formatting utilities
│   └── main.py              # Main orchestrator
├── data/                    # 📊 Data Files
│   ├── input/               # Input data files
│   │   ├── ml_insurance_challenge.csv   # Company data
│   │   ├── sic_codes.csv               # SIC hierarchy
│   │   └── insurance_taxonomy.csv      # Challenge labels
│   ├── output/              # Generated classification results
│   └── cache_backup/        # Cached embeddings backup
│       ├── embeddings/      # Model embeddings cache
│       └── naics_mappings/  # NAICS validation cache
├── venv/                    # 🐍 Virtual environment
├── requirements.txt         # 📦 Dependencies
└── README.md               # 📖 Documentation
```

## 🚀 Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Run the demo
python demo.py

# Or run the module directly
python -m src.main
```

## 📊 Classification Methods

### 1. **Semantic Classifier**
- Uses comprehensive company representation (description + tags + sector + category + niche)
- Sentence transformer embeddings with cosine similarity
- Good baseline performance

### 2. **Weighted Classifier** ⭐ **Best Performance**
- Prioritizes business tags (40%) over other features
- Weights: Tags=40%, Description=30%, Category=20%, Niche=10%
- **Overall confidence: 0.445** (vs 0.340 for semantic)

## 🎯 Results Summary

Based on 10-company sample testing:

| Method    | Avg Division | Avg Major Group | Avg Industry Group | **Overall** |
|-----------|-------------|-----------------|-------------------|-------------|
| Semantic  | 0.290       | 0.332          | 0.398             | **0.340**   |
| Weighted  | 0.399       | 0.441          | 0.494             | **0.445**   |

**Winner: Weighted Classifier** - 31% better overall performance

## 📁 Data Files

- `data/input/ml_insurance_challenge.csv` - Company data for classification
- `data/input/sic_codes.csv` - SIC code hierarchy and descriptions
- `data/input/insurance_taxonomy.csv` - Challenge taxonomy labels

## 🔧 Key Features

- **Clean Code Architecture**: Modular, testable, maintainable
- **Multiple Classification Methods**: Semantic and weighted approaches
- **Hierarchical Classification**: Division → Major Group → Industry Group → SIC Code
- **Proper CSV Handling**: Handles multiline fields correctly
- **Comprehensive Testing**: Sample testing with confidence metrics

## 🎯 Next Steps

1. **TF-IDF SIC→Label Mapping**: Use SIC descriptions to map to challenge labels
2. **OSHA SIC Manual Integration**: Acquire comprehensive SIC descriptions
3. **Label Classification**: Map companies to specific business activity labels

## 🧪 Development

The system follows clean code principles:
- **Single Responsibility**: Each module has one clear purpose
- **DRY**: No code duplication
- **Testable**: Modular design enables easy testing
- **Extensible**: Easy to add new classification methods

---

**Built with:** sentence-transformers, pandas, scikit-learn
**Validation:** External NAICS industry classification system 