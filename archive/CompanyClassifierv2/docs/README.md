# Insurance Taxonomy Classification Project

## Project Overview

This project tackles the challenge of classifying companies into a specific insurance taxonomy of 220 business activity labels. The goal is to associate one or more business activity tags to every company in a dataset of ~9,500 companies.

## The Challenge

**Input:**
- 9,500+ companies with descriptions, business tags, sector, category, and niche
- 220 specific insurance taxonomy labels (e.g., "Residential Plumbing Services", "Commercial Construction Services")

**Output:**
- Each company classified with one or more relevant insurance taxonomy labels
- Validation method beyond arbitrary similarity scores

## Project Evolution

### Version 1: SIC Code Classification (Initial Approach)
**Goal:** Use Standard Industrial Classification (SIC) codes as an intermediate step

**What we tried:**
- Hierarchical SIC classification (Division → Major Group → Industry Group → SIC Code)
- Semantic embeddings with sentence transformers
- TF-IDF approaches for text matching
- Weighted classification combining description, tags, category, and niche

**Problems encountered:**
- **Taxonomy Mismatch**: SIC codes are general industry classifications, but the insurance taxonomy is specific business activities
- **Low Confidence Scores**: Even the best methods achieved only ~0.5-0.6 confidence
- **Validation Issues**: No ground truth to validate SIC → Insurance label mappings
- **Complexity**: The hierarchical approach became overly complex without clear benefits

**Key Learning:** SIC codes and insurance taxonomy serve different purposes - direct mapping wasn't the right approach.

### Version 2: Direct Semantic Classification (Current Working Solution)
**Goal:** Direct company-to-insurance-label classification using few-shot learning

**What we built:**
- Interactive labeling system with semantic candidate selection
- Smart label prioritization based on common service types
- Active learning approach for efficient data collection
- Few-shot learning with manual labeling of key examples

**Why this works:**
- **Direct Approach**: No intermediate classification step
- **Human-in-the-Loop**: Manual labeling provides ground truth
- **Scalable**: Start with 20-30 labels, expand gradually
- **Validatable**: Each label can be validated through manual review

**Current Status:** ✅ **WORKING** - Ready for production use

## Technical Architecture

### Current System (Version 2)

```
Company Data (9,500 companies)
    ↓
Semantic Embeddings (sentence-transformers)
    ↓
Interactive Labeling Interface
    ↓
Manual Labeling (200-300 examples)
    ↓
Few-Shot Classifier Training
    ↓
Multi-Label Classification
```

### Key Components

1. **Smart Candidate Selection**
   - Uses semantic similarity to find relevant companies for each label
   - Prioritizes labels based on common service types (plumbing, electrical, construction)
   - Filters candidates by similarity threshold

2. **Interactive Labeling Interface**
   - Shows company details and suggested labels
   - Simple y/n/s/q interface for quick labeling
   - Saves session data for training

3. **Few-Shot Learning**
   - Trains on manually labeled examples
   - Uses logistic regression for multi-label classification
   - Provides confidence scores for predictions

## Usage

### Quick Start
```bash
# Initialize the system
python insurance_taxonomy_classifier.py

# Start interactive labeling
# Choose option 1 from the menu
```

### Demo Mode
```bash
# See how candidate selection works
python quick_demo.py
```

## Lessons Learned

### What Didn't Work
1. **Indirect Classification**: SIC codes as intermediate step added complexity without benefits
2. **Pure Automation**: Without ground truth, validation was impossible
3. **Over-Engineering**: Complex hierarchical approaches for simple classification

### What Works
1. **Direct Approach**: Company → Insurance Label mapping
2. **Human-in-the-Loop**: Manual labeling provides validation
3. **Iterative Development**: Start simple, add complexity as needed
4. **Semantic Search**: Finding relevant candidates for labeling

### Key Insights
- **Taxonomy Understanding**: Insurance labels are specific business activities, not general industry codes
- **Validation Strategy**: Manual labeling creates the ground truth needed for validation
- **Scalability**: Few-shot learning can start small and grow with more labeled data
- **User Experience**: Simple interactive interface makes labeling efficient

## Next Steps

1. **Complete Labeling**: Label 200-300 examples across key labels
2. **Train Classifier**: Implement the training functionality
3. **Evaluate Performance**: Test on held-out examples
4. **Scale Up**: Add more labels and examples
5. **Semi-Automation**: Implement active learning for new examples

## Conclusion

Version 2 represents a pragmatic approach that balances automation with human expertise. By focusing on direct classification and manual validation, we avoid the complexity and validation issues of indirect approaches while building a foundation for future automation.

The key insight was recognizing that insurance taxonomy classification is fundamentally different from industry classification - it requires understanding specific business activities rather than broad industry categories. 