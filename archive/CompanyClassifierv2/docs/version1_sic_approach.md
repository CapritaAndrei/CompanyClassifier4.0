# Version 1: SIC Code Classification Approach

## Overview

The initial approach attempted to use Standard Industrial Classification (SIC) codes as an intermediate step between company descriptions and insurance taxonomy labels. The idea was to classify companies into SIC codes first, then map SIC codes to insurance labels.

## What We Built

### 1. Hierarchical SIC Classification System

We implemented a multi-level classification system:

```
Company Description → Division → Major Group → Industry Group → SIC Code
```

**Example Flow:**
- Company: "Welchcivils is a civil engineering and construction company..."
- Division: C - Construction (confidence: 0.374)
- Major Group: 16 - Heavy Construction (confidence: 0.415)  
- Industry Group: 162 - Heavy Construction except Highway (confidence: 0.502)
- SIC Code: 1629 - Heavy Construction, Not Elsewhere Classified

### 2. Multiple Classification Methods

We tested several approaches:

#### A. Comprehensive Semantic Embeddings
- Used sentence-transformers ('all-MiniLM-L6-v2')
- Combined company description, tags, sector, category, niche
- Achieved confidence scores: ~0.4-0.5

#### B. TF-IDF Approach
- Term Frequency-Inverse Document Frequency for keyword matching
- Poor performance on company descriptions due to "messy" text
- Achieved confidence scores: ~0.05-0.1

#### C. Tag-Focused Matching
- Used only business tags for classification
- Better performance than TF-IDF but still limited
- Achieved confidence scores: ~0.4-0.5

#### D. Weighted Classification
- Combined multiple features with configurable weights
- Tags: 40%, Description: 30%, Category: 20%, Niche: 10%
- Best performance: ~0.5-0.6 confidence

### 3. SIC → Insurance Label Mapping

We attempted to bridge SIC codes to insurance labels using:

- **TF-IDF Mapping**: Find similar insurance labels for each SIC code
- **Semantic Similarity**: Compare SIC descriptions to insurance label descriptions
- **Hierarchical TF-IDF**: Multi-step classification through SIC hierarchy

## Problems Encountered

### 1. Taxonomy Mismatch

**The Core Issue:**
- SIC codes classify by **industry** (e.g., "Construction")
- Insurance labels classify by **business activity** (e.g., "Residential Plumbing Services")

**Example:**
- Company: "ABC Plumbing Company"
- SIC: 1711 - Plumbing, Heating, and Air-Conditioning
- Insurance Labels: "Residential Plumbing Services", "Commercial Plumbing Services", "HVAC Installation and Service"

The SIC code is too broad to map to specific insurance activities.

### 2. Low Confidence Scores

Even the best methods achieved only moderate confidence:
- Comprehensive Semantic: 0.430 average
- Tag-Focused: 0.514 average  
- Weighted: 0.559 average

These scores weren't high enough for reliable classification.

### 3. Validation Issues

**The Fundamental Problem:**
- No ground truth data linking SIC codes to insurance labels
- External mappings (like NCCI Workers Comp) were for different taxonomies
- Couldn't validate if SIC → Insurance mappings were correct

**Example Validation Challenge:**
```
SIC 1711 (Plumbing) → "Residential Plumbing Services"?
SIC 1711 (Plumbing) → "Commercial Plumbing Services"?  
SIC 1711 (Plumbing) → "HVAC Installation and Service"?
```

Without labeled examples, we couldn't determine which mappings were correct.

### 4. Complexity Without Benefits

The hierarchical approach added significant complexity:
- 4-step classification process (Division → Major Group → Industry Group → SIC)
- Multiple embedding models and similarity calculations
- Complex weighting schemes and parameter tuning
- Large codebase that was difficult to maintain

**Result:** More complexity, no clear improvement in accuracy.

## Technical Challenges

### 1. CSV Data Issues
- Complex CSV parsing with multi-line fields
- Quoting issues and encoding problems
- Had to switch from pandas to csv.DictReader for reliable parsing

### 2. Embedding Performance
- Large embedding matrices for 1000+ SIC codes
- Memory and computation overhead
- Slow processing for large datasets

### 3. Hierarchical TF-IDF Issues
- Sparse data at higher hierarchy levels
- `ValueError: max_df corresponds to < documents than min_df`
- Had to implement dynamic TF-IDF parameters

### 4. Misclassification Examples

**Landscaping Services Problem:**
- Company: "Landscaping Services"
- Expected: Division A (Agriculture)
- Actual: Division I (Services) - due to generic "services" keyword

This highlighted the limitations of keyword-based approaches.

## Code Evolution

### Initial Monolithic Script
Started with `final_sic_classifier.py` (924 lines) containing all logic.

### Refactoring to Modular Architecture
Split into `src/` directory with:
- `src/data/` - Data loading and SIC hierarchy
- `src/preprocessing/` - Text processing and company features  
- `src/classifiers/` - Different classification approaches
- `src/utils/` - Helper functions

### Multiple Classifier Implementations
- `semantic_classifier.py` - Pure semantic embeddings
- `weighted_classifier.py` - Multi-feature weighted approach
- `tfidf_sic_classifier.py` - TF-IDF SIC → Label mapping
- `hierarchical_tfidf_classifier.py` - Multi-step hierarchical TF-IDF

## Key Learnings

### 1. Understand the Problem Domain
- Insurance taxonomy ≠ Industry classification
- Specific business activities ≠ Broad industry categories
- Direct mapping approaches work better than indirect ones

### 2. Validation is Critical
- Without ground truth, you can't validate results
- External data sources may not match your taxonomy
- Manual validation is often necessary

### 3. Simplicity Over Complexity
- Complex approaches don't always yield better results
- Start simple, add complexity only when needed
- Focus on the core problem, not intermediate steps

### 4. Data Quality Matters
- "Messy" company descriptions require robust preprocessing
- Business tags are often more reliable than descriptions
- Different data formats require flexible parsing

## Why We Abandoned This Approach

### 1. No Clear Path to Validation
Without ground truth data, we couldn't determine if our SIC → Insurance mappings were correct.

### 2. Poor Performance
Even the best methods achieved only moderate confidence scores, not high enough for reliable classification.

### 3. Over-Engineering
The hierarchical approach added significant complexity without clear benefits.

### 4. Taxonomy Mismatch
The fundamental mismatch between SIC codes (industry classification) and insurance labels (business activities) made the approach inherently flawed.

## Transition to Version 2

The failure of Version 1 led to a key insight: **direct classification** is better than **indirect classification**. Instead of:

```
Company → SIC Code → Insurance Label
```

We should do:

```
Company → Insurance Label
```

This insight led to the development of Version 2: the few-shot learning approach with direct company-to-label classification. 