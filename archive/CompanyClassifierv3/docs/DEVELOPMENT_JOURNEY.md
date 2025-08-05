# Development Journey: From NAICS Mapping to BEACON Solution

## Overview

This document chronicles the complete development journey from initial concept to final working solution. The project evolved through **4 major versions**, each addressing different challenges and learning from previous failures.

**Final Solution**: BEACON-based NAICS prediction with Master Map insurance label mapping

## The Problem

**Original Challenge**: Classify 9,494 companies into 220 insurance taxonomy labels using only business descriptions and metadata.

**Core Constraint**: No existing training data for insurance labels - only business descriptions and NAICS codes.

## Version 1: Direct API Mapping (Failed)

### Approach
- Use DeepSeek API to directly map insurance labels to NAICS codes
- Generate training data by mapping NAICS → Insurance labels
- Train classifier on mapped data

### Problems Encountered
1. **API Hallucination**: DeepSeek generated fake NAICS descriptions
2. **Inconsistent Mappings**: Same label mapped to different codes across runs
3. **Low Coverage**: Only ~35% of labels successfully mapped
4. **Quality Issues**: Many mappings were semantically incorrect

### Why It Failed
- **No Ground Truth**: No way to validate API-generated mappings
- **Unreliable Source**: API responses were inconsistent and often incorrect
- **Scalability Issues**: Manual review of 220 labels × multiple NAICS codes each

### Lessons Learned
- External APIs cannot be trusted for critical mapping tasks
- Need validation mechanism for generated mappings
- Direct mapping approach too fragile for production use

---

## Version 2: Hierarchical NAICS Mapping (Partial Success)

### Approach
- Map insurance labels to higher-level NAICS categories (sectors, subsectors)
- Use comprehensive NAICS hierarchy data (2-6 digit codes)
- Interactive approval for inclusion matches

### Improvements Over Version 1
1. **Better Coverage**: Hierarchical mapping increased coverage significantly
2. **Interactive Quality Control**: Manual approval prevented bad mappings
3. **Structured Data**: Used official NAICS hierarchy files

### Problems Encountered
1. **Still Limited Coverage**: Only ~26 labels successfully mapped
2. **Manual Process**: Required human intervention for each mapping
3. **Semantic Gaps**: Many insurance labels didn't align with NAICS hierarchy

### Why It Was Limited
- **Hierarchical Mismatch**: Insurance taxonomy ≠ NAICS hierarchy
- **Manual Bottleneck**: Couldn't scale beyond ~26 labels
- **Semantic Differences**: Insurance labels often more specific than NAICS sectors

### Lessons Learned
- Hierarchical approach better than direct API mapping
- Interactive quality control essential
- Need to bridge semantic gap between taxonomies

---

## Version 3: Embedding-Based Semantic Mapping (Major Progress)

### Approach
- Use sentence embeddings to find semantic similarities
- Compare insurance labels to NAICS descriptions
- Interactive approval with similarity scores
- Caching for efficiency

### Major Improvements
1. **Semantic Understanding**: Captured meaning beyond exact text matching
2. **Higher Coverage**: 58 labels mapped (vs 26 in Version 2)
3. **1,458 NAICS Codes**: Significant expansion of mapped codes
4. **Quality Control**: Interactive approval maintained quality

### Problems Encountered
1. **Still Manual**: Required human intervention for each mapping
2. **Threshold Tuning**: Finding right similarity threshold was challenging
3. **Coverage Ceiling**: Hit ~26% coverage limit

### Why It Hit Limits
- **Manual Bottleneck**: Couldn't scale beyond ~58 labels efficiently
- **Semantic Complexity**: Some insurance labels too specific for NAICS mapping
- **Time Constraints**: Manual mapping of 220 labels would take weeks

### Lessons Learned
- Semantic similarity much better than exact matching
- Interactive approval essential for quality
- Need automated approach to scale beyond manual limits

---

## Version 4: BEACON-Based Solution (SUCCESS)

### The Breakthrough
**Key Insight**: Instead of mapping insurance labels to NAICS, use existing BEACON model to predict NAICS from descriptions, then map NAICS to insurance labels.

### Approach
1. **Train BEACON Model**: Use 40k+ training examples (company descriptions → NAICS codes)
2. **Predict NAICS**: Apply BEACON to new company descriptions
3. **Map to Insurance**: Use Master Map (NAICS → Insurance labels)
4. **Generate Training Data**: Create massive labeled dataset

### Why This Worked

#### ✅ **Leveraged Existing Infrastructure**
- BEACON already trained on 41,918 examples
- Proven methodology from U.S. Census Bureau
- High accuracy on NAICS prediction

#### ✅ **Massive Training Data Generation**
- 27.8% of BEACON data mappable to insurance labels
- 11,646 training examples immediately available
- 248 examples per label on average

#### ✅ **Scalable Process**
- No manual intervention required
- Can process thousands of companies automatically
- Quality controlled through Master Map

#### ✅ **Immediate Results**
- 47 insurance labels covered immediately
- Working pipeline from description to insurance label
- Can expand coverage by adding more NAICS mappings

### Technical Implementation

```python
# The Complete Flow
Company Description → BEACON Model → NAICS Code → Master Map → Insurance Label

# Example
"Chemical manufacturing company" → 325211 → "Rubber Manufacturing"
```

### Coverage Analysis
- **BEACON NAICS Codes**: 1,057 unique codes
- **Master Map Coverage**: 421 mapped codes (22.3%)
- **Training Examples**: 11,646 mappable examples (27.8%)
- **Insurance Labels**: 47 labels with training data

---

## Key Insights and Lessons

### 1. **Problem Reframing**
- **Initial Approach**: "How do we map insurance labels to NAICS?"
- **Final Approach**: "How do we use NAICS prediction to generate insurance training data?"

### 2. **Leverage Existing Solutions**
- BEACON already solved the hard part (description → NAICS)
- Focus on the mapping layer (NAICS → Insurance)
- Don't reinvent what already works

### 3. **Quality Over Quantity**
- Interactive approval in Versions 2-3 ensured quality
- Master Map contains only validated mappings
- Better to have 47 high-quality labels than 220 uncertain ones

### 4. **Iterative Development**
- Each version built on lessons from previous
- Failures provided valuable insights
- Continuous improvement through experimentation

### 5. **Scalability Considerations**
- Manual processes don't scale (Versions 1-3)
- Automated pipelines essential for production
- Balance between quality and automation

---

## Final Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Company         │    │ BEACON       │    │ Master Map      │    │ Insurance       │
│ Description     │───▶│ Model        │───▶│ (NAICS →        │───▶│ Label           │
│                 │    │ (40k trained)│    │ Insurance)      │    │                 │
└─────────────────┘    └──────────────┘    └─────────────────┘    └─────────────────┘
```

### Components
1. **BEACON Model**: Pre-trained on 41,918 examples
2. **Master Map**: 421 validated NAICS → Insurance mappings
3. **Training Data**: 11,646 examples across 47 labels
4. **Production Pipeline**: End-to-end classification system

---

## Next Steps

### Immediate (Version 4.1)
1. **Apply to 9k Dataset**: Process all companies through BEACON pipeline
2. **Expand Master Map**: Add more NAICS codes for better coverage
3. **Train Final Classifier**: Use generated training data

### Future Enhancements
1. **Active Learning**: For labels not covered by Master Map
2. **Multi-label Support**: Handle companies with multiple insurance needs
3. **Confidence Scoring**: Add uncertainty estimates to predictions

---

## Conclusion

The journey from Version 1 to Version 4 demonstrates the importance of:
- **Iterative problem-solving**
- **Learning from failures**
- **Leveraging existing solutions**
- **Quality over quantity**
- **Scalable architectures**

The final BEACON-based solution provides a robust, scalable, and immediately usable system for insurance classification, with clear paths for expansion and improvement. 