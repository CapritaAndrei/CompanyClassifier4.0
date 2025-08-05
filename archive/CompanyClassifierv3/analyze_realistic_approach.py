#!/usr/bin/env python3
"""
Analyze what's realistically possible with 908 labels and 2,378 examples
"""

import pandas as pd
import numpy as np
from collections import Counter

def analyze_feasibility():
    """Analyze the mathematical feasibility of the classification task"""
    
    # Load the data
    df = pd.read_csv('data/processed/training_data.csv')
    
    print("🔍 CLASSIFICATION FEASIBILITY ANALYSIS")
    print("=" * 60)
    
    # Basic stats
    n_examples = len(df)
    n_unique_labels = df['primary_label'].nunique()
    
    print(f"Total examples: {n_examples}")
    print(f"Unique labels: {n_unique_labels}")
    print(f"Average examples per label: {n_examples/n_unique_labels:.1f}")
    
    # Label distribution
    label_counts = df['primary_label'].value_counts()
    
    print(f"\n📊 LABEL DISTRIBUTION:")
    print(f"Labels with 1 example: {(label_counts == 1).sum()} ({(label_counts == 1).sum()/n_unique_labels:.1%})")
    print(f"Labels with 2-5 examples: {((label_counts >= 2) & (label_counts <= 5)).sum()}")
    print(f"Labels with 6-10 examples: {((label_counts >= 6) & (label_counts <= 10)).sum()}")
    print(f"Labels with 10+ examples: {(label_counts >= 10).sum()}")
    
    # What's realistically learnable?
    learnable_threshold = 10  # Need at least 10 examples to learn reasonably
    learnable_labels = label_counts[label_counts >= learnable_threshold]
    
    print(f"\n🎯 LEARNABLE LABELS (10+ examples):")
    print(f"Count: {len(learnable_labels)} labels ({len(learnable_labels)/n_unique_labels:.1%} of all labels)")
    print(f"Examples covered: {learnable_labels.sum()} ({learnable_labels.sum()/n_examples:.1%} of data)")
    
    print(f"\n🏆 TOP 20 MOST COMMON LABELS:")
    for label, count in label_counts.head(20).items():
        print(f"  {label}: {count} examples")
    
    # Realistic approaches
    print(f"\n💡 REALISTIC APPROACHES:")
    print(f"\n1. REDUCE LABEL SPACE:")
    print(f"   • Focus on top {len(learnable_labels)} labels with 10+ examples")
    print(f"   • Group similar labels (e.g., all 'Software X' → 'Software Services')")
    print(f"   • Create an 'Other' category for rare labels")
    
    print(f"\n2. HIERARCHICAL CLASSIFICATION:")
    print(f"   • First classify into broad domains (10-20 categories)")
    print(f"   • Then classify within domain if enough data")
    
    print(f"\n3. SIMILARITY-BASED APPROACH:")
    print(f"   • Don't predict exact label")
    print(f"   • Find most similar companies from training data")
    print(f"   • Return top-k similar companies' labels")
    
    # Group by domains for hierarchical approach
    print(f"\n🏢 DOMAIN-BASED GROUPING:")
    
    # Simple domain extraction based on common patterns
    domain_mapping = {
        'software': 'Technology',
        'financial': 'Financial Services',
        'manufacturing': 'Manufacturing',
        'retail': 'Retail & Commerce',
        'food': 'Food & Restaurant',
        'medical': 'Healthcare',
        'construction': 'Construction',
        'transport': 'Transportation',
        'insurance': 'Insurance',
        'real estate': 'Real Estate'
    }
    
    # Count domains
    domain_counts = Counter()
    for label in df['primary_label']:
        if pd.notna(label):
            label_lower = str(label).lower()
            domain_found = False
            for keyword, domain in domain_mapping.items():
                if keyword in label_lower:
                    domain_counts[domain] += 1
                    domain_found = True
                    break
            if not domain_found:
                domain_counts['Other'] += 1
    
    print(f"Broad domain distribution:")
    for domain, count in domain_counts.most_common():
        print(f"  {domain}: {count} examples")
    
    return learnable_labels


def create_reduced_label_mapping(df, min_examples=10):
    """Create a mapping to reduce label space to learnable labels"""
    
    label_counts = df['primary_label'].value_counts()
    
    # Labels with enough examples
    keep_labels = set(label_counts[label_counts >= min_examples].index)
    
    # Create mapping
    label_mapping = {}
    for label in df['primary_label'].unique():
        if pd.notna(label):
            if label in keep_labels:
                label_mapping[label] = label
            else:
                # Map to similar common label or 'Other'
                label_mapping[label] = 'Other - Specialized Service'
    
    print(f"\n📋 LABEL REDUCTION:")
    print(f"Original labels: {len(label_mapping)}")
    print(f"Reduced to: {len(set(label_mapping.values()))} labels")
    
    return label_mapping


if __name__ == "__main__":
    learnable_labels = analyze_feasibility()
    
    # Load data and create reduced mapping
    df = pd.read_csv('data/processed/training_data.csv')
    label_mapping = create_reduced_label_mapping(df)
    
    print(f"\n🎯 RECOMMENDATION:")
    print(f"The current approach of predicting 908 labels is mathematically infeasible.")
    print(f"With 538 single-example labels, the model is just memorizing, not learning.")
    print(f"\nInstead, consider:")
    print(f"1. Use only the {len(learnable_labels)} labels with 10+ examples")
    print(f"2. Implement hierarchical classification (domain → specific label)")
    print(f"3. Use similarity search instead of classification")
    print(f"4. Collect more data for important labels (at least 50-100 examples each)") 