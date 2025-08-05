#!/usr/bin/env python3
"""
Check Label Quality: Training Data vs Insurance Taxonomy
Analyze how well the training data labels match the 220 taxonomy labels
"""

import pandas as pd
import numpy as np
from collections import Counter
from difflib import SequenceMatcher
import re


def load_data():
    """Load taxonomy and training data"""
    print("📊 LOADING DATA")
    print("=" * 60)
    
    # Load taxonomy labels
    taxonomy_df = pd.read_csv('data/input/insurance_taxonomy - insurance_taxonomy.csv')
    taxonomy_labels = set(taxonomy_df['label'].str.strip())
    
    # Load training data
    training_df = pd.read_csv('data/processed/training_data.csv')
    training_df = training_df.dropna(subset=['primary_label'])
    
    print(f"Insurance taxonomy labels: {len(taxonomy_labels)}")
    print(f"Training data companies: {len(training_df)}")
    print(f"Unique training labels: {training_df['primary_label'].nunique()}")
    
    return taxonomy_labels, training_df


def analyze_label_matching(taxonomy_labels, training_df):
    """Analyze how training labels match taxonomy labels"""
    print("\n🔍 LABEL MATCHING ANALYSIS")
    print("=" * 60)
    
    training_labels = set(training_df['primary_label'].str.strip())
    
    # Perfect matches
    perfect_matches = training_labels.intersection(taxonomy_labels)
    
    # Training labels not in taxonomy
    training_only = training_labels - taxonomy_labels
    
    # Taxonomy labels not in training
    taxonomy_only = taxonomy_labels - training_labels
    
    print(f"Perfect matches: {len(perfect_matches)}/{len(training_labels)} training labels")
    print(f"Training labels not in taxonomy: {len(training_only)}")
    print(f"Taxonomy labels not used in training: {len(taxonomy_only)}")
    print(f"Match rate: {len(perfect_matches)/len(training_labels):.1%}")
    
    return perfect_matches, training_only, taxonomy_only


def find_similar_labels(training_only, taxonomy_labels, threshold=0.7):
    """Find potentially similar labels that might be variations"""
    print(f"\n🎯 FINDING SIMILAR LABELS (similarity > {threshold})")
    print("=" * 60)
    
    suggestions = []
    
    for train_label in training_only:
        best_match = None
        best_score = 0
        
        for tax_label in taxonomy_labels:
            # Calculate similarity
            score = SequenceMatcher(None, train_label.lower(), tax_label.lower()).ratio()
            
            if score > best_score and score > threshold:
                best_score = score
                best_match = tax_label
        
        if best_match:
            suggestions.append({
                'training_label': train_label,
                'suggested_taxonomy': best_match,
                'similarity': best_score
            })
    
    # Sort by similarity
    suggestions.sort(key=lambda x: x['similarity'], reverse=True)
    
    print(f"Found {len(suggestions)} potential matches:")
    for i, sugg in enumerate(suggestions[:20], 1):  # Show top 20
        print(f"{i:2d}. '{sugg['training_label']}'")
        print(f"    → '{sugg['suggested_taxonomy']}' (similarity: {sugg['similarity']:.3f})")
    
    if len(suggestions) > 20:
        print(f"    ... and {len(suggestions) - 20} more")
    
    return suggestions


def analyze_label_distribution(training_df):
    """Analyze distribution of labels in training data"""
    print(f"\n📈 LABEL DISTRIBUTION ANALYSIS") 
    print("=" * 60)
    
    label_counts = training_df['primary_label'].value_counts()
    
    print(f"Labels with 1 example: {sum(label_counts == 1)}")
    print(f"Labels with 2-5 examples: {sum((label_counts >= 2) & (label_counts <= 5))}")
    print(f"Labels with 6-10 examples: {sum((label_counts >= 6) & (label_counts <= 10))}")  
    print(f"Labels with 11+ examples: {sum(label_counts >= 11)}")
    
    print(f"\nTop 15 most common training labels:")
    for i, (label, count) in enumerate(label_counts.head(15).items(), 1):
        print(f"{i:2d}. {label}: {count} examples")
    
    print(f"\nLabels with only 1 example (first 10):")
    single_examples = label_counts[label_counts == 1]
    for i, (label, count) in enumerate(single_examples.head(10).items(), 1):
        print(f"{i:2d}. {label}")
    
    if len(single_examples) > 10:
        print(f"    ... and {len(single_examples) - 10} more")
    
    return label_counts


def check_taxonomy_coverage(taxonomy_labels, training_df):
    """Check which taxonomy labels are well covered in training"""
    print(f"\n🎯 TAXONOMY COVERAGE ANALYSIS")
    print("=" * 60)
    
    training_counts = training_df['primary_label'].value_counts()
    
    # Check coverage for each taxonomy label
    covered_labels = []
    uncovered_labels = []
    
    for tax_label in taxonomy_labels:
        if tax_label in training_counts:
            covered_labels.append((tax_label, training_counts[tax_label]))
        else:
            uncovered_labels.append(tax_label)
    
    covered_labels.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Taxonomy labels with training data: {len(covered_labels)}/{len(taxonomy_labels)}")
    print(f"Taxonomy labels without training data: {len(uncovered_labels)}")
    print(f"Coverage rate: {len(covered_labels)/len(taxonomy_labels):.1%}")
    
    print(f"\nWell-covered taxonomy labels:")
    for i, (label, count) in enumerate(covered_labels[:15], 1):
        print(f"{i:2d}. {label}: {count} examples")
    
    print(f"\nUncovered taxonomy labels (first 15):")
    for i, label in enumerate(uncovered_labels[:15], 1):
        print(f"{i:2d}. {label}")
    
    if len(uncovered_labels) > 15:
        print(f"    ... and {len(uncovered_labels) - 15} more")
    
    return covered_labels, uncovered_labels


def generate_recommendations(perfect_matches, training_only, taxonomy_only, suggestions):
    """Generate recommendations for improving label quality"""
    print(f"\n💡 RECOMMENDATIONS")
    print("=" * 60)
    
    print("1. IMMEDIATE ACTIONS:")
    if suggestions:
        print(f"   • Review {len(suggestions)} potential label mappings")
        print(f"   • Consider standardizing training labels to match taxonomy")
    
    print(f"   • {len(training_only)} training labels need taxonomy mapping")
    print(f"   • {len(taxonomy_only)} taxonomy labels have no training examples")
    
    print("\n2. CLASSIFICATION IMPACT:")
    match_rate = len(perfect_matches) / (len(perfect_matches) + len(training_only))
    print(f"   • Current match rate: {match_rate:.1%}")
    
    if match_rate < 0.8:
        print("   • ⚠️  Low match rate may hurt classification performance")
        print("   • Consider label standardization before training classifiers")
    else:
        print("   • ✅ Good match rate - ready for classification")
    
    print("\n3. NEXT STEPS:")
    print("   • Fix obvious label mismatches using similarity suggestions")
    print("   • Decide whether to collect more data for uncovered taxonomy labels")
    print("   • Test hierarchical classifier with current labels")


def main():
    """Main analysis workflow"""
    print("🏷️  LABEL QUALITY CHECK: TRAINING vs TAXONOMY")
    print("=" * 80)
    
    # Load data
    taxonomy_labels, training_df = load_data()
    
    # Analyze matching
    perfect_matches, training_only, taxonomy_only = analyze_label_matching(taxonomy_labels, training_df)
    
    # Find similar labels
    suggestions = find_similar_labels(training_only, taxonomy_labels)
    
    # Analyze distribution
    label_counts = analyze_label_distribution(training_df)
    
    # Check taxonomy coverage
    covered_labels, uncovered_labels = check_taxonomy_coverage(taxonomy_labels, training_df)
    
    # Generate recommendations
    generate_recommendations(perfect_matches, training_only, taxonomy_only, suggestions)
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    
    return {
        'taxonomy_labels': len(taxonomy_labels),
        'training_companies': len(training_df),
        'perfect_matches': len(perfect_matches),
        'training_only': len(training_only),
        'taxonomy_only': len(taxonomy_only),
        'suggestions': len(suggestions),
        'match_rate': len(perfect_matches) / (len(perfect_matches) + len(training_only))
    }


if __name__ == "__main__":
    results = main() 