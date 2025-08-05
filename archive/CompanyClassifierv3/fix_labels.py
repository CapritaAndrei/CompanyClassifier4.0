#!/usr/bin/env python3
"""
Smart Label Fixing: Remove number prefixes and carefully handle similar labels
"""

import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher
from collections import defaultdict


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


def fix_number_prefixes(training_df, taxonomy_labels):
    """Fix labels that are exact matches once number prefixes are removed"""
    print("\n🔧 FIXING NUMBER PREFIXES")
    print("=" * 60)
    
    # Pattern to match number prefixes like "100. ", "163. ", etc.
    number_prefix_pattern = r'^\d+\.\s*'
    
    prefix_fixes = {}
    fixed_count = 0
    
    unique_labels = training_df['primary_label'].unique()
    
    for label in unique_labels:
        # Check if label has number prefix
        if re.match(number_prefix_pattern, label):
            # Remove the number prefix
            cleaned_label = re.sub(number_prefix_pattern, '', label)
            
            # Check if cleaned version exists in taxonomy
            if cleaned_label in taxonomy_labels:
                prefix_fixes[label] = cleaned_label
                fixed_count += 1
                print(f"✅ '{label}' → '{cleaned_label}'")
    
    print(f"\nFixed {fixed_count} labels with number prefixes")
    return prefix_fixes


def fix_case_and_punctuation(training_df, taxonomy_labels, existing_fixes):
    """Fix case and punctuation differences"""
    print("\n🔧 FIXING CASE & PUNCTUATION")
    print("=" * 60)
    
    case_fixes = {}
    fixed_count = 0
    
    # Get labels not already fixed
    remaining_labels = [label for label in training_df['primary_label'].unique() 
                       if label not in existing_fixes]
    
    for train_label in remaining_labels:
        # Try different case variations
        variations = [
            train_label,
            train_label.lower(),
            train_label.upper(), 
            train_label.title(),
            train_label.replace('-', ' '),
            train_label.replace('_', ' '),
            train_label.strip()
        ]
        
        for variation in variations:
            if variation in taxonomy_labels:
                case_fixes[train_label] = variation
                fixed_count += 1
                print(f"✅ '{train_label}' → '{variation}'")
                break
    
    print(f"\nFixed {fixed_count} labels with case/punctuation issues")
    return case_fixes


def find_semantic_matches(training_df, taxonomy_labels, existing_fixes, similarity_threshold=0.95):
    """Find semantically similar labels with high threshold to avoid false positives"""
    print(f"\n🎯 FINDING SEMANTIC MATCHES (similarity > {similarity_threshold})")
    print("=" * 60)
    
    semantic_matches = {}
    
    # Get labels not already fixed
    remaining_labels = [label for label in training_df['primary_label'].unique() 
                       if label not in existing_fixes]
    
    print(f"Checking {len(remaining_labels)} remaining labels...")
    
    # High-confidence matches only
    high_confidence_matches = []
    
    for train_label in remaining_labels:
        best_match = None
        best_score = 0
        
        for tax_label in taxonomy_labels:
            score = SequenceMatcher(None, train_label.lower(), tax_label.lower()).ratio()
            
            if score > best_score and score > similarity_threshold:
                best_score = score
                best_match = tax_label
        
        if best_match:
            high_confidence_matches.append({
                'training': train_label,
                'taxonomy': best_match,
                'similarity': best_score
            })
    
    # Sort by similarity and show for review
    high_confidence_matches.sort(key=lambda x: x['similarity'], reverse=True)
    
    print(f"\nFound {len(high_confidence_matches)} high-confidence matches:")
    
    for i, match in enumerate(high_confidence_matches[:20], 1):
        print(f"{i:2d}. '{match['training']}'")
        print(f"    → '{match['taxonomy']}' (similarity: {match['similarity']:.3f})")
        
        # Auto-approve near-perfect matches (>0.98)
        if match['similarity'] > 0.98:
            semantic_matches[match['training']] = match['taxonomy']
            print(f"    ✅ AUTO-APPROVED")
        else:
            print(f"    ⚠️  NEEDS REVIEW")
    
    if len(high_confidence_matches) > 20:
        print(f"    ... and {len(high_confidence_matches) - 20} more")
    
    auto_approved = len([m for m in high_confidence_matches if m['similarity'] > 0.98])
    print(f"\nAuto-approved {auto_approved} near-perfect matches (>98% similarity)")
    
    return semantic_matches, high_confidence_matches


def apply_fixes(training_df, all_fixes):
    """Apply all label fixes to the training data"""
    print(f"\n🔄 APPLYING FIXES")
    print("=" * 60)
    
    # Create a copy of the dataframe
    fixed_df = training_df.copy()
    
    # Apply all fixes
    total_companies_affected = 0
    for old_label, new_label in all_fixes.items():
        mask = fixed_df['primary_label'] == old_label
        companies_affected = mask.sum()
        total_companies_affected += companies_affected
        
        fixed_df.loc[mask, 'primary_label'] = new_label
        print(f"'{old_label}' → '{new_label}' ({companies_affected} companies)")
    
    print(f"\nTotal companies affected: {total_companies_affected}")
    print(f"Unique labels before: {training_df['primary_label'].nunique()}")
    print(f"Unique labels after: {fixed_df['primary_label'].nunique()}")
    
    return fixed_df


def analyze_improvement(original_df, fixed_df, taxonomy_labels):
    """Analyze the improvement after fixes"""
    print(f"\n📊 IMPROVEMENT ANALYSIS")
    print("=" * 60)
    
    # Original stats
    orig_training_labels = set(original_df['primary_label'])
    orig_matches = orig_training_labels.intersection(taxonomy_labels)
    orig_match_rate = len(orig_matches) / len(orig_training_labels)
    
    # Fixed stats
    fixed_training_labels = set(fixed_df['primary_label'])
    fixed_matches = fixed_training_labels.intersection(taxonomy_labels)
    fixed_match_rate = len(fixed_matches) / len(fixed_training_labels)
    
    print(f"BEFORE FIXES:")
    print(f"  Perfect matches: {len(orig_matches)}/{len(orig_training_labels)} ({orig_match_rate:.1%})")
    
    print(f"\nAFTER FIXES:")
    print(f"  Perfect matches: {len(fixed_matches)}/{len(fixed_training_labels)} ({fixed_match_rate:.1%})")
    
    print(f"\nIMPROVEMENT:")
    print(f"  Match rate: {orig_match_rate:.1%} → {fixed_match_rate:.1%} (+{(fixed_match_rate-orig_match_rate):.1%})")
    print(f"  Labels reduced: {len(orig_training_labels)} → {len(fixed_training_labels)} (-{len(orig_training_labels)-len(fixed_training_labels)})")
    
    return {
        'orig_match_rate': orig_match_rate,
        'fixed_match_rate': fixed_match_rate,
        'improvement': fixed_match_rate - orig_match_rate
    }


def save_fixed_data(fixed_df, all_fixes):
    """Save the fixed training data and fix log"""
    print(f"\n💾 SAVING FIXED DATA")
    print("=" * 60)
    
    # Save fixed training data
    output_path = 'data/processed/training_data_fixed.csv'
    fixed_df.to_csv(output_path, index=False)
    print(f"Saved fixed training data to: {output_path}")
    
    # Save fix log
    fix_log_path = 'data/processed/label_fixes_applied.json'
    import json
    with open(fix_log_path, 'w') as f:
        json.dump(all_fixes, f, indent=2)
    print(f"Saved fix log to: {fix_log_path}")
    
    print(f"\n✅ Files saved! Use 'training_data_fixed.csv' for classification.")


def main():
    """Main label fixing workflow"""
    print("🔧 SMART LABEL FIXING")
    print("=" * 80)
    
    # Load data
    taxonomy_labels, training_df = load_data()
    
    # Step 1: Fix number prefixes (easy wins)
    prefix_fixes = fix_number_prefixes(training_df, taxonomy_labels)
    
    # Step 2: Fix case and punctuation
    case_fixes = fix_case_and_punctuation(training_df, taxonomy_labels, prefix_fixes)
    
    # Step 3: Find semantic matches (careful!)
    semantic_fixes, review_candidates = find_semantic_matches(
        training_df, taxonomy_labels, {**prefix_fixes, **case_fixes}
    )
    
    # Combine all fixes
    all_fixes = {**prefix_fixes, **case_fixes, **semantic_fixes}
    
    print(f"\n📋 SUMMARY OF FIXES:")
    print(f"  Number prefix fixes: {len(prefix_fixes)}")
    print(f"  Case/punctuation fixes: {len(case_fixes)}")
    print(f"  Semantic fixes (auto-approved): {len(semantic_fixes)}")
    print(f"  Total fixes: {len(all_fixes)}")
    
    # Apply fixes
    fixed_df = apply_fixes(training_df, all_fixes)
    
    # Analyze improvement
    improvement = analyze_improvement(training_df, fixed_df, taxonomy_labels)
    
    # Save results
    save_fixed_data(fixed_df, all_fixes)
    
    return fixed_df, all_fixes, improvement


if __name__ == "__main__":
    fixed_df, all_fixes, improvement = main() 