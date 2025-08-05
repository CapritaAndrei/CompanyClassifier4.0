#!/usr/bin/env python3
"""
Apply Auto-Fix Candidates: Apply the 14 high-confidence label mappings
"""

import pandas as pd
import json


def load_auto_fix_candidates():
    """Load the auto-fix candidates"""
    print("📊 LOADING AUTO-FIX CANDIDATES")
    print("=" * 60)
    
    with open('data/processed/auto_fix_candidates.json', 'r') as f:
        auto_fixes = json.load(f)
    
    print(f"Found {len(auto_fixes)} high-confidence auto-fix candidates:")
    for old_label, new_label in auto_fixes.items():
        print(f"  '{old_label}' → '{new_label}'")
    
    return auto_fixes


def apply_auto_fixes():
    """Apply the auto-fix candidates to training data"""
    print(f"\n🔧 APPLYING AUTO-FIXES")
    print("=" * 60)
    
    # Load auto-fix candidates
    auto_fixes = load_auto_fix_candidates()
    
    # Load current training data (might be the fixed version)
    try:
        training_df = pd.read_csv('data/processed/training_data_fixed.csv')
        print("Using previously fixed training data...")
    except:
        training_df = pd.read_csv('data/processed/training_data.csv')
        print("Using original training data...")
    
    # Apply auto-fixes
    total_companies_affected = 0
    for old_label, new_label in auto_fixes.items():
        mask = training_df['primary_label'] == old_label
        companies_affected = mask.sum()
        
        if companies_affected > 0:
            total_companies_affected += companies_affected
            training_df.loc[mask, 'primary_label'] = new_label
            print(f"✅ '{old_label}' → '{new_label}' ({companies_affected} companies)")
        else:
            print(f"⚠️  '{old_label}' not found in data")
    
    print(f"\nTotal companies affected by auto-fixes: {total_companies_affected}")
    
    # Save the auto-fixed data
    output_path = 'data/processed/training_data_auto_fixed.csv'
    training_df.to_csv(output_path, index=False)
    print(f"Saved auto-fixed data to: {output_path}")
    
    # Check improvement
    print(f"\n📊 IMPROVEMENT CHECK")
    print("=" * 60)
    
    # Load taxonomy for comparison
    taxonomy_df = pd.read_csv('data/input/insurance_taxonomy - insurance_taxonomy.csv')
    taxonomy_labels = set(taxonomy_df['label'].str.strip())
    
    # Check match rate
    training_labels = set(training_df['primary_label'].unique())
    perfect_matches = training_labels.intersection(taxonomy_labels)
    match_rate = len(perfect_matches) / len(training_labels)
    
    print(f"Perfect taxonomy matches: {len(perfect_matches)}/{len(training_labels)} ({match_rate:.1%})")
    print(f"Remaining problematic labels: {len(training_labels) - len(perfect_matches)}")
    
    return training_df, len(training_labels) - len(perfect_matches)


if __name__ == "__main__":
    print("🔧 APPLYING HIGH-CONFIDENCE AUTO-FIXES")
    print("=" * 80)
    
    fixed_df, remaining_problems = apply_auto_fixes()
    
    print(f"\n✅ AUTO-FIXES COMPLETE!")
    print(f"Remaining problematic labels: {remaining_problems}")
    print(f"Ready for DeepSeek re-classification on remaining cases.") 