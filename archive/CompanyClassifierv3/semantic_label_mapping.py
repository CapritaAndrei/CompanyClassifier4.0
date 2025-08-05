#!/usr/bin/env python3
"""
Semantic Label Mapping: Use embeddings to find similar taxonomy labels
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json


def load_data():
    """Load taxonomy labels and problematic training labels"""
    print("📊 LOADING DATA FOR SEMANTIC MAPPING")
    print("=" * 60)
    
    # Load taxonomy labels
    taxonomy_df = pd.read_csv('data/input/insurance_taxonomy - insurance_taxonomy.csv')
    taxonomy_labels = list(taxonomy_df['label'].str.strip())
    
    # Load training data
    training_df = pd.read_csv('data/processed/training_data.csv')
    training_df = training_df.dropna(subset=['primary_label'])
    
    # Load the fixes that were applied
    try:
        with open('data/processed/label_fixes_applied.json', 'r') as f:
            applied_fixes = json.load(f)
    except:
        applied_fixes = {}
    
    # Get unique training labels
    all_training_labels = set(training_df['primary_label'].unique())
    
    # Remove labels that were already fixed
    remaining_labels = all_training_labels - set(applied_fixes.keys())
    
    # Remove labels that already match taxonomy perfectly
    taxonomy_set = set(taxonomy_labels)
    problematic_labels = list(remaining_labels - taxonomy_set)
    
    print(f"Taxonomy labels: {len(taxonomy_labels)}")
    print(f"Total training labels: {len(all_training_labels)}")
    print(f"Already fixed labels: {len(applied_fixes)}")
    print(f"Perfect matches: {len(remaining_labels & taxonomy_set)}")
    print(f"Problematic labels to map: {len(problematic_labels)}")
    
    return taxonomy_labels, problematic_labels, training_df


def create_embeddings(taxonomy_labels, problematic_labels):
    """Create embeddings for both taxonomy and problematic labels"""
    print(f"\n🧠 CREATING EMBEDDINGS")
    print("=" * 60)
    
    # Load sentence transformer
    print("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create embeddings
    print("Creating embeddings for taxonomy labels...")
    taxonomy_embeddings = model.encode(taxonomy_labels, show_progress_bar=True)
    
    print("Creating embeddings for problematic labels...")
    problematic_embeddings = model.encode(problematic_labels, show_progress_bar=True)
    
    print(f"✅ Created embeddings for {len(taxonomy_labels)} taxonomy + {len(problematic_labels)} problematic labels")
    
    return model, taxonomy_embeddings, problematic_embeddings


def find_similar_labels(taxonomy_labels, problematic_labels, taxonomy_embeddings, problematic_embeddings, top_k=10):
    """Find most similar taxonomy labels for each problematic label"""
    print(f"\n🎯 FINDING SEMANTIC SIMILARITIES (top {top_k})")
    print("=" * 60)
    
    # Calculate similarities
    similarities = cosine_similarity(problematic_embeddings, taxonomy_embeddings)
    
    # For each problematic label, find top-k most similar taxonomy labels
    mappings = []
    
    print(f"Analyzing {len(problematic_labels)} problematic labels...\n")
    
    for i, prob_label in enumerate(problematic_labels):
        # Get similarities for this problematic label
        label_similarities = similarities[i]
        
        # Get top-k indices
        top_indices = np.argsort(label_similarities)[-top_k:][::-1]
        
        # Create mapping
        similar_labels = []
        for idx in top_indices:
            similar_labels.append({
                'taxonomy_label': taxonomy_labels[idx],
                'similarity': float(label_similarities[idx])
            })
        
        mappings.append({
            'problematic_label': prob_label,
            'similar_labels': similar_labels
        })
        
        # Show first 20 for immediate analysis
        if i < 20:
            print(f"{i+1:2d}. PROBLEMATIC: '{prob_label}'")
            print(f"    TOP MATCHES:")
            for j, match in enumerate(similar_labels[:5], 1):
                print(f"      {j}. {match['taxonomy_label']} (similarity: {match['similarity']:.3f})")
            print()
    
    if len(problematic_labels) > 20:
        print(f"... showing first 20 of {len(problematic_labels)} problematic labels")
    
    return mappings


def analyze_patterns(mappings):
    """Analyze patterns in the mappings"""
    print(f"\n🔍 PATTERN ANALYSIS")
    print("=" * 60)
    
    # Analyze by similarity thresholds
    high_sim = [m for m in mappings if m['similar_labels'][0]['similarity'] > 0.8]
    medium_sim = [m for m in mappings if 0.6 < m['similar_labels'][0]['similarity'] <= 0.8]
    low_sim = [m for m in mappings if m['similar_labels'][0]['similarity'] <= 0.6]
    
    print(f"SIMILARITY DISTRIBUTION:")
    print(f"  High similarity (>0.8): {len(high_sim)} labels")
    print(f"  Medium similarity (0.6-0.8): {len(medium_sim)} labels")
    print(f"  Low similarity (<0.6): {len(low_sim)} labels")
    
    # Look for common problematic patterns
    print(f"\nCOMMON PROBLEMATIC PATTERNS:")
    
    # Count common words in problematic labels
    from collections import Counter
    all_words = []
    for mapping in mappings:
        words = mapping['problematic_label'].lower().split()
        all_words.extend(words)
    
    common_words = Counter(all_words).most_common(15)
    print(f"  Most common words in problematic labels:")
    for word, count in common_words:
        print(f"    '{word}': {count} times")
    
    # Show high-confidence mappings that might be auto-fixable
    print(f"\nHIGH-CONFIDENCE MAPPINGS (might auto-fix):")
    for i, mapping in enumerate(high_sim[:10], 1):
        best_match = mapping['similar_labels'][0]
        print(f"  {i:2d}. '{mapping['problematic_label']}'")
        print(f"      → '{best_match['taxonomy_label']}' (similarity: {best_match['similarity']:.3f})")
    
    return high_sim, medium_sim, low_sim


def show_examples_by_category(mappings):
    """Show examples categorized by likely issues"""
    print(f"\n📂 EXAMPLES BY CATEGORY")
    print("=" * 60)
    
    # Categorize issues
    categories = {
        'too_generic': [],
        'too_specific': [],
        'wrong_domain': [],
        'made_up': []
    }
    
    for mapping in mappings[:50]:  # Look at first 50
        prob_label = mapping['problematic_label'].lower()
        best_match = mapping['similar_labels'][0]
        
        # Simple categorization logic
        if 'services' in prob_label and 'services' not in best_match['taxonomy_label'].lower():
            categories['too_generic'].append(mapping)
        elif len(prob_label.split()) > 5:
            categories['too_specific'].append(mapping)
        elif best_match['similarity'] < 0.4:
            categories['made_up'].append(mapping)
        else:
            categories['wrong_domain'].append(mapping)
    
    for category, examples in categories.items():
        if examples:
            print(f"\n{category.upper().replace('_', ' ')} ({len(examples)} examples):")
            for i, mapping in enumerate(examples[:3], 1):
                best_match = mapping['similar_labels'][0]
                print(f"  {i}. '{mapping['problematic_label']}'")
                print(f"     → '{best_match['taxonomy_label']}' (sim: {best_match['similarity']:.3f})")


def save_mappings(mappings, high_sim):
    """Save the mappings for review"""
    print(f"\n💾 SAVING MAPPINGS")
    print("=" * 60)
    
    # Save all mappings
    mappings_path = 'data/processed/semantic_label_mappings.json'
    with open(mappings_path, 'w') as f:
        json.dump(mappings, f, indent=2)
    print(f"Saved all mappings to: {mappings_path}")
    
    # Save high-confidence candidates for auto-fixing
    auto_fix_candidates = {}
    for mapping in high_sim:
        if mapping['similar_labels'][0]['similarity'] > 0.85:  # Very high confidence
            auto_fix_candidates[mapping['problematic_label']] = mapping['similar_labels'][0]['taxonomy_label']
    
    auto_fix_path = 'data/processed/auto_fix_candidates.json'
    with open(auto_fix_path, 'w') as f:
        json.dump(auto_fix_candidates, f, indent=2)
    
    print(f"Saved {len(auto_fix_candidates)} auto-fix candidates to: {auto_fix_path}")
    
    return auto_fix_candidates


def main():
    """Main semantic mapping workflow"""
    print("🧠 SEMANTIC LABEL MAPPING")
    print("=" * 80)
    
    # Load data
    taxonomy_labels, problematic_labels, training_df = load_data()
    
    if len(problematic_labels) == 0:
        print("✅ No problematic labels found! All labels match taxonomy.")
        return
    
    # Create embeddings
    model, taxonomy_embeddings, problematic_embeddings = create_embeddings(
        taxonomy_labels, problematic_labels
    )
    
    # Find similarities
    mappings = find_similar_labels(
        taxonomy_labels, problematic_labels, 
        taxonomy_embeddings, problematic_embeddings
    )
    
    # Analyze patterns
    high_sim, medium_sim, low_sim = analyze_patterns(mappings)
    
    # Show examples by category
    show_examples_by_category(mappings)
    
    # Save results
    auto_fix_candidates = save_mappings(mappings, high_sim)
    
    print(f"\n✅ SEMANTIC MAPPING COMPLETE!")
    print(f"Found {len(auto_fix_candidates)} high-confidence auto-fix candidates")
    print(f"Review the patterns above to decide next steps.")
    
    return mappings, auto_fix_candidates


if __name__ == "__main__":
    mappings, auto_fix_candidates = main() 