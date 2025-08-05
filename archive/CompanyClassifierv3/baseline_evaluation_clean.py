#!/usr/bin/env python3
"""
Baseline Evaluation on Clean Examples
Evaluate hierarchical vs similarity classifiers on the 1,037 clean examples with valid taxonomy labels
"""

import pandas as pd
import numpy as np
from similarity_based_classifier import SimilarityBasedClassifier
from hierarchical_classifier import HierarchicalClassifier
import time
import sys
sys.path.append('src')


def load_clean_examples():
    """Load the 1,037 companies with valid taxonomy labels"""
    
    # Load the training data
    training_df = pd.read_csv('data/processed/training_data_auto_fixed.csv')
    
    # Load taxonomy to identify valid labels
    taxonomy_df = pd.read_csv('data/input/insurance_taxonomy - insurance_taxonomy.csv')
    taxonomy_labels = set(taxonomy_df['label'].str.strip())
    
    # Find companies with valid labels
    valid_mask = training_df['primary_label'].isin(taxonomy_labels)
    clean_companies = training_df[valid_mask].copy()
    
    print(f"📊 Loaded {len(clean_companies)} clean companies with valid taxonomy labels")
    print(f"   Unique labels: {clean_companies['primary_label'].nunique()}")
    print(f"   Sectors: {clean_companies['sector'].nunique()}")
    
    return clean_companies


def create_test_train_split(clean_df, test_ratio=0.2):
    """Create test/train split from clean examples"""
    
    # Use stratified split to ensure all labels are represented
    from collections import defaultdict
    
    # Group by label to ensure representation
    label_groups = defaultdict(list)
    for idx, row in clean_df.iterrows():
        label_groups[row['primary_label']].append(idx)
    
    test_indices = []
    train_indices = []
    
    for label, indices in label_groups.items():
        n_test = max(1, int(len(indices) * test_ratio))  # At least 1 for testing
        n_test = min(n_test, len(indices) - 1)  # Leave at least 1 for training
        
        # Randomly select test examples
        np.random.seed(42)
        test_idx = np.random.choice(indices, size=n_test, replace=False)
        train_idx = [idx for idx in indices if idx not in test_idx]
        
        test_indices.extend(test_idx)
        train_indices.extend(train_idx)
    
    test_df = clean_df.loc[test_indices].reset_index(drop=True)
    train_df = clean_df.loc[train_indices].reset_index(drop=True)
    
    print(f"📊 Train/Test Split:")
    print(f"   Training: {len(train_df)} companies ({len(train_df)/len(clean_df):.1%})")
    print(f"   Testing: {len(test_df)} companies ({len(test_df)/len(clean_df):.1%})")
    print(f"   Test labels: {test_df['primary_label'].nunique()}")
    
    return train_df, test_df


def evaluate_baseline_performance(train_df, test_df):
    """Evaluate both classifiers on the clean examples"""
    
    print(f"\n🧪 BASELINE EVALUATION ON CLEAN EXAMPLES")
    print("=" * 80)
    
    # Save training data temporarily
    temp_train_file = 'temp_baseline_train.csv'
    train_df.to_csv(temp_train_file, index=False)
    
    # Initialize both classifiers
    print("🔄 Initializing classifiers...")
    
    print("  Loading similarity-based classifier...")
    start_time = time.time()
    similarity_classifier = SimilarityBasedClassifier(temp_train_file)
    sim_init_time = time.time() - start_time
    
    print("  Loading hierarchical classifier...")
    start_time = time.time()
    hierarchical_classifier = HierarchicalClassifier(temp_train_file)
    hier_init_time = time.time() - start_time
    
    print(f"  Similarity init: {sim_init_time:.1f}s")
    print(f"  Hierarchical init: {hier_init_time:.1f}s")
    
    # Evaluate both approaches
    results = []
    total_tests = len(test_df)
    
    print(f"\n🧪 Testing both approaches on {total_tests} clean examples...")
    start_eval_time = time.time()
    
    for idx, row in test_df.iterrows():
        if idx % 50 == 0:
            print(f"  Progress: {idx}/{total_tests} ({idx/total_tests:.1%})")
        
        company_data = {
            'description': row['description'],
            'business_tags': row['business_tags'],
            'sector': row.get('sector', ''),
            'category': row.get('category', ''),
            'niche': row.get('niche', '')
        }
        
        true_label = row['primary_label']
        
        # Test similarity-based classifier
        sim_result = similarity_classifier.predict(company_data)
        sim_pred = sim_result['primary_prediction']
        sim_confidence = sim_result['primary_confidence']
        sim_top3 = [p['label'] for p in sim_result['predictions'][:3]]
        sim_top5 = [p['label'] for p in sim_result['predictions'][:5]]
        
        # Test hierarchical classifier
        hier_result = hierarchical_classifier.predict(company_data)
        hier_pred = hier_result['primary_prediction']
        hier_confidence = hier_result['primary_confidence']
        hier_domain = hier_result['predicted_domain']
        hier_domain_conf = hier_result['domain_confidence']
        hier_top3 = [p['label'] for p in hier_result['predictions'][:3]]
        hier_top5 = [p['label'] for p in hier_result['predictions'][:5]]
        
        # Get true domain for hierarchical
        true_domain = hierarchical_classifier.label_to_domain.get(true_label, 'Other Services')
        
        results.append({
            'true_label': true_label,
            'true_domain': true_domain,
            'sector': row.get('sector', ''),
            'category': row.get('category', ''),
            
            # Similarity results
            'sim_pred': sim_pred,
            'sim_confidence': sim_confidence,
            'sim_correct': sim_pred == true_label,
            'sim_top3_correct': true_label in sim_top3,
            'sim_top5_correct': true_label in sim_top5,
            
            # Hierarchical results
            'hier_pred': hier_pred,
            'hier_confidence': hier_confidence,
            'hier_domain': hier_domain,
            'hier_domain_conf': hier_domain_conf,
            'hier_correct': hier_pred == true_label,
            'hier_top3_correct': true_label in hier_top3,
            'hier_top5_correct': true_label in hier_top5,
            'hier_domain_correct': hier_domain == true_domain
        })
    
    eval_time = time.time() - start_eval_time
    
    # Calculate comprehensive metrics
    total = len(results)
    
    # Similarity metrics
    sim_correct = sum(1 for r in results if r['sim_correct'])
    sim_top3_correct = sum(1 for r in results if r['sim_top3_correct'])
    sim_top5_correct = sum(1 for r in results if r['sim_top5_correct'])
    
    # Hierarchical metrics
    hier_correct = sum(1 for r in results if r['hier_correct'])
    hier_top3_correct = sum(1 for r in results if r['hier_top3_correct'])
    hier_top5_correct = sum(1 for r in results if r['hier_top5_correct'])
    hier_domain_correct = sum(1 for r in results if r['hier_domain_correct'])
    
    # Print comprehensive results
    print(f"\n📊 BASELINE PERFORMANCE ON CLEAN EXAMPLES")
    print("=" * 80)
    print(f"Evaluation time: {eval_time:.1f} seconds ({eval_time/total:.2f}s per company)")
    
    print(f"\n🎯 PRIMARY ACCURACY (Exact Match):")
    print(f"  Similarity-Based:   {sim_correct/total:.1%} ({sim_correct}/{total})")
    print(f"  Hierarchical:       {hier_correct/total:.1%} ({hier_correct}/{total})")
    improvement = (hier_correct - sim_correct) / total * 100
    print(f"  Improvement:        {improvement:+.1f} percentage points")
    
    print(f"\n🎯 TOP-3 ACCURACY:")
    print(f"  Similarity-Based:   {sim_top3_correct/total:.1%} ({sim_top3_correct}/{total})")
    print(f"  Hierarchical:       {hier_top3_correct/total:.1%} ({hier_top3_correct}/{total})")
    
    print(f"\n🎯 TOP-5 ACCURACY:")
    print(f"  Similarity-Based:   {sim_top5_correct/total:.1%} ({sim_top5_correct}/{total})")
    print(f"  Hierarchical:       {hier_top5_correct/total:.1%} ({hier_top5_correct}/{total})")
    
    print(f"\n🎯 DOMAIN ACCURACY (Hierarchical):")
    print(f"  Domain Prediction:  {hier_domain_correct/total:.1%} ({hier_domain_correct}/{total})")
    
    # Performance by sector
    print(f"\n📊 PERFORMANCE BY SECTOR:")
    sectors = test_df['sector'].value_counts().head(5)
    for sector in sectors.index:
        sector_results = [r for r in results if r['sector'] == sector]
        if len(sector_results) >= 5:  # Only show sectors with enough examples
            sector_sim = sum(1 for r in sector_results if r['sim_correct']) / len(sector_results)
            sector_hier = sum(1 for r in sector_results if r['hier_correct']) / len(sector_results)
            print(f"  {sector[:25]:<25}: Sim {sector_sim:.1%} | Hier {sector_hier:.1%} ({len(sector_results)} examples)")
    
    # Show examples where hierarchical wins
    print(f"\n🏆 EXAMPLES WHERE HIERARCHICAL WINS:")
    hier_wins = [r for r in results if r['hier_correct'] and not r['sim_correct']][:3]
    
    for i, result in enumerate(hier_wins, 1):
        print(f"\n  Example {i}:")
        print(f"    True: {result['true_label']}")
        print(f"    Similarity → {result['sim_pred']} (conf: {result['sim_confidence']:.3f}) ❌")
        print(f"    Hierarchical → {result['hier_pred']} (conf: {result['hier_confidence']:.3f}) ✅")
        print(f"    Domain: {result['hier_domain']} ({'✅' if result['hier_domain_correct'] else '❌'})")
    
    # Show examples where similarity wins
    print(f"\n🎯 EXAMPLES WHERE SIMILARITY WINS:")
    sim_wins = [r for r in results if r['sim_correct'] and not r['hier_correct']][:3]
    
    for i, result in enumerate(sim_wins, 1):
        print(f"\n  Example {i}:")
        print(f"    True: {result['true_label']}")
        print(f"    Similarity → {result['sim_pred']} (conf: {result['sim_confidence']:.3f}) ✅")
        print(f"    Hierarchical → {result['hier_pred']} (conf: {result['hier_confidence']:.3f}) ❌")
    
    # Cleanup
    import os
    os.remove(temp_train_file)
    
    # Save detailed results
    results_df = pd.DataFrame(results)
    results_df.to_csv('data/processed/baseline_evaluation_results.csv', index=False)
    print(f"\n💾 Detailed results saved to: data/processed/baseline_evaluation_results.csv")
    
    return {
        'similarity_accuracy': sim_correct/total,
        'similarity_top3': sim_top3_correct/total,
        'similarity_top5': sim_top5_correct/total,
        'hierarchical_accuracy': hier_correct/total,
        'hierarchical_top3': hier_top3_correct/total,
        'hierarchical_top5': hier_top5_correct/total,
        'hierarchical_domain': hier_domain_correct/total,
        'improvement': improvement,
        'total_examples': total,
        'eval_time': eval_time
    }


def main():
    """Main baseline evaluation function"""
    print("📊 BASELINE EVALUATION ON CLEAN EXAMPLES")
    print("=" * 80)
    print("This will establish performance baselines before adding high-confidence reclassified data")
    print()
    
    # Load clean examples
    clean_df = load_clean_examples()
    
    # Create train/test split
    train_df, test_df = create_test_train_split(clean_df, test_ratio=0.3)  # Use 30% for testing
    
    # Run evaluation
    baseline_results = evaluate_baseline_performance(train_df, test_df)
    
    # Summary
    print(f"\n🎯 BASELINE SUMMARY:")
    print(f"   Dataset: {baseline_results['total_examples']} clean examples")
    print(f"   Similarity accuracy: {baseline_results['similarity_accuracy']:.1%}")
    print(f"   Hierarchical accuracy: {baseline_results['hierarchical_accuracy']:.1%}")
    print(f"   Improvement: {baseline_results['improvement']:+.1f} percentage points")
    print(f"   Domain accuracy: {baseline_results['hierarchical_domain']:.1%}")
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"1. Run overnight DeepSeek processing to get high-confidence reclassifications")
    print(f"2. Run create_high_quality_dataset.py to combine data")
    print(f"3. Re-run this evaluation on the expanded dataset")
    print(f"4. Compare performance improvement from additional high-quality data")
    
    return baseline_results


if __name__ == "__main__":
    baseline_results = main() 