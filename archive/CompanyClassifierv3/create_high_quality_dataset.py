#!/usr/bin/env python3
"""
High-Quality Dataset Creator
Combines existing valid labels with high-confidence DeepSeek reclassification results
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List


def load_deepseek_results(results_file: str = "data/processed/deepseek_reclassification_results_final.csv") -> pd.DataFrame:
    """Load DeepSeek reclassification results"""
    
    if not Path(results_file).exists():
        print(f"❌ DeepSeek results file not found: {results_file}")
        print(f"Make sure the overnight processing has completed.")
        return None
    
    results_df = pd.read_csv(results_file)
    print(f"📊 Loaded DeepSeek results: {len(results_df)} companies processed")
    
    return results_df


def filter_high_confidence_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Filter DeepSeek results for high confidence only"""
    
    if results_df is None:
        return None
    
    # Filter for successful high-confidence results
    high_conf_mask = (
        (results_df['success'] == True) & 
        (results_df['confidence'] == 'High') &
        (results_df['primary_label'].notna())
    )
    
    high_conf_results = results_df[high_conf_mask].copy()
    
    print(f"🎯 High confidence results: {len(high_conf_results)} companies")
    print(f"   Success rate: {len(high_conf_results) / len(results_df):.1%}")
    
    # Show confidence breakdown
    conf_counts = results_df['confidence'].value_counts()
    print(f"\n📈 Confidence breakdown:")
    for conf, count in conf_counts.items():
        print(f"   {conf}: {count} ({count/len(results_df):.1%})")
    
    return high_conf_results


def load_existing_valid_labels() -> pd.DataFrame:
    """Load companies that already have valid taxonomy labels"""
    
    # Load the training data
    training_df = pd.read_csv('data/processed/training_data_auto_fixed.csv')
    
    # Load taxonomy to identify valid labels
    taxonomy_df = pd.read_csv('data/input/insurance_taxonomy - insurance_taxonomy.csv')
    taxonomy_labels = set(taxonomy_df['label'].str.strip())
    
    # Find companies with valid labels
    valid_mask = training_df['primary_label'].isin(taxonomy_labels)
    valid_companies = training_df[valid_mask].copy()
    
    print(f"✅ Existing valid companies: {len(valid_companies)}")
    
    return valid_companies


def create_high_quality_dataset(valid_companies: pd.DataFrame, 
                               high_conf_results: pd.DataFrame,
                               output_file: str = "data/processed/high_quality_training_dataset.csv") -> pd.DataFrame:
    """Combine valid companies with high-confidence reclassified companies"""
    
    print(f"\n🔧 Creating high-quality dataset...")
    
    # Prepare the reclassified companies
    if high_conf_results is not None and len(high_conf_results) > 0:
        # Load original training data to get full company info
        training_df = pd.read_csv('data/processed/training_data_auto_fixed.csv')
        
        # Create reclassified dataset
        reclassified_companies = []
        for _, result in high_conf_results.iterrows():
            original_idx = result['original_index']
            original_company = training_df.loc[original_idx].copy()
            
            # Update with new high-confidence label
            original_company['primary_label'] = result['primary_label']
            original_company['reclassified'] = True
            original_company['original_problematic_label'] = result['original_label']
            original_company['deepseek_confidence'] = result['confidence']
            original_company['all_deepseek_labels'] = str(result.get('new_labels', []))
            
            reclassified_companies.append(original_company)
        
        reclassified_df = pd.DataFrame(reclassified_companies)
        print(f"   Prepared {len(reclassified_df)} reclassified companies")
    else:
        reclassified_df = pd.DataFrame()
        print(f"   No high-confidence reclassified companies to add")
    
    # Prepare existing valid companies
    valid_companies = valid_companies.copy()
    valid_companies['reclassified'] = False
    valid_companies['original_problematic_label'] = ''
    valid_companies['deepseek_confidence'] = 'N/A'
    valid_companies['all_deepseek_labels'] = ''
    
    # Combine datasets
    if len(reclassified_df) > 0:
        combined_df = pd.concat([valid_companies, reclassified_df], ignore_index=True)
    else:
        combined_df = valid_companies
    
    # Sort by sector, then category for organization
    combined_df = combined_df.sort_values(['sector', 'category', 'primary_label'])
    
    # Save the high-quality dataset
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    
    print(f"\n✅ High-quality dataset created:")
    print(f"   Total companies: {len(combined_df):,}")
    print(f"   Originally valid: {len(valid_companies):,}")
    print(f"   High-conf reclassified: {len(reclassified_df):,}")
    print(f"   Saved to: {output_path}")
    
    return combined_df


def analyze_dataset_quality(dataset_df: pd.DataFrame):
    """Analyze the quality and composition of the final dataset"""
    
    print(f"\n📊 DATASET QUALITY ANALYSIS")
    print(f"=" * 50)
    
    # Basic statistics
    total_companies = len(dataset_df)
    reclassified_count = dataset_df['reclassified'].sum()
    original_valid_count = total_companies - reclassified_count
    
    print(f"📈 Composition:")
    print(f"   Total companies: {total_companies:,}")
    print(f"   Originally valid: {original_valid_count:,} ({original_valid_count/total_companies:.1%})")
    print(f"   Reclassified (high-conf): {reclassified_count:,} ({reclassified_count/total_companies:.1%})")
    
    # Sector distribution
    print(f"\n🏭 Sector distribution:")
    sector_counts = dataset_df['sector'].value_counts().head(10)
    for sector, count in sector_counts.items():
        print(f"   {sector}: {count:,} companies")
    
    # Label distribution
    print(f"\n🏷️  Top labels:")
    label_counts = dataset_df['primary_label'].value_counts().head(10)
    for label, count in label_counts.items():
        print(f"   {label}: {count} companies")
    
    # Quality metrics
    print(f"\n✅ Quality indicators:")
    
    # Check for duplicates
    duplicates = dataset_df.duplicated(subset=['description']).sum()
    print(f"   Duplicate descriptions: {duplicates} ({duplicates/total_companies:.1%})")
    
    # Check for missing descriptions
    missing_desc = dataset_df['description'].isna().sum()
    print(f"   Missing descriptions: {missing_desc} ({missing_desc/total_companies:.1%})")
    
    # Taxonomy validation
    taxonomy_df = pd.read_csv('data/input/insurance_taxonomy - insurance_taxonomy.csv')
    taxonomy_labels = set(taxonomy_df['label'].str.strip())
    invalid_labels = dataset_df[~dataset_df['primary_label'].isin(taxonomy_labels)]
    print(f"   Invalid taxonomy labels: {len(invalid_labels)} ({len(invalid_labels)/total_companies:.1%})")
    
    if len(invalid_labels) > 0:
        print(f"      Invalid labels: {invalid_labels['primary_label'].unique()}")


def create_training_splits(dataset_df: pd.DataFrame, 
                          train_ratio: float = 0.8,
                          output_dir: str = "data/processed/training_splits/"):
    """Create train/validation splits for model training"""
    
    print(f"\n🎯 Creating training splits...")
    
    # Stratified split by primary label to ensure balanced representation
    from sklearn.model_selection import train_test_split
    
    try:
        train_df, val_df = train_test_split(
            dataset_df, 
            test_size=1-train_ratio, 
            stratify=dataset_df['primary_label'],
            random_state=42
        )
    except ValueError:
        # If stratification fails (some labels have only 1 example), do random split
        print("   Stratification failed, using random split...")
        train_df, val_df = train_test_split(
            dataset_df, 
            test_size=1-train_ratio, 
            random_state=42
        )
    
    # Save splits
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_file = output_path / "train.csv"
    val_file = output_path / "validation.csv"
    
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    
    print(f"   Training set: {len(train_df):,} companies → {train_file}")
    print(f"   Validation set: {len(val_df):,} companies → {val_file}")
    
    # Show label distribution in splits
    print(f"\n📊 Label distribution in splits:")
    train_labels = len(train_df['primary_label'].unique())
    val_labels = len(val_df['primary_label'].unique())
    total_labels = len(dataset_df['primary_label'].unique())
    
    print(f"   Total unique labels: {total_labels}")
    print(f"   Labels in training: {train_labels} ({train_labels/total_labels:.1%})")
    print(f"   Labels in validation: {val_labels} ({val_labels/total_labels:.1%})")


def main():
    """Main function to create high-quality dataset"""
    print("🌟 HIGH-QUALITY DATASET CREATOR")
    print("=" * 60)
    print(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Step 1: Load DeepSeek results
    print("📥 Step 1: Loading DeepSeek reclassification results...")
    deepseek_results = load_deepseek_results()
    
    if deepseek_results is None:
        print("❌ Cannot proceed without DeepSeek results. Run overnight processing first.")
        return
    
    # Step 2: Filter for high confidence
    print(f"\n🎯 Step 2: Filtering for high confidence results...")
    high_conf_results = filter_high_confidence_results(deepseek_results)
    
    # Step 3: Load existing valid companies
    print(f"\n✅ Step 3: Loading existing valid companies...")
    valid_companies = load_existing_valid_labels()
    
    # Step 4: Create combined high-quality dataset
    print(f"\n🔧 Step 4: Creating high-quality dataset...")
    final_dataset = create_high_quality_dataset(valid_companies, high_conf_results)
    
    # Step 5: Analyze dataset quality
    analyze_dataset_quality(final_dataset)
    
    # Step 6: Create training splits
    try:
        create_training_splits(final_dataset)
    except ImportError:
        print(f"\n⚠️ Scikit-learn not available - skipping train/val splits")
        print(f"   Install with: pip install scikit-learn")
    
    print(f"\n🎉 HIGH-QUALITY DATASET CREATION COMPLETE!")
    print(f"📁 Files created:")
    print(f"   • data/processed/high_quality_training_dataset.csv")
    print(f"   • data/processed/training_splits/train.csv")
    print(f"   • data/processed/training_splits/validation.csv")
    print(f"\n🚀 Ready for model training or further analysis!")


if __name__ == "__main__":
    main() 