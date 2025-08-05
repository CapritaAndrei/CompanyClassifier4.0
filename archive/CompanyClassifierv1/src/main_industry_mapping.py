"""
Industry Mapping Approach - Main Script

This script implements the new strategy:
1. Map company niche classifications to NAICS codes
2. Map NAICS codes to insurance classifications via industry standards
3. Create ground truth dataset for supervised learning
4. Build supervised ML classifier
"""

import sys
import os
import time
import pandas as pd

# Path setup
src_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from .config import COMPANY_DATA_FILE, TAXONOMY_FILE, SAMPLE_SIZE
from .data.loader import load_data
from .mapping.naics_mapper import NAICSMapper
from .mapping.ncci_interface import NCCIInterface


def main():
    """Main execution function for industry mapping approach."""
    print("🚀 === INDUSTRY MAPPING STRATEGY === 🚀")
    print("Transforming from similarity matching to supervised learning")
    print("using industry-standard NAICS → Insurance classification mappings")
    
    overall_start_time = time.time()
    
    # Step 1: Load raw data
    print("\n--- Step 1: Loading Data ---")
    companies_df, taxonomy_df = load_data(COMPANY_DATA_FILE, TAXONOMY_FILE)
    
    if companies_df.empty or taxonomy_df.empty:
        print("❌ Critical error: No data available. Exiting.")
        sys.exit(1)
    
    # Apply sampling if specified
    if SAMPLE_SIZE is not None:
        companies_df = companies_df.head(SAMPLE_SIZE).copy()
        print(f"📊 Using sample of {len(companies_df)} companies")
    
    print(f"✅ Data loaded: {len(companies_df)} companies, {len(taxonomy_df)} insurance labels")
    
    # Step 2: Map Company Niches to NAICS Codes
    print("\n--- Step 2: Mapping Company Niches → NAICS Codes ---")
    naics_mapper = NAICSMapper()
    
    # Get mapping statistics first
    mapping_stats = naics_mapper.get_mapping_statistics(companies_df)
    print(f"📋 Mapping Analysis:")
    print(f"   Total unique niches: {mapping_stats['total_unique_niches']}")
    print(f"   Mappable niches: {mapping_stats['mappable_niches']}")
    print(f"   Mappable companies: {mapping_stats['mappable_companies']}")
    print(f"   Mapping coverage: {mapping_stats['mappable_companies']/len(companies_df):.1%}")
    
    # Apply NAICS mapping
    companies_with_naics = naics_mapper.map_companies_dataframe(companies_df)
    
    # Save NAICS mapping results
    naics_output_path = "data/processed/company_naics_mapping.csv"
    os.makedirs("data/processed", exist_ok=True)
    naics_mapper.save_mappings_to_csv(companies_df, naics_output_path)
    
    # Step 3: Map NAICS Codes to Insurance Classifications
    print("\n--- Step 3: Mapping NAICS Codes → Insurance Classifications ---")
    ncci_interface = NCCIInterface()
    
    # Create ground truth dataset
    ground_truth_df = ncci_interface.create_ground_truth_dataset(
        companies_with_naics, taxonomy_df, min_confidence=0.5
    )
    
    # Save ground truth for ML training
    ground_truth_path = "data/processed/ground_truth_dataset.csv"
    ncci_interface.save_ground_truth(ground_truth_df, ground_truth_path)
    
    # Step 4: Analyze Ground Truth Quality
    print("\n--- Step 4: Ground Truth Analysis ---")
    analyze_ground_truth_quality(ground_truth_df, companies_df, taxonomy_df)
    
    # Step 5: Research Next Steps
    print("\n--- Step 5: Next Steps for Full Implementation ---")
    print_next_steps(mapping_stats, ground_truth_df)
    
    total_time = time.time() - overall_start_time
    print(f"\n🎯 TOTAL EXECUTION TIME: {total_time:.2f} seconds")
    print("✅ Industry mapping strategy implementation complete!")


def analyze_ground_truth_quality(ground_truth_df, companies_df, taxonomy_df):
    """Analyze the quality and coverage of the ground truth dataset."""
    
    total_companies = len(companies_df)
    ground_truth_companies = len(ground_truth_df)
    coverage = ground_truth_companies / total_companies
    
    print(f"📊 Ground Truth Quality Analysis:")
    print(f"   Coverage: {ground_truth_companies}/{total_companies} ({coverage:.1%})")
    print(f"   Unique insurance labels: {ground_truth_df['insurance_label'].nunique()}")
    print(f"   Total taxonomy labels: {len(taxonomy_df)}")
    print(f"   Label coverage: {ground_truth_df['insurance_label'].nunique()/len(taxonomy_df):.1%}")
    
    # Show label distribution
    print(f"\n📈 Top Insurance Labels in Ground Truth:")
    label_counts = ground_truth_df['insurance_label'].value_counts().head(10)
    for label, count in label_counts.items():
        print(f"   {count:3d} companies → {label}")
    
    # Show confidence distribution
    confidence_stats = ground_truth_df['combined_score'].describe()
    print(f"\n🎯 Confidence Score Distribution:")
    print(f"   Mean: {confidence_stats['mean']:.3f}")
    print(f"   Median: {confidence_stats['50%']:.3f}")
    print(f"   Min: {confidence_stats['min']:.3f}")
    print(f"   Max: {confidence_stats['max']:.3f}")


def print_next_steps(mapping_stats, ground_truth_df):
    """Print actionable next steps for full implementation."""
    
    print("🔬 RESEARCH PRIORITIES:")
    print("1. 🌐 NCCI Class Look-Up Tool Access")
    print("   → Test free online tool at ncci.com/classlookup")
    print("   → Map sample NAICS codes manually")
    print("   → Evaluate API access options")
    
    print("\n2. 📈 Improve NAICS Mapping Coverage")
    unmappable_count = mapping_stats['unmappable_companies']
    if unmappable_count > 0:
        print(f"   → {unmappable_count} companies still unmapped")
        print("   → Review top unmappable niches:")
        for niche, count in mapping_stats['unmappable_niche_details'][:5]:
            print(f"      • {count:3d} companies: {niche}")
    
    print("\n3. 🤖 Supervised ML Implementation")
    if len(ground_truth_df) > 100:
        print(f"   → {len(ground_truth_df)} companies with ground truth")
        print("   → Build text classifier: (description, business_tags) → insurance_label")
        print("   → Use scikit-learn or transformers")
        print("   → Implement train/validation/test split")
    
    print("\n4. 🔗 Industry System Integration")
    print("   → Research Verisk LightSpeed API pricing")
    print("   → Investigate AM Best classification access")
    print("   → Consider IRMI Cross-Reference Guide")
    
    print("\n5. 📊 Validation & Evaluation")
    print("   → Cross-validate against industry benchmarks")
    print("   → A/B test vs current similarity approach")
    print("   → Measure classification accuracy on held-out data")


def test_naics_research_tool():
    """Quick test to help research NCCI tool manually."""
    print("\n🔍 NAICS RESEARCH HELPER")
    print("Use these sample NAICS codes to test NCCI Class Look-Up tool:")
    
    test_naics = [
        ("237990", "Other Heavy and Civil Engineering Construction"),
        ("311411", "Frozen Fruit, Juice, and Vegetable Manufacturing"),
        ("111998", "All Other Miscellaneous Crop Farming"),
        ("541620", "Environmental Consulting Services"),
        ("811121", "Automotive Body, Paint, and Interior Repair")
    ]
    
    print("\nManual Research Steps:")
    print("1. Go to: https://www.ncci.com/classlookup")
    print("2. Search for each NAICS code:")
    
    for naics_code, description in test_naics:
        print(f"   → {naics_code}: {description}")
    
    print("\n3. Record the insurance class codes returned")
    print("4. Update manual_mappings in ncci_interface.py")


if __name__ == "__main__":
    main()
    
    # Uncomment to get NAICS research helper
    # test_naics_research_tool() 