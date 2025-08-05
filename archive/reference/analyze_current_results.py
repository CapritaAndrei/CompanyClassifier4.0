"""
Comprehensive analysis of current classification results
Refresh understanding of the dataset and identify key problems
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from collections import Counter, defaultdict
import ast

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def analyze_classification_results():
    """Analyze the current full classification results"""
    
    print("🔍 ANALYZING CURRENT CLASSIFICATION RESULTS")
    print("="*80)
    print("Goal: Refresh understanding of dataset state and identify problems")
    print("="*80)
    
    # Load the results
    results_path = Path("src/data/output/full_classification_results.csv")
    
    if not results_path.exists():
        print(f"❌ Results file not found: {results_path}")
        return
    
    print("📂 Loading classification results...")
    try:
        df = pd.read_csv(results_path)
        print(f"✅ Loaded {len(df):,} companies")
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return
    
    # Basic overview
    print(f"\n📊 BASIC OVERVIEW:")
    print(f"="*40)
    print(f"Total companies: {len(df):,}")
    
    # Coverage analysis
    companies_with_labels = len(df[df['num_labels_assigned'] > 0])
    companies_without_labels = len(df[df['num_labels_assigned'] == 0])
    coverage_percent = (companies_with_labels / len(df)) * 100
    
    print(f"Companies WITH labels: {companies_with_labels:,} ({coverage_percent:.1f}%)")
    print(f"Companies WITHOUT labels: {companies_without_labels:,} ({100-coverage_percent:.1f}%)")
    
    # Label distribution
    total_labels_assigned = df['num_labels_assigned'].sum()
    avg_labels_per_company = total_labels_assigned / len(df)
    avg_labels_per_labeled_company = df[df['num_labels_assigned'] > 0]['num_labels_assigned'].mean()
    
    print(f"Total labels assigned: {total_labels_assigned:,}")
    print(f"Average labels per company: {avg_labels_per_company:.2f}")
    print(f"Average labels per LABELED company: {avg_labels_per_labeled_company:.2f}")
    
    # Business tags analysis
    print(f"\n🏷️ BUSINESS TAGS ANALYSIS:")
    print(f"="*40)
    
    # Check for missing business tags
    missing_tags = df[df['business_tags'].isna() | (df['business_tags'] == '') | (df['business_tags'] == '[]')]
    print(f"Companies WITHOUT business tags: {len(missing_tags):,} ({len(missing_tags)/len(df)*100:.1f}%)")
    
    if len(missing_tags) > 0:
        print(f"   📋 Sample companies without business tags:")
        for i, (_, row) in enumerate(missing_tags.head(3).iterrows()):
            print(f"      {i+1}. Company {row['company_id']}: {row['sector']} - {row['category']}")
            print(f"         Labels assigned: {row['num_labels_assigned']}")
    
    # Sector analysis
    print(f"\n🏢 SECTOR ANALYSIS:")
    print(f"="*30)
    
    sector_counts = df['sector'].value_counts()
    print("Companies per sector:")
    for sector, count in sector_counts.items():
        if pd.notna(sector):
            labeled_in_sector = len(df[(df['sector'] == sector) & (df['num_labels_assigned'] > 0)])
            coverage_in_sector = (labeled_in_sector / count) * 100
            print(f"   {sector}: {count:,} companies ({coverage_in_sector:.1f}% labeled)")
    
    # Label frequency analysis
    print(f"\n📈 LABEL FREQUENCY ANALYSIS:")
    print(f"="*40)
    
    # Extract all labels and their frequencies
    all_labels = []
    label_scores = []
    
    for _, row in df.iterrows():
        if pd.notna(row['insurance_labels']) and row['insurance_labels'] != '':
            labels = row['insurance_labels'].split('; ')
            all_labels.extend(labels)
            
        # Extract scores if available
        if pd.notna(row['all_labels_with_scores']) and row['all_labels_with_scores'] != '':
            try:
                score_parts = row['all_labels_with_scores'].split('; ')
                for part in score_parts:
                    if '(' in part and ')' in part:
                        score_str = part.split('(')[-1].replace(')', '')
                        try:
                            score = float(score_str)
                            label_scores.append(score)
                        except:
                            pass
            except:
                pass
    
    label_counter = Counter(all_labels)
    unique_labels_used = len(label_counter)
    
    print(f"Unique labels used: {unique_labels_used}")
    print(f"Most frequent labels:")
    for label, count in label_counter.most_common(10):
        print(f"   {label}: {count:,} companies")
    
    print(f"\nRarest labels:")
    for label, count in label_counter.most_common()[-10:]:
        print(f"   {label}: {count} companies")
    
    # Confidence score analysis
    if label_scores:
        print(f"\n📊 CONFIDENCE SCORE ANALYSIS:")
        print(f"="*40)
        print(f"Total label assignments analyzed: {len(label_scores):,}")
        print(f"Score range: {min(label_scores):.3f} - {max(label_scores):.3f}")
        print(f"Average confidence: {np.mean(label_scores):.3f}")
        print(f"Median confidence: {np.median(label_scores):.3f}")
        
        # Score distribution
        high_conf = sum(1 for s in label_scores if s >= 0.6)
        med_conf = sum(1 for s in label_scores if 0.5 <= s < 0.6)
        low_conf = sum(1 for s in label_scores if s < 0.5)
        
        print(f"High confidence (≥0.6): {high_conf:,} ({high_conf/len(label_scores)*100:.1f}%)")
        print(f"Medium confidence (0.5-0.6): {med_conf:,} ({med_conf/len(label_scores)*100:.1f}%)")
        print(f"Low confidence (<0.5): {low_conf:,} ({low_conf/len(label_scores)*100:.1f}%)")
    
    # Problem identification
    print(f"\n⚠️ KEY PROBLEMS IDENTIFIED:")
    print(f"="*40)
    
    problems = []
    
    # Coverage problem
    if coverage_percent < 70:
        problems.append(f"LOW COVERAGE: Only {coverage_percent:.1f}% of companies have labels")
    
    # Missing business tags
    if len(missing_tags) > 100:
        problems.append(f"MISSING BUSINESS TAGS: {len(missing_tags):,} companies have no business tags")
    
    # Rare labels problem
    rare_labels = sum(1 for count in label_counter.values() if count == 1)
    if rare_labels > 20:
        problems.append(f"RARE LABELS: {rare_labels} labels used only once")
    
    # Very frequent labels (might be too generic)
    very_frequent = sum(1 for count in label_counter.values() if count > len(df) * 0.1)
    if very_frequent > 5:
        problems.append(f"OVER-GENERIC LABELS: {very_frequent} labels used on >10% of companies")
    
    for i, problem in enumerate(problems, 1):
        print(f"   {i}. {problem}")
    
    if not problems:
        print("   ✅ No major problems detected!")
    
    # Data cleaning recommendations
    print(f"\n💡 DATA CLEANING RECOMMENDATIONS:")
    print(f"="*50)
    
    print(f"1. 🎯 Coverage Improvement:")
    print(f"   - {companies_without_labels:,} companies need labels")
    print(f"   - Focus on companies with business tags first")
    print(f"   - For {len(missing_tags):,} companies without tags: use description-based approach")
    
    print(f"\n2. 🧹 Label Quality Filtering:")
    labels_used_once = sum(1 for count in label_counter.values() if count == 1)
    labels_used_rarely = sum(1 for count in label_counter.values() if count <= 3)
    print(f"   - Remove {labels_used_once} labels used only once")
    print(f"   - Consider removing {labels_used_rarely} labels used ≤3 times")
    
    print(f"\n3. 🎚️ Threshold Optimization:")
    if label_scores:
        low_confidence_pct = low_conf / len(label_scores) * 100
        if low_confidence_pct > 30:
            print(f"   - {low_confidence_pct:.1f}% of assignments have low confidence (<0.5)")
            print(f"   - Consider raising threshold to 0.52-0.55")
    
    print(f"\n4. 📋 Sector-Specific Analysis:")
    print(f"   - Review label distribution per sector")
    print(f"   - Identify cross-sector bleeding issues")
    print(f"   - Use reverse mapping to improve sector alignment")
    
    # Save analysis results
    print(f"\n💾 SAVING DETAILED ANALYSIS...")
    
    analysis_results = {
        'overview': {
            'total_companies': len(df),
            'companies_with_labels': companies_with_labels,
            'coverage_percent': coverage_percent,
            'total_labels_assigned': total_labels_assigned,
            'unique_labels_used': unique_labels_used,
        },
        'sector_breakdown': dict(sector_counts),
        'label_frequencies': dict(label_counter.most_common(50)),
        'problems_identified': problems,
        'companies_without_tags': len(missing_tags)
    }
    
    # Save to file for reference
    import json
    with open('current_analysis_summary.json', 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    print(f"✅ Analysis saved to: current_analysis_summary.json")
    
    return analysis_results

def analyze_unlabeled_companies():
    """Deep dive into companies that received no labels"""
    
    print(f"\n" + "="*80)
    print(f"🔍 DEEP DIVE: UNLABELED COMPANIES")
    print(f"="*80)
    
    results_path = Path("src/data/output/full_classification_results.csv")
    df = pd.read_csv(results_path)
    
    unlabeled = df[df['num_labels_assigned'] == 0]
    
    if len(unlabeled) == 0:
        print("✅ All companies have labels!")
        return
    
    print(f"📊 Analyzing {len(unlabeled):,} unlabeled companies...")
    
    # Sector breakdown of unlabeled
    print(f"\n🏢 Unlabeled companies by sector:")
    unlabeled_by_sector = unlabeled['sector'].value_counts()
    for sector, count in unlabeled_by_sector.items():
        if pd.notna(sector):
            total_in_sector = len(df[df['sector'] == sector])
            pct_unlabeled = (count / total_in_sector) * 100
            print(f"   {sector}: {count:,}/{total_in_sector:,} ({pct_unlabeled:.1f}% unlabeled)")
    
    # Business tags analysis for unlabeled
    unlabeled_no_tags = unlabeled[unlabeled['business_tags'].isna() | 
                                 (unlabeled['business_tags'] == '') | 
                                 (unlabeled['business_tags'] == '[]')]
    
    print(f"\n🏷️ Unlabeled companies without business tags: {len(unlabeled_no_tags):,}")
    print(f"   That's {len(unlabeled_no_tags)/len(unlabeled)*100:.1f}% of unlabeled companies")
    
    # Show samples
    print(f"\n📋 Sample unlabeled companies:")
    for i, (_, row) in enumerate(unlabeled.head(5).iterrows()):
        has_tags = pd.notna(row['business_tags']) and row['business_tags'] not in ['', '[]']
        tags_status = "✅ Has tags" if has_tags else "❌ No tags"
        
        print(f"\n   {i+1}. Company {row['company_id']}: {tags_status}")
        print(f"      Sector: {row['sector']}")
        print(f"      Category: {row['category']}")
        if has_tags:
            try:
                tags = ast.literal_eval(row['business_tags'])[:3]
                print(f"      Tags: {tags}...")
            except:
                print(f"      Tags: {str(row['business_tags'])[:100]}...")
        
        # Show truncated description
        desc = str(row['description'])[:150] + "..." if len(str(row['description'])) > 150 else str(row['description'])
        print(f"      Description: {desc}")

def show_next_steps():
    """Show recommended next steps"""
    
    print(f"\n" + "="*80)
    print(f"🚀 RECOMMENDED NEXT STEPS")
    print(f"="*80)
    
    print(f"1. 🎯 Immediate Actions:")
    print(f"   a) Run heatmap curation on current dataset")
    print(f"   b) Remove labels used ≤3 times (likely noise)")
    print(f"   c) Address unlabeled companies")
    
    print(f"\n2. 🔄 Automated Cleaning Strategy:")
    print(f"   a) Frequency-based filtering (remove rare labels)")
    print(f"   b) Confidence-based filtering (remove low-confidence assignments)")  
    print(f"   c) Sector-alignment filtering (remove cross-sector bleeding)")
    
    print(f"\n3. 📋 Missing Label Strategy:")
    print(f"   a) Companies with business tags: Lower threshold, force assignment")
    print(f"   b) Companies without tags: Description-based similarity")
    print(f"   c) Use niche/category most common labels as fallback")
    
    print(f"\n4. 🎛️ Parameter Optimization:")
    print(f"   a) Test lower thresholds (0.45-0.48) for coverage")
    print(f"   b) Implement sector-aware classification")
    print(f"   c) Use reverse mapping for better alignment")

if __name__ == "__main__":
    try:
        results = analyze_classification_results()
        analyze_unlabeled_companies()
        show_next_steps()
        
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print(f"📊 Check 'current_analysis_summary.json' for detailed results")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()