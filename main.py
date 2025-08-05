"""
AUTOMATED INSURANCE COMPANY CLASSIFIER PIPELINE
===============================================
Orchestrates the complete 99% coverage classification system

Flow:
1. Load input data and taxonomy
2. Run classifyByOriginalBusinessTags (60% coverage)
3. Run classifyWithSyntheticBusinessTags (39% more coverage) 
4. Run heatmap verification automatically
5. Generate comprehensive analytics and conclusions
"""

import pandas as pd
import sys
from pathlib import Path
from datetime import datetime
import json
import subprocess

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config.settings import *

class AutomatedInsuranceClassifier:
    """Complete automated pipeline for insurance company classification"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.pipeline_id = self.start_time.strftime('%Y%m%d_%H%M%S')
        self.results = {
            'pipeline_id': self.pipeline_id,
            'start_time': self.start_time.isoformat(),
            'stages': {}
        }
        
        print("🚀 AUTOMATED INSURANCE COMPANY CLASSIFIER")
        print("="*80)
        print(f"Pipeline ID: {self.pipeline_id}")
        print(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
    
    def load_input_data(self):
        """Load input data and taxonomy"""
        print("\n📂 STAGE 1: LOADING INPUT DATA")
        print("-" * 50)
        
        try:
            # Load company data
            print("📊 Loading company dataset...")
            self.company_df = pd.read_csv(COMPANY_DATA_FILE)
            print(f"✅ Loaded {len(self.company_df):,} companies")
            
            # Load taxonomy
            print("📋 Loading insurance taxonomy...")
            self.taxonomy_df = pd.read_csv(TAXONOMY_FILE)
            print(f"✅ Loaded {len(self.taxonomy_df)} taxonomy labels")
            
            self.results['stages']['data_loading'] = {
                'status': 'success',
                'companies_loaded': len(self.company_df),
                'taxonomy_labels_loaded': len(self.taxonomy_df),
                'timestamp': datetime.now().isoformat()
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load input data: {e}")
            self.results['stages']['data_loading'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    def run_original_business_tags_classification(self):
        """Run first tier classification using original business tags"""
        print("\n🏷️ STAGE 2: CLASSIFY BY ORIGINAL BUSINESS TAGS")
        print("-" * 50)
        print("Expected coverage: ~60% of companies")
        
        try:
            # Import and run the first classifier
            from run_full_classification import run_full_classification
            
            print("🔄 Running original business tags classification...")
            tier1_result = run_full_classification()
            
            if tier1_result:
                # Load results - check for the most recent file
                import glob
                tier1_files = glob.glob(str(DATA_OUTPUT_PATH / "quality_classification_results_*.csv"))
                if tier1_files:
                    # Get the most recent file
                    latest_file = max(tier1_files, key=lambda x: Path(x).stat().st_mtime)
                    tier1_df = pd.read_csv(latest_file)
                    labeled_count = len(tier1_df[tier1_df['num_labels_assigned'] > 0])
                    coverage_pct = (labeled_count / len(tier1_df)) * 100
                    
                    print(f"✅ Tier 1 complete: {labeled_count:,}/{len(tier1_df):,} companies labeled ({coverage_pct:.1f}%)")
                    
                    self.results['stages']['tier1_classification'] = {
                        'status': 'success',
                        'companies_labeled': labeled_count,
                        'total_companies': len(tier1_df),
                        'coverage_percentage': coverage_pct,
                        'output_file': latest_file,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    return True
                else:
                    raise Exception("Tier 1 results file not found")
            else:
                raise Exception("Tier 1 classification returned False")
                
        except Exception as e:
            print(f"❌ Tier 1 classification failed: {e}")
            self.results['stages']['tier1_classification'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    def run_synthetic_business_tags_classification(self):
        """Run second tier classification using synthetic business tags"""
        print("\n🧠 STAGE 3: CLASSIFY WITH SYNTHETIC BUSINESS TAGS")
        print("-" * 50)
        print("Expected additional coverage: ~39% more companies")
        
        try:
            # Import and run the second classifier
            import classifyWithSyntheticBusinessTags
            AutomatedDataCleaner = classifyWithSyntheticBusinessTags.AutomatedDataCleaner
            
            print("🔄 Running synthetic business tags classification...")
            cleaner = AutomatedDataCleaner()
            cleaned_df, analytics = cleaner.run_full_cleaning()
            
            if cleaned_df is not None:
                total_labeled = len(cleaned_df[cleaned_df['num_labels_assigned'] > 0])
                total_companies = len(cleaned_df)
                final_coverage = (total_labeled / total_companies) * 100
                
                print(f"✅ Tier 2 complete: {total_labeled:,}/{total_companies:,} companies labeled ({final_coverage:.1f}%)")
                
                self.results['stages']['tier2_classification'] = {
                    'status': 'success',
                    'final_companies_labeled': total_labeled,
                    'total_companies': total_companies,
                    'final_coverage_percentage': final_coverage,
                    'output_file': str(DATA_OUTPUT_PATH / "cleaned_classification_results.csv"),
                    'timestamp': datetime.now().isoformat()
                }
                
                self.final_results_df = cleaned_df
                return True
            else:
                raise Exception("Tier 2 classification returned no results")
                
        except Exception as e:
            print(f"❌ Tier 2 classification failed: {e}")
            self.results['stages']['tier2_classification'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    def run_heatmap_verification(self):
        """Run automated heatmap verification and cleaning"""
        print("\n🔥 STAGE 4: HEATMAP VERIFICATION & CLEANING")
        print("-" * 50)
        print("Automated quality control and noise removal")
        
        try:
            # Import and run heatmap cleaner
            import verificationFunctionHeatMap
            HeatmapCleaner = verificationFunctionHeatMap.HeatmapCleaner
            
            print("🔄 Running heatmap verification...")
            cleaner = HeatmapCleaner()
            
            # Load the cleaned results for heatmap processing
            input_file = DATA_OUTPUT_PATH / "cleaned_classification_results.csv"
            output_file = DATA_OUTPUT_PATH / f"heatmap_cleaned_results_{self.pipeline_id}.csv"
            
            # Run the heatmap cleaning process
            cleaner.load_data(str(input_file))
            cleaner.run_heatmap_cleaning()
            
            # Load the results
            if cleaner.df is not None:
                final_labeled = len(cleaner.df[cleaner.df['num_labels_assigned'] > 0])
                final_total = len(cleaner.df)
                heatmap_coverage = (final_labeled / final_total) * 100
                
                # Calculate what was removed
                original_coverage = self.results['stages']['tier2_classification']['final_coverage_percentage']
                coverage_change = heatmap_coverage - original_coverage
                
                print(f"✅ Heatmap verification complete: {final_labeled:,}/{final_total:,} companies ({heatmap_coverage:.1f}%)")
                print(f"📊 Coverage change: {coverage_change:+.1f} percentage points")
                
                self.results['stages']['heatmap_verification'] = {
                    'status': 'success',
                    'heatmap_companies_labeled': final_labeled,
                    'heatmap_coverage_percentage': heatmap_coverage,
                    'coverage_change': coverage_change,
                    'output_file': str(output_file),
                    'timestamp': datetime.now().isoformat()
                }
                
                self.heatmap_results_df = cleaner.df
                return True
            else:
                raise Exception("Heatmap verification returned no results")
                
        except Exception as e:
            print(f"❌ Heatmap verification failed: {e}")
            self.results['stages']['heatmap_verification'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    def generate_comprehensive_analytics(self):
        """Generate comprehensive analytics and conclusions"""
        print("\n📊 STAGE 5: COMPREHENSIVE ANALYTICS")
        print("-" * 50)
        
        try:
            end_time = datetime.now()
            total_duration = end_time - self.start_time
            
            # Collect all metrics
            tier1_coverage = self.results['stages']['tier1_classification']['coverage_percentage']
            tier2_coverage = self.results['stages']['tier2_classification']['final_coverage_percentage']
            heatmap_coverage = self.results['stages']['heatmap_verification']['heatmap_coverage_percentage']
            
            coverage_improvement_tier2 = tier2_coverage - tier1_coverage
            coverage_change_heatmap = heatmap_coverage - tier2_coverage
            
            analytics = {
                'pipeline_summary': {
                    'pipeline_id': self.pipeline_id,
                    'total_duration_seconds': total_duration.total_seconds(),
                    'total_duration_formatted': str(total_duration),
                    'end_time': end_time.isoformat()
                },
                'coverage_progression': {
                    'initial_coverage_pct': 0.0,
                    'tier1_coverage_pct': tier1_coverage,
                    'tier2_coverage_pct': tier2_coverage,
                    'final_heatmap_coverage_pct': heatmap_coverage
                },
                'coverage_improvements': {
                    'tier1_improvement': tier1_coverage,
                    'tier2_improvement': coverage_improvement_tier2,
                    'heatmap_adjustment': coverage_change_heatmap,
                    'total_improvement': heatmap_coverage
                },
                'data_quality_metrics': {
                    'companies_processed': self.results['stages']['tier1_classification']['total_companies'],
                    'taxonomy_labels_available': self.results['stages']['data_loading']['taxonomy_labels_loaded'],
                    'final_companies_labeled': self.results['stages']['heatmap_verification']['heatmap_companies_labeled']
                }
            }
            
            # Save analytics
            analytics_file = f"pipeline_analytics_{self.pipeline_id}.json"
            with open(analytics_file, 'w') as f:
                json.dump({**self.results, 'analytics': analytics}, f, indent=2)
            
            # Print summary
            print("🎯 PIPELINE EXECUTION SUMMARY")
            print("="*60)
            print(f"📅 Duration: {total_duration}")
            print(f"📊 Companies processed: {analytics['data_quality_metrics']['companies_processed']:,}")
            print(f"📋 Taxonomy labels: {analytics['data_quality_metrics']['taxonomy_labels_available']}")
            print()
            print("📈 COVERAGE PROGRESSION:")
            print(f"   Initial:     {analytics['coverage_progression']['initial_coverage_pct']:6.1f}%")
            print(f"   Tier 1:      {analytics['coverage_progression']['tier1_coverage_pct']:6.1f}% (+{analytics['coverage_improvements']['tier1_improvement']:5.1f}%)")
            print(f"   Tier 2:      {analytics['coverage_progression']['tier2_coverage_pct']:6.1f}% (+{analytics['coverage_improvements']['tier2_improvement']:5.1f}%)")
            print(f"   Final:       {analytics['coverage_progression']['final_heatmap_coverage_pct']:6.1f}% ({analytics['coverage_improvements']['heatmap_adjustment']:+5.1f}%)")
            print()
            print("🎉 CONCLUSIONS:")
            print(f"   • Achieved {analytics['coverage_progression']['final_heatmap_coverage_pct']:.1f}% coverage")
            print(f"   • Tier 1 (original tags): {analytics['coverage_improvements']['tier1_improvement']:.1f}% coverage")
            print(f"   • Tier 2 (synthetic tags): +{analytics['coverage_improvements']['tier2_improvement']:.1f}% additional coverage")
            print(f"   • Heatmap cleaning: {analytics['coverage_improvements']['heatmap_adjustment']:+.1f}% adjustment")
            print(f"   • Total labeled: {analytics['data_quality_metrics']['final_companies_labeled']:,} companies")
            print()
            print(f"💾 Full analytics saved to: {analytics_file}")
            print("="*60)
            
            return True
            
        except Exception as e:
            print(f"❌ Analytics generation failed: {e}")
            return False
    
    def run_complete_pipeline(self):
        """Execute the complete automated pipeline"""
        success = True
        
        # Stage 1: Load data
        if not self.load_input_data():
            return False
        
        # Stage 2: Tier 1 classification  
        if not self.run_original_business_tags_classification():
            return False
            
        # Stage 3: Tier 2 classification
        if not self.run_synthetic_business_tags_classification():
            return False
            
        # Stage 4: Heatmap verification
        if not self.run_heatmap_verification():
            return False
            
        # Stage 5: Analytics
        if not self.generate_comprehensive_analytics():
            return False
            
        print(f"\n🎉 PIPELINE COMPLETE! Duration: {datetime.now() - self.start_time}")
        return True

def main():
    """Main entry point for automated insurance classification pipeline"""
    pipeline = AutomatedInsuranceClassifier()
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("\n✅ All stages completed successfully!")
        return 0
    else:
        print("\n❌ Pipeline failed. Check logs for details.")
        return 1

if __name__ == "__main__":
    exit(main())