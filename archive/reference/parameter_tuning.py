"""
Parameter Tuning Script: Find optimal settings for high-quality label assignments
Tests different threshold combinations on first 10 companies
"""

import logging
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config.settings import *
from src.data.loader import DataLoader
from src.models.classifier import BusinessTagsClassifier

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise for testing
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class ParameterTuner:
    """Tunes classification parameters for optimal quality"""
    
    def __init__(self):
        self.classifier = None
        self.company_df = None
        self.company_tags = None
        self.company_metadata = None
        self.taxonomy_labels = None
        
    def load_data(self):
        """Load data once for multiple tests"""
        print("🔄 Loading data...")
        
        # Load first 10 companies for testing
        self.company_df = DataLoader.load_company_data(COMPANY_DATA_FILE, sample_size=10)
        if self.company_df is None:
            print("❌ Failed to load company data")
            return False
            
        self.taxonomy_labels = DataLoader.load_taxonomy_labels(TAXONOMY_FILE)
        if self.taxonomy_labels is None:
            print("❌ Failed to load taxonomy labels")
            return False
            
        self.company_tags = DataLoader.get_company_business_tags(self.company_df)
        self.company_metadata = DataLoader.get_company_metadata(self.company_df)
        
        # Initialize classifier
        self.classifier = BusinessTagsClassifier(model_name=EMBEDDING_MODEL)
        success = self.classifier.load_taxonomy(
            self.taxonomy_labels, 
            cache_path=TAXONOMY_EMBEDDINGS_CACHE
        )
        
        if not success:
            print("❌ Failed to load classifier")
            return False
            
        print(f"✅ Loaded {len(self.company_df)} companies and {len(self.taxonomy_labels)} labels")
        return True
        
    def test_parameters(self, similarity_threshold: float, top_k: int, max_labels: int):
        """Test a specific parameter combination"""
        
        # Run classification with these parameters
        results = self.classifier.classify_multiple_companies(
            self.company_tags,
            companies_metadata=self.company_metadata,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            keyword_boost=0.0  # Pure semantic approach
        )
        
        # Analyze results
        summary = self.classifier.get_classification_summary(results)
        
        # Calculate quality metrics
        total_companies = summary['total_companies']
        companies_with_labels = summary['companies_with_labels']
        total_labels = summary['total_labels_assigned']
        unique_labels = summary['unique_labels_used']
        avg_labels = summary['avg_labels_per_company']
        
        # Quality metrics
        coverage = (companies_with_labels / total_companies) * 100
        selectivity = avg_labels  # Lower is more selective
        
        return {
            'similarity_threshold': similarity_threshold,
            'top_k': top_k,
            'max_labels': max_labels,
            'coverage': coverage,
            'companies_with_labels': companies_with_labels,
            'total_labels_assigned': total_labels,
            'unique_labels_used': unique_labels,
            'avg_labels_per_company': avg_labels,
            'selectivity_score': selectivity,
            'results': results
        }
        
    def display_results(self, test_result: dict):
        """Display test results in a clear format"""
        
        print(f"\n{'='*60}")
        print(f"🧪 TEST RESULTS")
        print(f"{'='*60}")
        print(f"📊 Parameters:")
        print(f"  Similarity Threshold: {test_result['similarity_threshold']}")
        print(f"  Top K: {test_result['top_k']}")
        print(f"  Max Labels: {test_result['max_labels']}")
        
        print(f"\n📈 Performance:")
        print(f"  Coverage: {test_result['coverage']:.1f}% ({test_result['companies_with_labels']}/10 companies)")
        print(f"  Total Labels Assigned: {test_result['total_labels_assigned']}")
        print(f"  Unique Labels Used: {test_result['unique_labels_used']}")
        print(f"  Avg Labels/Company: {test_result['avg_labels_per_company']:.2f}")
        
        # Show individual company results
        print(f"\n👥 Individual Company Results:")
        results = test_result['results']
        
        for i, (company_id, labels) in enumerate(results.items(), 1):
            company_row = self.company_df[self.company_df['company_id'] == company_id].iloc[0]
            
            print(f"\n  Company {i} (ID: {company_id}):")
            print(f"    Sector: {company_row['sector']}")
            print(f"    Category: {company_row['category']}")
            print(f"    Business Tags: {self.company_tags[company_id][:3]}...")  # Show first 3 tags
            
            if labels:
                print(f"    Assigned Labels ({len(labels)}):")
                for label, confidence in labels[:3]:  # Show top 3
                    quality = "🟢 Excellent" if confidence > 0.7 else "🟡 Good" if confidence > 0.5 else "🔴 Weak"
                    print(f"      {quality} {label} ({confidence:.3f})")
                if len(labels) > 3:
                    print(f"      ... and {len(labels) - 3} more")
            else:
                print(f"    ❌ No labels assigned")
                
    def run_parameter_sweep(self):
        """Run multiple parameter combinations to find optimal settings"""
        
        print(f"\n🎯 PARAMETER SWEEP FOR HIGH-QUALITY LABELS")
        print(f"{'='*80}")
        
        # Define parameter ranges to test
        threshold_tests = [0.50, 0.55, 0.60, 0.65, 0.70]
        top_k_tests = [5, 7, 10]
        max_labels_tests = [3, 5, 7]
        
        best_results = []
        
        for threshold in threshold_tests:
            for top_k in top_k_tests:
                for max_labels in max_labels_tests:
                    print(f"\n🔍 Testing: Threshold={threshold}, TopK={top_k}, MaxLabels={max_labels}")
                    
                    result = self.test_parameters(threshold, top_k, max_labels)
                    
                    # Quick summary
                    print(f"   Coverage: {result['coverage']:.1f}%, Avg Labels: {result['avg_labels_per_company']:.2f}")
                    
                    # Store result
                    best_results.append(result)
        
        # Find best configurations
        print(f"\n{'='*80}")
        print(f"📊 BEST CONFIGURATIONS")
        print(f"{'='*80}")
        
        # Sort by different criteria
        high_coverage = sorted(best_results, key=lambda x: x['coverage'], reverse=True)[:3]
        low_selectivity = sorted(best_results, key=lambda x: x['avg_labels_per_company'])[:3]
        
        print(f"\n🎯 Highest Coverage (at least 80% companies get labels):")
        for i, result in enumerate(high_coverage, 1):
            if result['coverage'] >= 80:
                print(f"  {i}. Threshold={result['similarity_threshold']}, TopK={result['top_k']}, MaxLabels={result['max_labels']}")
                print(f"     Coverage: {result['coverage']:.1f}%, Avg Labels: {result['avg_labels_per_company']:.2f}")
        
        print(f"\n⭐ Most Selective (fewer labels per company):")
        for i, result in enumerate(low_selectivity, 1):
            print(f"  {i}. Threshold={result['similarity_threshold']}, TopK={result['top_k']}, MaxLabels={result['max_labels']}")
            print(f"     Coverage: {result['coverage']:.1f}%, Avg Labels: {result['avg_labels_per_company']:.2f}")
        
        return best_results
        
    def interactive_tuning(self):
        """Interactive parameter tuning with user input"""
        
        print(f"\n🎮 INTERACTIVE PARAMETER TUNING")
        print(f"{'='*50}")
        print("Enter parameters to test (or 'done' to finish)")
        
        while True:
            try:
                print(f"\nCurrent suggestions:")
                print(f"  • High selectivity: threshold=0.65, top_k=5, max_labels=3")
                print(f"  • Balanced: threshold=0.55, top_k=7, max_labels=5")
                print(f"  • High coverage: threshold=0.50, top_k=10, max_labels=7")
                
                user_input = input(f"\nEnter 'threshold,top_k,max_labels' (e.g., '0.6,5,3') or 'done': ").strip()
                
                if user_input.lower() == 'done':
                    break
                    
                # Parse input
                parts = user_input.split(',')
                if len(parts) != 3:
                    print("❌ Please enter three values separated by commas")
                    continue
                    
                threshold = float(parts[0])
                top_k = int(parts[1])
                max_labels = int(parts[2])
                
                # Validate ranges
                if not (0.3 <= threshold <= 0.9):
                    print("❌ Threshold should be between 0.3 and 0.9")
                    continue
                if not (1 <= top_k <= 20):
                    print("❌ Top K should be between 1 and 20")
                    continue
                if not (1 <= max_labels <= 15):
                    print("❌ Max labels should be between 1 and 15")
                    continue
                
                # Test parameters
                print(f"\n🧪 Testing your parameters...")
                result = self.test_parameters(threshold, top_k, max_labels)
                self.display_results(result)
                
                # Ask if user wants to see details
                show_details = input(f"\nShow detailed company results? (y/n): ").strip().lower()
                if show_details == 'y':
                    # Already shown in display_results
                    pass
                    
            except ValueError:
                print("❌ Invalid input format. Please use numbers.")
            except KeyboardInterrupt:
                print(f"\n\n👋 Exiting parameter tuning")
                break

def main():
    """Main parameter tuning workflow"""
    
    print("🎯 PARAMETER TUNING FOR HIGH-QUALITY LABELS")
    print("="*80)
    print("Goal: Find settings that give us 'almost perfect' labels only")
    print("Strategy: Increase thresholds to be more selective")
    print("="*80)
    
    # Initialize tuner
    tuner = ParameterTuner()
    
    if not tuner.load_data():
        return
    
    # Ask user what they want to do
    print(f"\nChoose tuning approach:")
    print(f"  1. 🚀 Quick parameter sweep (automated)")
    print(f"  2. 🎮 Interactive tuning (manual)")
    print(f"  3. 🧪 Test specific parameters")
    
    choice = input(f"\nYour choice (1-3): ").strip()
    
    if choice == '1':
        tuner.run_parameter_sweep()
    elif choice == '2':
        tuner.interactive_tuning()
    elif choice == '3':
        # Quick test with suggested high-quality parameters
        print(f"\n🧪 Testing suggested high-quality parameters...")
        result = tuner.test_parameters(
            similarity_threshold=0.65,
            top_k=5, 
            max_labels=3
        )
        tuner.display_results(result)
    else:
        print("❌ Invalid choice")
        
    print(f"\n🎉 Parameter tuning completed!")
    print(f"💡 Use the best parameters in your config/settings.py")

if __name__ == "__main__":
    main()