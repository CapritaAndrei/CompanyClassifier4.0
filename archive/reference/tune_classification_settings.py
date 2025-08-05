"""
Classification Settings Tuner
Test different thresholds and parameters on first 10 companies to optimize for high-quality labels
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClassificationTuner:
    """Test different classification settings to optimize for label quality"""
    
    def __init__(self):
        self.classifier = None
        self.company_df = None
        self.company_tags = None
        self.company_metadata = None
        self.taxonomy_labels = None
        
    def load_data(self):
        """Load all necessary data"""
        logger.info("Loading data...")
        
        # Load first 10 companies for testing
        self.company_df = DataLoader.load_company_data(COMPANY_DATA_FILE, sample_size=10)
        if self.company_df is None:
            return False
        
        # Load taxonomy
        self.taxonomy_labels = DataLoader.load_taxonomy_labels(TAXONOMY_FILE)
        if self.taxonomy_labels is None:
            return False
        
        # Extract tags and metadata
        self.company_tags = DataLoader.get_company_business_tags(self.company_df)
        self.company_metadata = DataLoader.get_company_metadata(self.company_df)
        
        # Initialize classifier
        self.classifier = BusinessTagsClassifier(model_name=EMBEDDING_MODEL)
        success = self.classifier.load_taxonomy(
            self.taxonomy_labels, 
            cache_path=TAXONOMY_EMBEDDINGS_CACHE
        )
        
        if not success:
            return False
        
        print(f"✅ Loaded {len(self.company_df)} companies and {len(self.taxonomy_labels)} taxonomy labels")
        return True
    
    def test_settings_combination(self, similarity_threshold, top_k, max_labels, 
                                keyword_boost=0.0, description="Test"):
        """
        Test a specific combination of settings
        
        Args:
            similarity_threshold: Minimum similarity score
            top_k: Number of top labels to consider
            max_labels: Maximum labels to assign per company
            keyword_boost: Keyword boost (keeping at 0.0 for pure semantic)
            description: Description of this test
            
        Returns:
            Dictionary with results and statistics
        """
        logger.info(f"Testing: {description}")
        logger.info(f"  Similarity threshold: {similarity_threshold}")
        logger.info(f"  Top K: {top_k}")
        logger.info(f"  Max labels: {max_labels}")
        
        # Run classification with these settings
        classification_results = self.classifier.classify_multiple_companies(
            self.company_tags,
            companies_metadata=self.company_metadata,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            keyword_boost=keyword_boost
        )
        
        # Calculate statistics
        summary = self.classifier.get_classification_summary(classification_results)
        
        # Detailed analysis
        companies_with_labels = summary['companies_with_labels']
        total_labels = summary['total_labels_assigned']
        unique_labels = summary['unique_labels_used']
        avg_labels = summary['avg_labels_per_company']
        
        # Calculate quality metrics
        high_confidence_count = 0
        confidence_scores = []
        
        for company_id, labels in classification_results.items():
            # Apply max_labels limit
            limited_labels = labels[:max_labels]
            
            for label, confidence in limited_labels:
                confidence_scores.append(confidence)
                if confidence > 0.6:  # High confidence threshold
                    high_confidence_count += 1
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        min_confidence = min(confidence_scores) if confidence_scores else 0
        max_confidence = max(confidence_scores) if confidence_scores else 0
        
        results = {
            'description': description,
            'settings': {
                'similarity_threshold': similarity_threshold,
                'top_k': top_k,
                'max_labels': max_labels
            },
            'coverage': {
                'companies_with_labels': companies_with_labels,
                'companies_without_labels': 10 - companies_with_labels,
                'coverage_percentage': (companies_with_labels / 10) * 100
            },
            'label_stats': {
                'total_labels_assigned': total_labels,
                'unique_labels_used': unique_labels,
                'avg_labels_per_company': avg_labels
            },
            'quality_metrics': {
                'avg_confidence': avg_confidence,
                'min_confidence': min_confidence,
                'max_confidence': max_confidence,
                'high_confidence_labels': high_confidence_count,
                'high_confidence_percentage': (high_confidence_count / len(confidence_scores)) * 100 if confidence_scores else 0
            },
            'classification_results': classification_results
        }
        
        return results
    
    def display_results(self, results):
        """Display results in a readable format"""
        
        print(f"\n{'='*80}")
        print(f"📊 {results['description']}")
        print(f"{'='*80}")
        
        # Settings
        settings = results['settings']
        print(f"⚙️  Settings:")
        print(f"   Similarity Threshold: {settings['similarity_threshold']}")
        print(f"   Top K: {settings['top_k']}")
        print(f"   Max Labels: {settings['max_labels']}")
        
        # Coverage
        coverage = results['coverage']
        print(f"\n📈 Coverage:")
        print(f"   Companies with labels: {coverage['companies_with_labels']}/10 ({coverage['coverage_percentage']:.1f}%)")
        print(f"   Companies without labels: {coverage['companies_without_labels']}")
        
        # Label statistics
        stats = results['label_stats']
        print(f"\n🏷️  Label Statistics:")
        print(f"   Total labels assigned: {stats['total_labels_assigned']}")
        print(f"   Unique labels used: {stats['unique_labels_used']}")
        print(f"   Average labels per company: {stats['avg_labels_per_company']:.2f}")
        
        # Quality metrics
        quality = results['quality_metrics']
        print(f"\n⭐ Quality Metrics:")
        print(f"   Average confidence: {quality['avg_confidence']:.3f}")
        print(f"   Confidence range: {quality['min_confidence']:.3f} - {quality['max_confidence']:.3f}")
        print(f"   High confidence labels (>0.6): {quality['high_confidence_labels']} ({quality['high_confidence_percentage']:.1f}%)")
        
        # Show sample assignments
        print(f"\n📋 Sample Assignments:")
        for i, (company_id, labels) in enumerate(list(results['classification_results'].items())[:3]):
            company_row = self.company_df[self.company_df['company_id'] == company_id].iloc[0]
            print(f"\n   Company {company_id} ({company_row['sector']}):")
            print(f"   Tags: {self.company_tags[company_id][:3]}...")  # Show first 3 tags
            
            if labels:
                for label, confidence in labels[:3]:  # Show top 3 labels
                    quality_indicator = "🟢" if confidence > 0.6 else "🟡" if confidence > 0.5 else "🔴"
                    print(f"     {quality_indicator} {label}: {confidence:.3f}")
            else:
                print(f"     ❌ No labels assigned")
    
    def run_tuning_session(self):
        """Run multiple test configurations to find optimal settings"""
        
        print("🎯 CLASSIFICATION SETTINGS TUNER")
        print("="*80)
        print("Testing different settings to optimize for high-quality labels")
        print("="*80)
        
        if not self.load_data():
            print("❌ Failed to load data")
            return
        
        # Define test configurations
        test_configs = [
            # Current settings (baseline)
            {
                'similarity_threshold': 0.45,
                'top_k': 11,
                'max_labels': 7,
                'description': "BASELINE - Current Settings"
            },
            
            # More selective (higher threshold)
            {
                'similarity_threshold': 0.55,
                'top_k': 11,
                'max_labels': 5,
                'description': "SELECTIVE - Higher Threshold"
            },
            
            # Very selective (even higher threshold)
            {
                'similarity_threshold': 0.65,
                'top_k': 11,
                'max_labels': 3,
                'description': "VERY SELECTIVE - Much Higher Threshold"
            },
            
            # Ultra selective (only very confident matches)
            {
                'similarity_threshold': 0.70,
                'top_k': 8,
                'max_labels': 2,
                'description': "ULTRA SELECTIVE - Only Best Matches"
            },
            
            # Extreme selective (almost perfect matches only)
            {
                'similarity_threshold': 0.75,
                'top_k': 5,
                'max_labels': 1,
                'description': "EXTREME SELECTIVE - Almost Perfect Only"
            }
        ]
        
        # Store all results for comparison
        all_results = []
        
        # Run each configuration
        for config in test_configs:
            results = self.test_settings_combination(**config)
            self.display_results(results)
            all_results.append(results)
        
        # Comparison summary
        print(f"\n{'='*80}")
        print(f"📊 COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        print(f"{'Configuration':<25} {'Coverage':<12} {'Avg Labels':<12} {'Avg Conf':<12} {'High Conf %':<12}")
        print("-" * 80)
        
        for results in all_results:
            coverage = results['coverage']['coverage_percentage']
            avg_labels = results['label_stats']['avg_labels_per_company']
            avg_conf = results['quality_metrics']['avg_confidence']
            high_conf_pct = results['quality_metrics']['high_confidence_percentage']
            
            print(f"{results['description'][:24]:<25} {coverage:>8.1f}%    {avg_labels:>8.2f}     {avg_conf:>8.3f}     {high_conf_pct:>8.1f}%")
        
        # Recommendation
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   • Balance coverage vs quality based on your needs")
        print(f"   • Higher thresholds = fewer but better labels")
        print(f"   • Consider 'SELECTIVE' or 'VERY SELECTIVE' for quality focus")
        print(f"   • Use heatmap system to validate chosen configuration")
        
        return all_results

def main():
    """Run the classification tuning session"""
    tuner = ClassificationTuner()
    results = tuner.run_tuning_session()
    
    print(f"\n🎉 Tuning session completed!")
    print(f"   Choose your preferred configuration and update src/config/settings.py")
    print(f"   Then run the heatmap system for data validation/cleaning")

if __name__ == "__main__":
    main() 