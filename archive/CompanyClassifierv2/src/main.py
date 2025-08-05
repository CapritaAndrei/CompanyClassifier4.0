"""
Main orchestrator for SIC classification system
"""
import numpy as np
from .data.data_loader import load_company_data, load_sic_data, get_sample_companies
from .data.sic_hierarchy import SICHierarchy
from .classifiers.semantic_classifier import SemanticSICClassifier
from .classifiers.weighted_classifier import WeightedSICClassifier
from .utils.formatting import format_sic_code


class SICClassificationSystem:
    """Main system orchestrator for SIC classification"""
    
    def __init__(self):
        """Initialize the classification system"""
        self.sic_hierarchy = None
        self.classifiers = {}
        
    def setup(self):
        """Setup the classification system"""
        print("🎯 Setting up SIC Classification System")
        print("=" * 50)
        
        # Load SIC data and build hierarchy
        print("📥 Loading SIC data...")
        sic_data = load_sic_data()
        print(f"✅ Loaded {len(sic_data)} SIC codes")
        
        # Build hierarchy
        self.sic_hierarchy = SICHierarchy(sic_data)
        self.sic_hierarchy.build_hierarchy()
        
        divisions = len(self.sic_hierarchy.hierarchy['divisions'])
        major_groups = len(self.sic_hierarchy.hierarchy['major_groups'])
        industry_groups = len(self.sic_hierarchy.hierarchy['industry_groups'])
        
        print(f"✅ Hierarchy: {divisions} divisions, {major_groups} major groups, {industry_groups} industry groups")
        
        # Initialize classifiers
        print("🚀 Initializing classifiers...")
        
        self.classifiers['semantic'] = SemanticSICClassifier(self.sic_hierarchy)
        self.classifiers['weighted'] = WeightedSICClassifier(self.sic_hierarchy)
        
        # Create embeddings for all classifiers
        for name, classifier in self.classifiers.items():
            print(f"Creating embeddings for {name} classifier...")
            classifier.create_embeddings()
        
        print("✅ System setup complete!")
        
    def test_single_company(self, company_data):
        """Test classification on a single company"""
        print(f"\n🏢 Testing Company:")
        print(f"   Description: {company_data['description'][:100]}...")
        print(f"   Tags: {company_data['business_tags'][:100]}...")
        print(f"   Category: {company_data['category']}")
        
        results = {}
        
        for name, classifier in self.classifiers.items():
            print(f"\n{name.upper()} CLASSIFIER:")
            result = classifier.classify_company(company_data)
            results[name] = result
            
            # Format and display result
            formatted_code = format_sic_code(
                result['division_code'], 
                result.get('major_group_code'), 
                result.get('industry_group_code'), 
                result.get('sic_code')
            )
            
            print(f"   📂 Classification: {formatted_code}")
            print(f"   📂 Division: {result['division_code']} - {result['division_name']} ({result['division_confidence']:.3f})")
            
            if result['major_group_code']:
                print(f"   📂 Major Group: {str(result['major_group_code']).zfill(2)} ({result['major_group_confidence']:.3f})")
            
            if result['industry_group_code']:
                print(f"   📂 Industry Group: {str(result['industry_group_code']).zfill(3)} ({result['industry_group_confidence']:.3f})")
            
            if result['sic_code']:
                # Get SIC description
                sic_desc = self.sic_hierarchy.sic_data[
                    self.sic_hierarchy.sic_data['SIC'] == result['sic_code']
                ]['Description'].iloc[0][:50]
                print(f"   📂 SIC Code: {str(result['sic_code']).zfill(4)} - {sic_desc}... ({result['sic_confidence']:.3f})")
        
        return results
    
    def test_sample_companies(self, sample_size=10):
        """Test classification on a sample of companies"""
        print(f"\n🧪 Testing {sample_size} companies...")
        
        # Load company data
        companies = load_company_data()
        sample_companies = get_sample_companies(companies, sample_size)
        
        all_results = {name: [] for name in self.classifiers.keys()}
        
        for i, company in enumerate(sample_companies):
            print(f"\n{'='*60}")
            print(f"🏢 COMPANY {i+1}: {company['description'][:50]}...")
            print('='*60)
            
            for name, classifier in self.classifiers.items():
                try:
                    result = classifier.classify_company(company)
                    all_results[name].append(result)
                    
                    # Quick summary
                    formatted_code = format_sic_code(
                        result['division_code'], 
                        result.get('major_group_code'), 
                        result.get('industry_group_code'), 
                        result.get('sic_code')
                    )
                    
                    avg_confidence = np.mean([
                        result['division_confidence'],
                        result.get('major_group_confidence', 0),
                        result.get('industry_group_confidence', 0)
                    ])
                    
                    print(f"{name.capitalize():12} | {formatted_code:15} | Avg: {avg_confidence:.3f}")
                    
                except Exception as e:
                    print(f"❌ Error in {name} classifier: {e}")
                    continue
        
        # Calculate aggregate statistics
        self._print_aggregate_stats(all_results)
        
        return all_results
    
    def _print_aggregate_stats(self, all_results):
        """Print aggregate statistics for all classifiers"""
        print(f"\n{'='*60}")
        print("📊 AGGREGATE STATISTICS")
        print('='*60)
        
        print(f"{'Method':<15} | {'Avg Div':<8} | {'Avg MG':<8} | {'Avg IG':<8} | {'Overall':<8}")
        print("-" * 60)
        
        for name, results in all_results.items():
            if results:
                div_scores = [r['division_confidence'] for r in results]
                mg_scores = [r.get('major_group_confidence', 0) for r in results]
                ig_scores = [r.get('industry_group_confidence', 0) for r in results]
                
                avg_div = np.mean(div_scores)
                avg_mg = np.mean(mg_scores)
                avg_ig = np.mean(ig_scores)
                overall = (avg_div + avg_mg + avg_ig) / 3
                
                print(f"{name.capitalize():<15} | {avg_div:<8.3f} | {avg_mg:<8.3f} | {avg_ig:<8.3f} | {overall:<8.3f}")


def main():
    """Main entry point"""
    system = SICClassificationSystem()
    system.setup()
    
    # Test on sample of companies
    system.test_sample_companies(sample_size=10)
    
    print(f"\n✅ Classification testing complete!")


if __name__ == "__main__":
    main() 