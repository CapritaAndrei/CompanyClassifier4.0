#!/usr/bin/env python3
"""
Quick Demo of the Insurance Taxonomy Classifier
Shows how candidate finding works without manual interaction
"""

from insurance_taxonomy_classifier import InsuranceTaxonomyClassifier

def demo_candidate_finding():
    """Demo the candidate finding functionality"""
    print("🎯 Insurance Taxonomy Classifier - DEMO MODE")
    print("=" * 50)
    
    # Initialize classifier
    classifier = InsuranceTaxonomyClassifier()
    
    # Load data
    classifier.load_taxonomy()
    classifier.load_companies(sample_size=1000)  # Smaller for demo
    
    # Demo: Find candidates for different label types
    test_labels = [
        "Residential Plumbing Services",
        "Commercial Construction Services", 
        "HVAC Installation and Service",
        "Landscaping Services",
        "Software Development Services"
    ]
    
    print("\n🔍 DEMO: Finding candidates for different labels")
    print("=" * 50)
    
    for label in test_labels:
        print(f"\n📋 Label: {label}")
        print("-" * 40)
        
        candidates = classifier.find_labeling_candidates(label, n_candidates=3, min_similarity=0.25)
        
        if candidates:
            for i, candidate in enumerate(candidates):
                company = candidate['company']
                similarity = candidate['similarity']
                
                print(f"\n   {i+1}. Similarity: {similarity:.3f}")
                print(f"      Description: {company['description'][:120]}...")
                if company.get('business_tags'):
                    print(f"      Tags: {company['business_tags'][:80]}...")
                print(f"      Category: {company.get('category', 'N/A')}")
        else:
            print("   No good candidates found")
    
    print(f"\n✅ Demo complete! The system found relevant companies for each label.")
    print(f"📝 In real use, you would label these examples as matches/non-matches")
    print(f"🚀 After collecting ~200-300 labels, you can train a classifier!")

if __name__ == "__main__":
    demo_candidate_finding() 