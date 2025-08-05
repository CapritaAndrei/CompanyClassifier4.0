#!/usr/bin/env python3
"""
Rigorous Quality Audit for NAICS Mappings
==========================================

This script performs a thorough semantic audit of all mappings to identify
and fix quality issues that string similarity alone cannot catch.
"""

import pandas as pd
import json
from typing import Dict, List, Tuple, Set
import re

class SemanticValidator:
    def __init__(self):
        # Load reference data
        self.load_reference_data()
        
        # Define semantic domains and their keywords
        self.semantic_domains = {
            'construction': ['construction', 'building', 'installation', 'concrete', 'foundation', 'roofing', 'driveway', 'excavation', 'plumbing', 'electrical', 'hvac', 'painting', 'carpentry', 'masonry', 'welding'],
            'manufacturing': ['manufacturing', 'production', 'processing', 'fabrication', 'assembly', 'textile', 'chemical', 'plastic', 'metal', 'food', 'pharmaceutical', 'rubber'],
            'professional_services': ['consulting', 'legal', 'accounting', 'design', 'engineering', 'research', 'management', 'marketing', 'human_resources'],
            'technology': ['software', 'website', 'digital', 'data', 'technology', 'computer', 'programming', 'development', 'seo'],
            'healthcare': ['veterinary', 'health', 'medical', 'therapy', 'care', 'treatment', 'clinical'],
            'agriculture': ['agricultural', 'farming', 'livestock', 'crop', 'soil', 'pesticide', 'landscaping', 'gardening', 'nursery'],
            'financial': ['insurance', 'financial', 'banking', 'investment', 'real_estate', 'property'],
            'cleaning': ['cleaning', 'maintenance', 'waste', 'recycling', 'environmental'],
            'transportation': ['transportation', 'shipping', 'logistics', 'delivery', 'freight'],
            'entertainment': ['entertainment', 'recreation', 'sports', 'fitness', 'arts', 'media', 'gaming'],
            'education': ['education', 'training', 'teaching', 'learning', 'curriculum'],
            'retail': ['retail', 'sales', 'store', 'wholesale', 'merchant']
        }
        
    def load_reference_data(self):
        """Load all reference data"""
        # Load NAICS descriptions
        self.naics_index = pd.read_excel('2017_NAICS_Index_File.xlsx')
        self.naics_index.columns = ['naics_code', 'description']
        self.naics_index = self.naics_index.dropna()
        self.naics_index['naics_code'] = self.naics_index['naics_code'].astype(str)
        
        # Load insurance labels
        insurance_df = pd.read_csv('data/input/insurance_taxonomy - insurance_taxonomy.csv')
        self.insurance_labels = insurance_df['label'].tolist()
        
        # Load current mapping to audit
        with open('data/processed/final_refined_naics_mapping.json', 'r') as f:
            self.current_mapping = json.load(f)
    
    def get_semantic_domain(self, text: str) -> Set[str]:
        """Identify semantic domains for a text"""
        text_lower = text.lower()
        domains = set()
        
        for domain, keywords in self.semantic_domains.items():
            for keyword in keywords:
                if keyword in text_lower:
                    domains.add(domain)
        
        return domains
    
    def validate_mapping(self, insurance_label: str, naics_code: str) -> Dict:
        """Perform rigorous semantic validation of a mapping"""
        
        # Get NAICS description
        naics_match = self.naics_index[self.naics_index['naics_code'] == naics_code]
        if naics_match.empty:
            return {
                'valid': False,
                'reason': 'NAICS code not found',
                'naics_description': 'NOT FOUND',
                'semantic_match': False
            }
        
        naics_description = naics_match.iloc[0]['description']
        
        # Get semantic domains for both
        insurance_domains = self.get_semantic_domain(insurance_label)
        naics_domains = self.get_semantic_domain(naics_description)
        
        # Check for semantic overlap
        domain_overlap = insurance_domains.intersection(naics_domains)
        semantic_match = len(domain_overlap) > 0
        
        # Special validation rules
        validation_result = self.apply_validation_rules(insurance_label, naics_description)
        
        return {
            'valid': semantic_match and validation_result['valid'],
            'naics_description': naics_description,
            'insurance_domains': list(insurance_domains),
            'naics_domains': list(naics_domains),
            'domain_overlap': list(domain_overlap),
            'semantic_match': semantic_match,
            'validation_rules': validation_result
        }
    
    def apply_validation_rules(self, insurance_label: str, naics_description: str) -> Dict:
        """Apply specific validation rules"""
        
        insurance_lower = insurance_label.lower()
        naics_lower = naics_description.lower()
        
        # Define obvious mismatches
        mismatches = [
            ('software', 'aerobic'),
            ('software', 'dance'),
            ('software', 'exercise'),
            ('website', 'archeological'),
            ('website', 'archaeological'),
            ('development', 'dance'),
            ('development', 'exercise'),
            ('health', 'building'),
            ('safety', 'renovation'),
            ('consulting', 'construction'),
            ('digital', 'aerobic'),
            ('technology', 'fitness'),
            ('marketing', 'dance'),
            ('data', 'exercise'),
            ('programming', 'recreation'),
            ('coding', 'entertainment')
        ]
        
        # Check for obvious mismatches
        for ins_word, naics_word in mismatches:
            if ins_word in insurance_lower and naics_word in naics_lower:
                return {
                    'valid': False,
                    'reason': f'Semantic mismatch: {ins_word} vs {naics_word}'
                }
        
        # Check for sector mismatches
        tech_keywords = ['software', 'website', 'digital', 'technology', 'data', 'programming', 'seo']
        fitness_keywords = ['aerobic', 'dance', 'exercise', 'fitness', 'recreation', 'sports']
        
        has_tech = any(word in insurance_lower for word in tech_keywords)
        has_fitness = any(word in naics_lower for word in fitness_keywords)
        
        if has_tech and has_fitness:
            return {
                'valid': False,
                'reason': 'Technology service mapped to fitness/recreation'
            }
        
        return {'valid': True, 'reason': 'Passed validation rules'}
    
    def audit_all_mappings(self) -> Dict:
        """Audit all current mappings"""
        
        print("🔍 RIGOROUS SEMANTIC AUDIT OF ALL MAPPINGS")
        print("=" * 60)
        
        valid_mappings = {}
        invalid_mappings = {}
        audit_summary = {
            'total_mappings': len(self.current_mapping),
            'valid_mappings': 0,
            'invalid_mappings': 0,
            'validation_reasons': {}
        }
        
        for insurance_label, naics_code in self.current_mapping.items():
            validation = self.validate_mapping(insurance_label, naics_code)
            
            if validation['valid']:
                valid_mappings[insurance_label] = {
                    'naics_code': naics_code,
                    'validation': validation
                }
                audit_summary['valid_mappings'] += 1
            else:
                invalid_mappings[insurance_label] = {
                    'naics_code': naics_code,
                    'validation': validation
                }
                audit_summary['invalid_mappings'] += 1
                
                # Count validation reasons
                reason = validation.get('reason', 'Unknown')
                audit_summary['validation_reasons'][reason] = audit_summary['validation_reasons'].get(reason, 0) + 1
        
        return valid_mappings, invalid_mappings, audit_summary
    
    def show_worst_mappings(self, invalid_mappings: Dict, n: int = 20):
        """Show the worst mappings for review"""
        
        print(f"\n❌ WORST {n} MAPPINGS IDENTIFIED:")
        print("-" * 80)
        
        count = 0
        for label, data in invalid_mappings.items():
            if count >= n:
                break
                
            naics_code = data['naics_code']
            validation = data['validation']
            naics_desc = validation['naics_description']
            reason = validation.get('reason', 'No specific reason')
            
            print(f"\n{count+1}. {label}")
            print(f"   → {naics_code}: {naics_desc}")
            print(f"   ❌ REASON: {reason}")
            
            if 'insurance_domains' in validation and 'naics_domains' in validation:
                print(f"   📋 Insurance domains: {validation['insurance_domains']}")
                print(f"   🏭 NAICS domains: {validation['naics_domains']}")
                print(f"   🔗 Overlap: {validation['domain_overlap']}")
            
            count += 1
    
    def create_high_quality_mapping(self, valid_mappings: Dict) -> Dict:
        """Create a high-quality mapping from valid entries"""
        
        high_quality_mapping = {}
        
        for label, data in valid_mappings.items():
            high_quality_mapping[label] = data['naics_code']
        
        return high_quality_mapping
    
    def analyze_beacon_coverage(self, high_quality_mapping: Dict):
        """Analyze BEACON coverage for the high-quality mapping"""
        
        try:
            # Load BEACON data
            beacon_2017 = pd.read_csv('example_data_2017.txt', delimiter='|')
            beacon_2022 = pd.read_csv('example_data_2022.txt', delimiter='|')
            all_beacon = pd.concat([beacon_2017, beacon_2022])
            beacon_counts = all_beacon['NAICS'].value_counts().to_dict()
            
            # Calculate coverage
            hq_codes = set(high_quality_mapping.values())
            overlapping_codes = hq_codes.intersection(set(str(code) for code in beacon_counts.keys()))
            
            total_examples = sum(beacon_counts.get(int(code), 0) for code in overlapping_codes if code.isdigit())
            
            print(f"\n📊 HIGH-QUALITY MAPPING BEACON ANALYSIS:")
            print(f"   ✅ High-quality NAICS codes: {len(hq_codes)}")
            print(f"   ✅ Codes with BEACON data: {len(overlapping_codes)}")
            print(f"   ✅ Total training examples: {total_examples:,}")
            print(f"   ✅ Average per code: {total_examples/len(overlapping_codes):.0f}")
            
            return total_examples, len(overlapping_codes)
            
        except Exception as e:
            print(f"   ⚠️ Could not analyze BEACON: {e}")
            return 0, 0

def main():
    """Run the rigorous quality audit"""
    
    validator = SemanticValidator()
    
    # Audit all mappings
    valid_mappings, invalid_mappings, audit_summary = validator.audit_all_mappings()
    
    # Print summary
    print(f"\n📊 AUDIT SUMMARY:")
    print(f"   Total mappings audited: {audit_summary['total_mappings']}")
    print(f"   ✅ Valid mappings: {audit_summary['valid_mappings']}")
    print(f"   ❌ Invalid mappings: {audit_summary['invalid_mappings']}")
    print(f"   📈 Quality rate: {audit_summary['valid_mappings']/audit_summary['total_mappings']*100:.1f}%")
    
    # Show validation reasons
    print(f"\n🔍 INVALIDITY REASONS:")
    for reason, count in audit_summary['validation_reasons'].items():
        print(f"   • {reason}: {count} mappings")
    
    # Show worst mappings
    validator.show_worst_mappings(invalid_mappings)
    
    # Create high-quality mapping
    high_quality_mapping = validator.create_high_quality_mapping(valid_mappings)
    
    # Analyze BEACON coverage
    total_examples, beacon_codes = validator.analyze_beacon_coverage(high_quality_mapping)
    
    # Save high-quality mapping
    with open('data/processed/high_quality_naics_mapping.json', 'w') as f:
        json.dump(high_quality_mapping, f, indent=2)
    
    # Save audit report
    audit_report = {
        'audit_summary': audit_summary,
        'valid_mappings': valid_mappings,
        'invalid_mappings': invalid_mappings,
        'beacon_analysis': {
            'total_examples': total_examples,
            'beacon_codes': beacon_codes,
            'quality_codes': len(high_quality_mapping)
        }
    }
    
    with open('data/processed/rigorous_audit_report.json', 'w') as f:
        json.dump(audit_report, f, indent=2)
    
    print(f"\n💾 RESULTS SAVED:")
    print(f"   • data/processed/high_quality_naics_mapping.json ({len(high_quality_mapping)} mappings)")
    print(f"   • data/processed/rigorous_audit_report.json (detailed audit)")
    
    print(f"\n🎯 FINAL QUALITY RESULTS:")
    print(f"   • High-quality mappings: {len(high_quality_mapping)}")
    print(f"   • Training examples available: {total_examples:,}")
    print(f"   • Quality improvement needed: {audit_summary['invalid_mappings']} bad mappings removed")

if __name__ == "__main__":
    main() 