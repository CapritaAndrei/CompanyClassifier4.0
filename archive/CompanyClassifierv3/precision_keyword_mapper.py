"""
Precision Keyword Mapper for Business Tags → Insurance Taxonomy Labels
Focuses on high-confidence, low false-positive mappings
"""

import pandas as pd
import ast
import json
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set

class PrecisionKeywordMapper:
    """
    High-precision keyword mapper for business tags to insurance labels
    """
    
    def __init__(self):
        self.insurance_labels = self._load_insurance_taxonomy()
        self.direct_mappings = self._create_direct_mappings()
        self.semantic_mappings = self._create_semantic_mappings()
        self.exclusion_patterns = self._create_exclusion_patterns()
        
    def _load_insurance_taxonomy(self) -> List[str]:
        """Load the 220 insurance taxonomy labels"""
        try:
            df = pd.read_csv('data/input/insurance_taxonomy - insurance_taxonomy.csv')
            return df['label'].tolist()
        except:
            # Fallback with common insurance labels
            return [
                'Construction Services', 'Pipeline Construction Services', 'Excavation Services',
                'Cable Installation Services', 'Water Treatment Services', 'Well Drilling Services',
                'Chemical Manufacturing', 'Plastic Manufacturing', 'Food Processing Services',
                'Pharmaceutical Manufacturing', 'Printing Services', 'Textile Manufacturing Services',
                'Software Manufacturing', 'Consulting Services', 'Management Consulting',
                'Financial Services', 'Insurance Services', 'Real Estate Services',
                'Veterinary Services', 'Medical Services', 'Healthcare Services',
                'Transportation Services', 'Logistics Services', 'Travel Services',
                'Agricultural Equipment Services', 'Landscaping Services', 'Catering Services',
                'Laboratory Services', 'Engineering Services', 'Welding Services',
                'Training Services', 'Arts Services', 'Publishing Services'
            ]
    
    def _create_direct_mappings(self) -> Dict[str, str]:
        """Create direct 1:1 mappings where business tag exactly matches insurance label"""
        direct_mappings = {}
        
        # Exact matches (case-insensitive)
        for label in self.insurance_labels:
            direct_mappings[label.lower()] = label
            
        # Common variations
        variations = {
            'construction services': 'Construction Services',
            'consulting services': 'Consulting Services', 
            'healthcare services': 'Healthcare Services',
            'medical services': 'Medical Services',
            'veterinary services': 'Veterinary Services',
            'financial services': 'Financial Services',
            'insurance services': 'Insurance Services',
            'transportation services': 'Transportation Services',
            'logistics services': 'Logistics Services',
            'catering services': 'Catering Services',
            'engineering services': 'Engineering Services',
            'software services': 'Software Manufacturing',
            'it services': 'Software Manufacturing',
            'manufacturing': 'Chemical Manufacturing',  # Generic fallback
            'food manufacturing': 'Food Processing Services',
            'chemical manufacturing': 'Chemical Manufacturing',
            'plastic manufacturing': 'Plastic Manufacturing',
            'pharmaceutical manufacturing': 'Pharmaceutical Manufacturing',
        }
        
        direct_mappings.update(variations)
        return direct_mappings
    
    def _create_semantic_mappings(self) -> Dict[str, List[Tuple[str, float]]]:
        """Create semantic mappings with confidence scores"""
        semantic_mappings = defaultdict(list)
        
        # Construction-related mappings
        construction_patterns = {
            # High confidence (0.9+)
            r'pipeline|water.*connection|gas.*connection|utility.*connection': [
                ('Pipeline Construction Services', 0.95)
            ],
            r'electrical.*install|electrical.*services|electrical.*contractor': [
                ('Cable Installation Services', 0.90)
            ],
            r'excavation|earthwork|site.*preparation': [
                ('Excavation Services', 0.90)
            ],
            r'water.*treatment|wastewater|sewage': [
                ('Water Treatment Services', 0.90)
            ],
            r'well.*drilling|bore.*hole|water.*well': [
                ('Well Drilling Services', 0.95)
            ],
            
            # Medium confidence (0.7-0.8)
            r'construction.*materials|building.*materials': [
                ('Construction Services', 0.75)
            ],
            r'fiber.*optic|telecommunications.*install': [
                ('Cable Installation Services', 0.75)
            ],
        }
        
        # Manufacturing-related mappings  
        manufacturing_patterns = {
            r'chemical.*production|chemical.*processing': [
                ('Chemical Manufacturing', 0.90)
            ],
            r'plastic.*production|polymer.*manufacturing': [
                ('Plastic Manufacturing', 0.90)
            ],
            r'food.*production|food.*processing|beverage.*production': [
                ('Food Processing Services', 0.85)
            ],
            r'pharmaceutical|drug.*manufacturing|medical.*device': [
                ('Pharmaceutical Manufacturing', 0.90)
            ],
            r'printing|publishing|print.*services': [
                ('Printing Services', 0.85)
            ],
            r'textile|clothing.*manufacturing|fabric': [
                ('Textile Manufacturing Services', 0.85)
            ],
        }
        
        # Professional services mappings
        professional_patterns = {
            r'software.*development|it.*consulting|programming': [
                ('Software Manufacturing', 0.85)
            ],
            r'management.*consulting|business.*consulting': [
                ('Management Consulting', 0.90)
            ],
            r'engineering.*consulting|technical.*consulting': [
                ('Engineering Services', 0.85)
            ],
            r'financial.*planning|accounting|tax.*services': [
                ('Financial Services', 0.80)
            ],
            r'real.*estate|property.*management': [
                ('Real Estate Services', 0.85)
            ],
        }
        
        # Healthcare mappings
        healthcare_patterns = {
            r'veterinary|animal.*care|pet.*care': [
                ('Veterinary Services', 0.90)
            ],
            r'medical.*practice|healthcare.*provider|clinic': [
                ('Medical Services', 0.85)
            ],
        }
        
        # Agriculture mappings
        agriculture_patterns = {
            r'agricultural.*equipment|farm.*equipment': [
                ('Agricultural Equipment Services', 0.90)
            ],
            r'landscaping|lawn.*care|garden': [
                ('Landscaping Services', 0.85)
            ],
        }
        
        # Transportation mappings
        transportation_patterns = {
            r'logistics|shipping|freight': [
                ('Logistics Services', 0.85)
            ],
            r'travel.*agency|tourism': [
                ('Travel Services', 0.85)
            ],
        }
        
        # Food services mappings
        food_patterns = {
            r'catering|event.*catering|food.*catering': [
                ('Catering Services', 0.90)
            ],
            r'restaurant|food.*service|dining': [
                ('Catering Services', 0.75)
            ],
        }
        
        # Combine all patterns
        all_patterns = {}
        all_patterns.update(construction_patterns)
        all_patterns.update(manufacturing_patterns)
        all_patterns.update(professional_patterns)
        all_patterns.update(healthcare_patterns)
        all_patterns.update(agriculture_patterns)
        all_patterns.update(transportation_patterns)
        all_patterns.update(food_patterns)
        
        # Compile patterns
        for pattern, mappings in all_patterns.items():
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
            semantic_mappings[compiled_pattern] = mappings
            
        return semantic_mappings
    
    def _create_exclusion_patterns(self) -> List[re.Pattern]:
        """Create patterns that should be excluded to avoid false positives"""
        exclusion_patterns = [
            r'wheelchair.*accessible',  # Accessibility features, not services
            r'credit.*card|payment',    # Payment methods, not services
            r'in-store.*shopping|pickup', # Store features, not services
            r'parking.*lot',            # Facilities, not services
            r'air.*conditioning|heating', # Unless specifically HVAC services
        ]
        
        return [re.compile(pattern, re.IGNORECASE) for pattern in exclusion_patterns]
    
    def map_business_tag(self, tag: str) -> Tuple[str, float, str]:
        """
        Map a single business tag to insurance label
        
        Returns:
            (insurance_label, confidence_score, mapping_type)
        """
        tag_clean = tag.strip().lower()
        
        # Check exclusions first
        for exclusion_pattern in self.exclusion_patterns:
            if exclusion_pattern.search(tag_clean):
                return None, 0.0, 'excluded'
        
        # Check direct mappings first (highest confidence)
        if tag_clean in self.direct_mappings:
            return self.direct_mappings[tag_clean], 1.0, 'direct'
        
        # Check semantic mappings
        best_match = None
        best_confidence = 0.0
        
        for pattern, mappings in self.semantic_mappings.items():
            if pattern.search(tag_clean):
                for label, confidence in mappings:
                    if confidence > best_confidence:
                        best_match = label
                        best_confidence = confidence
        
        if best_match and best_confidence >= 0.7:  # Minimum confidence threshold
            return best_match, best_confidence, 'semantic'
        
        return None, 0.0, 'no_match'
    
    def map_business_tags_list(self, tags_list: List[str]) -> List[Tuple[str, float, str]]:
        """Map a list of business tags, return all successful mappings"""
        mappings = []
        for tag in tags_list:
            label, confidence, mapping_type = self.map_business_tag(tag)
            if label:
                mappings.append((label, confidence, mapping_type))
        return mappings
    
    def get_best_insurance_label(self, tags_list: List[str]) -> Tuple[str, float, str]:
        """Get the best insurance label for a list of business tags"""
        mappings = self.map_business_tags_list(tags_list)
        
        if not mappings:
            return None, 0.0, 'no_match'
        
        # Sort by confidence, then by mapping type priority (direct > semantic)
        mappings.sort(key=lambda x: (x[1], 1 if x[2] == 'direct' else 0), reverse=True)
        
        return mappings[0]

def test_keyword_mapper():
    """Test the keyword mapper on sample data"""
    print("🧪 Testing Precision Keyword Mapper")
    print("=" * 50)
    
    mapper = PrecisionKeywordMapper()
    
    # Test cases
    test_cases = [
        ['Construction Services', 'Building Materials'],
        ['Software Development', 'IT Consulting', 'Programming'],
        ['Veterinary Services', 'Pet Care'],
        ['Water Connection Installation', 'Utility Network Connections'],
        ['Chemical Manufacturing', 'Production'],
        ['Electrical Services', 'Installation Services'],
        ['Food Production', 'Manufacturing'],
        ['Wheelchair Accessible Entrance'],  # Should be excluded
        ['Credit Card Payment'],              # Should be excluded
    ]
    
    for i, tags in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i}: {tags}")
        
        # Get individual mappings
        for tag in tags:
            label, confidence, mapping_type = mapper.map_business_tag(tag)
            if label:
                print(f"   '{tag}' → '{label}' (confidence: {confidence:.2f}, type: {mapping_type})")
            else:
                print(f"   '{tag}' → No match (type: {mapping_type})")
        
        # Get best overall label
        best_label, best_conf, best_type = mapper.get_best_insurance_label(tags)
        if best_label:
            print(f"   ✅ BEST: '{best_label}' (confidence: {best_conf:.2f})")
        else:
            print(f"   ❌ NO MATCH")

def analyze_real_data_coverage():
    """Analyze how well the mapper covers real business tags"""
    print("\n📊 Analyzing Real Data Coverage")
    print("=" * 50)
    
    # Load real data
    df = pd.read_csv('data/input/ml_insurance_challenge.csv')
    mapper = PrecisionKeywordMapper()
    
    total_companies = 0
    mapped_companies = 0
    mapping_stats = defaultdict(int)
    confidence_distribution = defaultdict(int)
    
    for idx, row in df.iterrows():
        if pd.isna(row['business_tags']):
            continue
            
        try:
            tags_list = ast.literal_eval(row['business_tags'])
            if isinstance(tags_list, list):
                total_companies += 1
                
                best_label, confidence, mapping_type = mapper.get_best_insurance_label(tags_list)
                
                if best_label:
                    mapped_companies += 1
                    mapping_stats[mapping_type] += 1
                    confidence_bucket = f"{int(confidence*10)/10:.1f}-{int(confidence*10+1)/10:.1f}"
                    confidence_distribution[confidence_bucket] += 1
        except:
            continue
    
    coverage_rate = mapped_companies / total_companies if total_companies > 0 else 0
    
    print(f"📈 COVERAGE ANALYSIS:")
    print(f"   Total companies: {total_companies}")
    print(f"   Successfully mapped: {mapped_companies}")
    print(f"   Coverage rate: {coverage_rate:.2%}")
    
    print(f"\n📊 MAPPING TYPE DISTRIBUTION:")
    for mapping_type, count in mapping_stats.items():
        percentage = count / mapped_companies * 100 if mapped_companies > 0 else 0
        print(f"   {mapping_type}: {count} ({percentage:.1f}%)")
    
    print(f"\n📊 CONFIDENCE DISTRIBUTION:")
    for conf_bucket in sorted(confidence_distribution.keys()):
        count = confidence_distribution[conf_bucket]
        percentage = count / mapped_companies * 100 if mapped_companies > 0 else 0
        print(f"   {conf_bucket}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    test_keyword_mapper()
    analyze_real_data_coverage() 