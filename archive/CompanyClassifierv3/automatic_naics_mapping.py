#!/usr/bin/env python3
"""
Automatic NAICS Mapping Script for Insurance Taxonomy
====================================================

This script automatically maps the 220 insurance taxonomy labels to NAICS codes by:
1. Exact label matching - "Apparel Manufacturing" → 315xxx NAICS codes  
2. Keyword matching - Extract main keywords and match to NAICS descriptions
3. Hierarchical analysis - Show coverage and patterns by 3-digit NAICS categories

Goal: Expand from 20 → 100+ NAICS codes mapped to unlock 5,000+ training examples
"""

import pandas as pd
import json
import re
from collections import defaultdict, Counter
from difflib import SequenceMatcher
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Set
import warnings
warnings.filterwarnings('ignore')

class AutomaticNAICSMapper:
    def __init__(self):
        """Initialize the NAICS mapper with all reference data"""
        self.insurance_labels = []
        self.naics_codes = pd.DataFrame()
        self.naics_index = pd.DataFrame()  
        self.beacon_data = pd.DataFrame()
        self.current_mapping = {}
        self.new_mapping = {}
        
        # Load all data
        self.load_data()
        
    def load_data(self):
        """Load all reference datasets"""
        print("🔄 Loading reference data...")
        
        # Load insurance taxonomy labels
        insurance_df = pd.read_csv('data/input/insurance_taxonomy - insurance_taxonomy.csv')
        self.insurance_labels = insurance_df['label'].tolist()
        print(f"✅ Loaded {len(self.insurance_labels)} insurance labels")
        
        # Load NAICS official codes and titles
        self.naics_codes = pd.read_excel('6-digit_2017_Codes.xlsx')
        self.naics_codes = self.naics_codes.dropna()
        self.naics_codes.columns = ['naics_code', 'title', 'extra']
        self.naics_codes = self.naics_codes[['naics_code', 'title']]
        self.naics_codes['naics_code'] = self.naics_codes['naics_code'].astype(int).astype(str)
        print(f"✅ Loaded {len(self.naics_codes)} official NAICS codes")
        
        # Load NAICS detailed index (20K+ descriptions)
        self.naics_index = pd.read_excel('2017_NAICS_Index_File.xlsx')
        self.naics_index.columns = ['naics_code', 'description']
        self.naics_index = self.naics_index.dropna()
        self.naics_index['naics_code'] = self.naics_index['naics_code'].astype(str)
        print(f"✅ Loaded {len(self.naics_index)} detailed NAICS descriptions")
        
        # Load current mapping
        with open('data/processed/insurance_to_naics_mapping.json', 'r') as f:
            self.current_mapping = json.load(f)
        print(f"✅ Loaded current mapping with {len(self.current_mapping)} entries")
        
        # Sample BEACON data for analysis
        try:
            beacon_sample = pd.read_csv('example_data_2017.txt', delimiter='|', nrows=10000)
            self.beacon_data = beacon_sample
            print(f"✅ Loaded sample of {len(beacon_sample)} BEACON business descriptions")
        except Exception as e:
            print(f"⚠️  Could not load BEACON data: {e}")
            
    def clean_text(self, text: str) -> str:
        """Clean and normalize text for matching"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase and remove special characters
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove common stop words
        stop_words = {'and', 'or', 'the', 'of', 'for', 'in', 'on', 'at', 'to', 'a', 'an'}
        words = [w for w in text.split() if w not in stop_words]
        
        return ' '.join(words)
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        cleaned = self.clean_text(text)
        words = cleaned.split()
        
        # Filter out very short words and common terms
        meaningful_words = [w for w in words if len(w) > 2]
        
        return meaningful_words
    
    def exact_match_mapping(self) -> Dict[str, List[Tuple[str, str, float]]]:
        """Find exact or near-exact matches between insurance labels and NAICS descriptions"""
        print("\n🔍 Stage 1: Exact Label Matching")
        
        exact_matches = {}
        
        for label in self.insurance_labels:
            matches = []
            cleaned_label = self.clean_text(label)
            
            # Search in both official titles and detailed index
            all_naics = pd.concat([
                self.naics_codes.rename(columns={'title': 'description'}),
                self.naics_index
            ])
            
            for _, row in all_naics.iterrows():
                cleaned_desc = self.clean_text(row['description'])
                
                # Calculate similarity ratio
                similarity = SequenceMatcher(None, cleaned_label, cleaned_desc).ratio()
                
                # High similarity threshold for exact matches
                if similarity > 0.8:
                    matches.append((row['naics_code'], row['description'], similarity))
            
            if matches:
                # Sort by similarity and keep top matches
                matches.sort(key=lambda x: x[2], reverse=True)
                exact_matches[label] = matches[:5]  # Keep top 5 matches
                
        print(f"✅ Found exact matches for {len(exact_matches)} labels")
        return exact_matches
    
    def keyword_match_mapping(self) -> Dict[str, List[Tuple[str, str, float]]]:
        """Find matches based on keyword overlap"""
        print("\n🔍 Stage 2: Keyword-Based Matching")
        
        keyword_matches = {}
        
        for label in self.insurance_labels:
            matches = []
            label_keywords = set(self.extract_keywords(label))
            
            if not label_keywords:
                continue
                
            # Search in NAICS index for keyword overlaps
            for _, row in self.naics_index.iterrows():
                desc_keywords = set(self.extract_keywords(row['description']))
                
                if not desc_keywords:
                    continue
                
                # Calculate keyword overlap score
                intersection = label_keywords.intersection(desc_keywords)
                union = label_keywords.union(desc_keywords)
                
                if intersection and union:
                    jaccard_score = len(intersection) / len(union)
                    overlap_ratio = len(intersection) / len(label_keywords)
                    
                    # Combined score favoring both overlap and ratio
                    combined_score = (jaccard_score * 0.4) + (overlap_ratio * 0.6)
                    
                    if combined_score > 0.3:  # Reasonable threshold
                        matches.append((row['naics_code'], row['description'], combined_score))
            
            if matches:
                # Sort by combined score and keep top matches
                matches.sort(key=lambda x: x[2], reverse=True)
                keyword_matches[label] = matches[:10]  # Keep top 10 matches
                
        print(f"✅ Found keyword matches for {len(keyword_matches)} labels")
        return keyword_matches
    
    def semantic_clustering_analysis(self) -> Dict[str, List[str]]:
        """Analyze semantic clusters to improve mapping quality"""
        print("\n🔍 Stage 3: Semantic Clustering Analysis")
        
        clusters = defaultdict(list)
        
        # Group insurance labels by common keywords
        for label in self.insurance_labels:
            keywords = self.extract_keywords(label)
            
            # Create cluster keys from primary keywords
            for keyword in keywords:
                if len(keyword) > 3:  # Skip very short words
                    clusters[keyword].append(label)
        
        # Find clusters with multiple labels (potential groups)
        meaningful_clusters = {k: v for k, v in clusters.items() if len(v) > 1}
        
        print(f"✅ Found {len(meaningful_clusters)} semantic clusters")
        return meaningful_clusters
    
    def analyze_naics_hierarchy(self, naics_codes: List[str]) -> Dict[str, int]:
        """Analyze NAICS codes by hierarchy levels"""
        hierarchy = defaultdict(int)
        
        for code in naics_codes:
            if len(code) >= 2:
                hierarchy[f"2-digit: {code[:2]}"] += 1
            if len(code) >= 3:
                hierarchy[f"3-digit: {code[:3]}"] += 1
            if len(code) >= 4:
                hierarchy[f"4-digit: {code[:4]}"] += 1
        
        return dict(hierarchy)
    
    def create_improved_mapping(self, exact_matches: Dict, keyword_matches: Dict) -> Dict[str, str]:
        """Create improved mapping by combining all matching strategies"""
        print("\n🛠️  Stage 4: Creating Improved Mapping")
        
        improved_mapping = {}
        mapping_sources = {}
        
        for label in self.insurance_labels:
            best_match = None
            source = "unknown"
            
            # Priority 1: High-confidence exact matches
            if label in exact_matches:
                matches = exact_matches[label]
                if matches and matches[0][2] > 0.85:  # Very high similarity
                    best_match = matches[0][0]
                    source = f"exact_match_{matches[0][2]:.3f}"
            
            # Priority 2: Good keyword matches if no exact match
            if not best_match and label in keyword_matches:
                matches = keyword_matches[label]
                if matches and matches[0][2] > 0.5:  # Good keyword overlap
                    best_match = matches[0][0]
                    source = f"keyword_match_{matches[0][2]:.3f}"
            
            # Priority 3: Medium-confidence exact matches
            if not best_match and label in exact_matches:
                matches = exact_matches[label]
                if matches:
                    best_match = matches[0][0]
                    source = f"exact_match_{matches[0][2]:.3f}"
            
            # Priority 4: Keep existing mapping if still no match
            if not best_match and label in self.current_mapping:
                best_match = self.current_mapping[label]
                source = "existing_mapping"
            
            if best_match:
                improved_mapping[label] = best_match
                mapping_sources[label] = source
        
        self.new_mapping = improved_mapping
        
        print(f"✅ Created improved mapping with {len(improved_mapping)} entries")
        print(f"   - Exact matches: {sum(1 for s in mapping_sources.values() if 'exact' in s)}")
        print(f"   - Keyword matches: {sum(1 for s in mapping_sources.values() if 'keyword' in s)}")
        print(f"   - Existing kept: {sum(1 for s in mapping_sources.values() if 'existing' in s)}")
        
        return improved_mapping, mapping_sources
    
    def analyze_beacon_coverage(self) -> Dict[str, int]:
        """Analyze how many BEACON examples we can unlock"""
        if self.beacon_data.empty:
            return {}
            
        print("\n📊 Stage 5: BEACON Coverage Analysis")
        
        # Count available examples by NAICS code
        beacon_counts = self.beacon_data['NAICS'].value_counts().to_dict()
        
        # Calculate coverage for current vs new mapping
        current_coverage = 0
        new_coverage = 0
        
        current_codes = set(self.current_mapping.values())
        new_codes = set(self.new_mapping.values())
        
        for code in current_codes:
            current_coverage += beacon_counts.get(code, 0)
            
        for code in new_codes:
            new_coverage += beacon_counts.get(code, 0)
        
        coverage_analysis = {
            'current_examples': current_coverage,
            'new_examples': new_coverage,
            'improvement': new_coverage - current_coverage,
            'current_codes': len(current_codes),
            'new_codes': len(new_codes),
            'code_improvement': len(new_codes) - len(current_codes)
        }
        
        print(f"✅ Coverage Analysis Complete:")
        print(f"   - Current examples: {current_coverage:,}")
        print(f"   - New examples: {new_coverage:,}")
        print(f"   - Improvement: +{coverage_analysis['improvement']:,} examples")
        print(f"   - Code expansion: {len(current_codes)} → {len(new_codes)} codes")
        
        return coverage_analysis
    
    def generate_report(self, exact_matches: Dict, keyword_matches: Dict, 
                       improved_mapping: Dict, mapping_sources: Dict, 
                       coverage_analysis: Dict):
        """Generate comprehensive analysis report"""
        print("\n📈 Generating Comprehensive Report")
        
        report = {
            'summary': {
                'total_insurance_labels': len(self.insurance_labels),
                'labels_mapped': len(improved_mapping),
                'mapping_coverage': len(improved_mapping) / len(self.insurance_labels) * 100,
                'unique_naics_codes': len(set(improved_mapping.values())),
                'exact_match_success': len(exact_matches),
                'keyword_match_success': len(keyword_matches)
            },
            'naics_hierarchy': self.analyze_naics_hierarchy(list(improved_mapping.values())),
            'beacon_coverage': coverage_analysis,
            'top_naics_categories': Counter([code[:3] for code in improved_mapping.values()]).most_common(10),
            'mapping_quality': {
                'high_confidence': sum(1 for s in mapping_sources.values() if 'exact' in s and float(s.split('_')[-1]) > 0.9),
                'medium_confidence': sum(1 for s in mapping_sources.values() if 'keyword' in s and float(s.split('_')[-1]) > 0.6),
                'low_confidence': sum(1 for s in mapping_sources.values() if 'existing' in s)
            }
        }
        
        # Save detailed report
        with open('data/processed/naics_mapping_analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2)
            
        # Save improved mapping
        with open('data/processed/improved_insurance_to_naics_mapping.json', 'w') as f:
            json.dump(improved_mapping, f, indent=2)
            
        # Save mapping sources for transparency
        with open('data/processed/mapping_sources.json', 'w') as f:
            json.dump(mapping_sources, f, indent=2)
        
        return report
    
    def print_detailed_results(self, report: Dict):
        """Print detailed analysis results"""
        print("\n" + "="*70)
        print("🎯 AUTOMATIC NAICS MAPPING RESULTS")
        print("="*70)
        
        summary = report['summary']
        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"   • Insurance Labels: {summary['total_insurance_labels']}")
        print(f"   • Successfully Mapped: {summary['labels_mapped']} ({summary['mapping_coverage']:.1f}%)")
        print(f"   • Unique NAICS Codes: {summary['unique_naics_codes']}")
        print(f"   • Exact Match Success: {summary['exact_match_success']} labels")
        print(f"   • Keyword Match Success: {summary['keyword_match_success']} labels")
        
        if report['beacon_coverage']:
            beacon = report['beacon_coverage']
            print(f"\n🚀 BEACON COVERAGE IMPROVEMENT:")
            print(f"   • Training Examples: {beacon['current_examples']:,} → {beacon['new_examples']:,}")
            print(f"   • Example Improvement: +{beacon['improvement']:,} ({beacon['improvement']/beacon['current_examples']*100:.1f}% increase)")
            print(f"   • NAICS Codes: {beacon['current_codes']} → {beacon['new_codes']}")
            print(f"   • Code Improvement: +{beacon['code_improvement']} codes")
        
        print(f"\n🎯 MAPPING QUALITY DISTRIBUTION:")
        quality = report['mapping_quality']
        print(f"   • High Confidence (exact >90%): {quality['high_confidence']} labels")
        print(f"   • Medium Confidence (keyword >60%): {quality['medium_confidence']} labels") 
        print(f"   • Low Confidence (existing): {quality['low_confidence']} labels")
        
        print(f"\n🏭 TOP NAICS CATEGORIES (3-digit):")
        for category, count in report['top_naics_categories']:
            category_name = self.get_naics_category_name(category)
            print(f"   • {category} ({category_name}): {count} labels")
        
        print(f"\n💾 FILES SAVED:")
        print(f"   • data/processed/improved_insurance_to_naics_mapping.json")
        print(f"   • data/processed/naics_mapping_analysis_report.json")
        print(f"   • data/processed/mapping_sources.json")
        
        print("\n" + "="*70)
        
    def get_naics_category_name(self, code_3digit: str) -> str:
        """Get the name of a 3-digit NAICS category"""
        naics_categories = {
            '111': 'Crop Production', '112': 'Animal Production', '113': 'Forestry and Logging',
            '114': 'Fishing, Hunting and Trapping', '115': 'Agriculture and Forestry Support',
            '211': 'Oil and Gas Extraction', '212': 'Mining (except Oil and Gas)',
            '213': 'Support Activities for Mining', '221': 'Utilities', '236': 'Construction of Buildings',
            '237': 'Heavy and Civil Engineering Construction', '238': 'Specialty Trade Contractors',
            '311': 'Food Manufacturing', '312': 'Beverage and Tobacco Product Manufacturing',
            '313': 'Textile Mills', '314': 'Textile Product Mills', '315': 'Apparel Manufacturing',
            '316': 'Leather and Allied Product Manufacturing', '321': 'Wood Product Manufacturing',
            '322': 'Paper Manufacturing', '323': 'Printing and Related Support Activities',
            '324': 'Petroleum and Coal Products Manufacturing', '325': 'Chemical Manufacturing',
            '326': 'Plastics and Rubber Products Manufacturing', '327': 'Nonmetallic Mineral Product Manufacturing',
            '331': 'Primary Metal Manufacturing', '332': 'Fabricated Metal Product Manufacturing',
            '333': 'Machinery Manufacturing', '334': 'Computer and Electronic Product Manufacturing',
            '335': 'Electrical Equipment, Appliance, and Component Manufacturing',
            '336': 'Transportation Equipment Manufacturing', '337': 'Furniture and Related Product Manufacturing',
            '339': 'Miscellaneous Manufacturing', '423': 'Merchant Wholesalers, Durable Goods',
            '424': 'Merchant Wholesalers, Nondurable Goods', '425': 'Wholesale Electronic Markets',
            '441': 'Motor Vehicle and Parts Dealers', '442': 'Furniture and Home Furnishings Stores',
            '443': 'Electronics and Appliance Stores', '444': 'Building Material and Garden Equipment',
            '445': 'Food and Beverage Stores', '446': 'Health and Personal Care Stores',
            '447': 'Gasoline Stations', '448': 'Clothing and Clothing Accessories Stores',
            '451': 'Sporting Goods, Hobby, Musical Instrument, and Book Stores',
            '452': 'General Merchandise Stores', '453': 'Miscellaneous Store Retailers',
            '454': 'Nonstore Retailers', '481': 'Air Transportation', '482': 'Rail Transportation',
            '483': 'Water Transportation', '484': 'Truck Transportation', '485': 'Transit and Ground Passenger Transportation',
            '486': 'Pipeline Transportation', '487': 'Scenic and Sightseeing Transportation',
            '488': 'Support Activities for Transportation', '491': 'Postal Service',
            '492': 'Couriers and Messengers', '493': 'Warehousing and Storage',
            '511': 'Publishing Industries', '512': 'Motion Picture and Sound Recording Industries',
            '515': 'Broadcasting', '517': 'Telecommunications', '518': 'Data Processing, Hosting, and Related Services',
            '519': 'Other Information Services', '521': 'Monetary Authorities-Central Bank',
            '522': 'Credit Intermediation and Related Activities', '523': 'Securities, Commodity Contracts, and Other Financial Investments',
            '524': 'Insurance Carriers and Related Activities', '525': 'Funds, Trusts, and Other Financial Vehicles',
            '531': 'Real Estate', '532': 'Rental and Leasing Services', '533': 'Lessors of Nonfinancial Intangible Assets',
            '541': 'Professional, Scientific, and Technical Services', '551': 'Management of Companies and Enterprises',
            '561': 'Administrative and Support Services', '562': 'Waste Management and Remediation Services',
            '611': 'Educational Services', '621': 'Ambulatory Health Care Services',
            '622': 'Hospitals', '623': 'Nursing and Residential Care Facilities',
            '624': 'Social Assistance', '711': 'Performing Arts, Spectator Sports, and Related Industries',
            '712': 'Museums, Historical Sites, and Similar Institutions', '713': 'Amusement, Gambling, and Recreation Industries',
            '721': 'Accommodation', '722': 'Food Services and Drinking Places',
            '811': 'Repair and Maintenance', '812': 'Personal and Laundry Services',
            '813': 'Religious, Grantmaking, Civic, Professional, and Similar Organizations',
            '814': 'Private Households', '921': 'Executive, Legislative, and Other General Government Support',
            '922': 'Justice, Public Order, and Safety Activities', '923': 'Administration of Human Resource Programs',
            '924': 'Administration of Environmental Quality Programs', '925': 'Administration of Housing Programs, Urban Planning, and Community Development',
            '926': 'Administration of Economic Programs', '927': 'Space Research and Technology',
            '928': 'National Security and International Affairs'
        }
        return naics_categories.get(code_3digit, 'Unknown Category')
    
    def run_full_analysis(self):
        """Run the complete automatic NAICS mapping analysis"""
        print("🚀 Starting Automatic NAICS Mapping Analysis")
        print("Goal: Expand from 20 → 100+ NAICS codes to unlock 5,000+ training examples\n")
        
        # Stage 1: Exact matching
        exact_matches = self.exact_match_mapping()
        
        # Stage 2: Keyword matching  
        keyword_matches = self.keyword_match_mapping()
        
        # Stage 3: Semantic clustering
        clusters = self.semantic_clustering_analysis()
        
        # Stage 4: Create improved mapping
        improved_mapping, mapping_sources = self.create_improved_mapping(exact_matches, keyword_matches)
        
        # Stage 5: Analyze BEACON coverage
        coverage_analysis = self.analyze_beacon_coverage()
        
        # Stage 6: Generate comprehensive report
        report = self.generate_report(exact_matches, keyword_matches, improved_mapping, 
                                    mapping_sources, coverage_analysis)
        
        # Stage 7: Print results
        self.print_detailed_results(report)
        
        return report, improved_mapping

def main():
    """Main execution function"""
    mapper = AutomaticNAICSMapper()
    report, improved_mapping = mapper.run_full_analysis()
    
    print(f"\n🎉 Analysis Complete! Check the generated files for detailed results.")
    return mapper, report, improved_mapping

if __name__ == "__main__":
    mapper, report, improved_mapping = main() 