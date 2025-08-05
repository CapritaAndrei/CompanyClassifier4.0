#!/usr/bin/env python3
"""
Hierarchical NAICS Mapping Script - Enhanced Version with 2017+2022 Support

This script finds exact text matches and inclusion matches at different NAICS hierarchy levels
using both 2017 and 2022 NAICS data for maximum coverage.

Key features:
- Exact matches are auto-approved
- Inclusion matches require user approval  
- "Except" rule filters out bad matches
- Shows which NAICS version(s) matched
- Handles both 2017 and 2022 data for BEACON compatibility
"""

import pandas as pd
import json
import re
from typing import Dict, List, Tuple

class HierarchicalNAICSMapper:
    def __init__(self):
        # 2017 NAICS data
        self.code_descriptions_2017 = {}
        self.sector_codes_2017 = {}
        self.subsector_codes_2017 = {}
        self.industry_group_codes_2017 = {}
        self.industry_codes_2017 = {}
        self.us_industry_codes_2017 = {}
        
        # 2022 NAICS data
        self.code_descriptions_2022 = {}
        self.sector_codes_2022 = {}
        self.subsector_codes_2022 = {}
        self.industry_group_codes_2022 = {}
        self.industry_codes_2022 = {}
        self.us_industry_codes_2022 = {}
        
        self.approved_mappings = []
        self.rejected_mappings = []
        
    def load_naics_data(self):
        """Load both 2017 and 2022 NAICS data"""
        print("Loading comprehensive NAICS data (2017 + 2022)...")
        
        try:
            # Load 2017 data
            print("Loading 2017 NAICS data...")
            df_2017 = pd.read_excel('data/input/2-6 digit_2017_Codes.xlsx')
            print(f"Loaded {len(df_2017)} entries from 2017 file")
            self._build_hierarchy_from_comprehensive(df_2017, '2017')
            
            # Load 2022 data
            print("Loading 2022 NAICS data...")
            df_2022 = pd.read_excel('data/input/2-6 digit_2022_Codes.xlsx')
            print(f"Loaded {len(df_2022)} entries from 2022 file")
            self._build_hierarchy_from_comprehensive(df_2022, '2022')
            
            print(f"\n📊 Combined NAICS Data Summary:")
            print(f"  2017: {len(self.code_descriptions_2017)} codes")
            print(f"  2022: {len(self.code_descriptions_2022)} codes")
            print(f"  Total unique: {len(set(self.code_descriptions_2017.keys()) | set(self.code_descriptions_2022.keys()))} codes")
            
        except Exception as e:
            print(f"Error loading NAICS data: {e}")
            return False
            
        return True
    
    def _build_hierarchy_from_comprehensive(self, df, version):
        """Build hierarchical structure from comprehensive NAICS data"""
        print(f"Building {version} NAICS hierarchy...")
        
        # Select the right dictionaries based on version
        if version == '2017':
            code_descriptions = self.code_descriptions_2017
            sector_codes = self.sector_codes_2017
            subsector_codes = self.subsector_codes_2017
            industry_group_codes = self.industry_group_codes_2017
            industry_codes = self.industry_codes_2017
            us_industry_codes = self.us_industry_codes_2017
        else:  # 2022
            code_descriptions = self.code_descriptions_2022
            sector_codes = self.sector_codes_2022
            subsector_codes = self.subsector_codes_2022
            industry_group_codes = self.industry_group_codes_2022
            industry_codes = self.industry_codes_2022
            us_industry_codes = self.us_industry_codes_2022
        
        # Process all entries
        for _, row in df.iterrows():
            if pd.isna(row.iloc[1]) or pd.isna(row.iloc[2]):  # Skip empty rows
                continue
                
            code = str(row.iloc[1]).strip()  # Column 1: NAICS Code
            description = str(row.iloc[2]).strip()  # Column 2: NAICS Title
            
            # Skip non-numeric codes (like headers)
            if not code.replace('-', '').isdigit():
                continue
                
            # Store in main dictionary
            code_descriptions[code] = description
            
            # Classify by hierarchy level based on code length
            code_length = len(code)
            
            if code_length == 2 or '-' in code:  # 2-digit sectors (like "48-49")
                sector_codes[code] = description
            elif code_length == 3:  # 3-digit subsectors
                subsector_codes[code] = description
            elif code_length == 4:  # 4-digit industry groups
                industry_group_codes[code] = description
            elif code_length == 5:  # 5-digit industries
                industry_codes[code] = description
            elif code_length == 6:  # 6-digit US industries
                us_industry_codes[code] = description
        
        print(f"Built {version} hierarchy:")
        print(f"  - Sectors: {len(sector_codes)}")
        print(f"  - Subsectors: {len(subsector_codes)}")
        print(f"  - Industry Groups: {len(industry_group_codes)}")
        print(f"  - Industries: {len(industry_codes)}")
        print(f"  - US Industries: {len(us_industry_codes)}")
    
    def load_all_insurance_labels(self):
        """Load all 220 insurance taxonomy labels"""
        labels = []
        
        # Load from the main insurance taxonomy file
        df = pd.read_csv('data/input/insurance_taxonomy - insurance_taxonomy.csv')
        labels = df['label'].tolist()
        
        print(f"Loaded {len(labels)} total insurance taxonomy labels")
        return labels
    
    def _has_except_exclusion(self, search_term: str, description: str) -> bool:
        """Check if description contains 'except' followed by our search term (bad match)"""
        search_term_lower = search_term.lower().strip()
        description_lower = description.lower().strip()
        
        # Look for "except" patterns that would exclude our search term
        except_patterns = [
            f"except {search_term_lower}",
            f"except.*{search_term_lower}",
            f"{search_term_lower}.*except",
        ]
        
        for pattern in except_patterns:
            if re.search(pattern, description_lower):
                # Check if "except" is within 1-2 words of our search term
                words = description_lower.split()
                search_words = search_term_lower.split()
                
                for i, word in enumerate(words):
                    if word in search_words:
                        # Check surrounding words for "except"
                        start = max(0, i - 3)
                        end = min(len(words), i + 4)
                        context = words[start:end]
                        if "except" in context:
                            return True
        
        return False
    
    def find_exact_and_inclusion_matches(self, label: str) -> Dict:
        """Find exact and inclusion matches in both 2017 and 2022 data"""
        print(f"\n{'='*70}")
        print(f"PROCESSING LABEL: {label}")
        print(f"{'='*70}")
        
        matches_found = {
            'label': label,
            'approved_matches': [],
            'rejected_matches': []
        }
        
        # Normalize label for matching (but keep original for display)
        label_normalized = label.lower().strip()
        
        # Search both versions
        versions = [
            ('2017', {
                'sector': self.sector_codes_2017,
                'subsector': self.subsector_codes_2017, 
                'industry_group': self.industry_group_codes_2017,
                'industry': self.industry_codes_2017,
                'us_industry': self.us_industry_codes_2017
            }),
            ('2022', {
                'sector': self.sector_codes_2022,
                'subsector': self.subsector_codes_2022,
                'industry_group': self.industry_group_codes_2022, 
                'industry': self.industry_codes_2022,
                'us_industry': self.us_industry_codes_2022
            })
        ]
        
        for version, hierarchy_levels in versions:
            print(f"\n🔍 Searching {version} NAICS data...")
            
            for level_name, codes_dict in hierarchy_levels.items():
                level_display = f"{level_name} ({version})"
                print(f"  Checking {level_display}...")
                
                level_matches = self._search_level_exact_and_inclusion(
                    label, label_normalized, codes_dict, level_name, version
                )
                
                for match in level_matches:
                    if match['approved']:
                        matches_found['approved_matches'].append(match)
                    else:
                        matches_found['rejected_matches'].append(match)
        
        return matches_found
    
    def _search_level_exact_and_inclusion(self, original_label: str, label_normalized: str, 
                                         codes_dict: Dict, level: str, version: str) -> List[Dict]:
        """Search for exact and inclusion matches at a specific level"""
        matches = []
        
        for code, description in codes_dict.items():
            desc_normalized = description.lower().strip()
            
            # Apply "except" rule filter first
            if self._has_except_exclusion(original_label, description):
                print(f"    ❌ FILTERED OUT (except rule): {code} - {description}")
                continue
            
            # Check for exact match (auto-approve)
            if label_normalized == desc_normalized:
                print(f"    ✅ EXACT MATCH - AUTO APPROVED ({version})")
                print(f"       Label: '{original_label}'")
                print(f"       NAICS: {code} - {description}")
                
                subordinate_codes = self._find_subordinate_codes(code, level, version)
                print(f"       → Expands to {len(subordinate_codes)} codes")
                
                matches.append({
                    'match_type': 'exact',
                    'parent_code': code,
                    'parent_description': description,
                    'parent_level': level,
                    'naics_version': version,
                    'subordinate_codes': subordinate_codes,
                    'total_codes': len(subordinate_codes),
                    'approved': True,
                    'approval_reason': f'exact_match_auto_approved_{version}'
                })
            
            # Check for inclusion match (requires approval)
            elif label_normalized in desc_normalized or desc_normalized in label_normalized:
                print(f"    🔍 INCLUSION MATCH - NEEDS APPROVAL ({version})")
                print(f"       Label: '{original_label}'")
                print(f"       NAICS: {code} - {description}")
                
                subordinate_codes = self._find_subordinate_codes(code, level, version)
                print(f"       → Would expand to {len(subordinate_codes)} codes")
                
                # Ask for approval
                while True:
                    response = input(f"       Approve this {version} mapping? (y/n/s for skip): ").lower().strip()
                    if response in ['y', 'yes']:
                        print(f"       ✅ APPROVED by user ({version})")
                        approved = True
                        approval_reason = f'inclusion_match_user_approved_{version}'
                        break
                    elif response in ['n', 'no']:
                        print(f"       ❌ REJECTED by user ({version})")
                        approved = False
                        approval_reason = f'inclusion_match_user_rejected_{version}'
                        break
                    elif response in ['s', 'skip']:
                        print(f"       ⏭️  SKIPPED by user ({version})")
                        approved = False
                        approval_reason = f'inclusion_match_user_skipped_{version}'
                        break
                    else:
                        print("       Please enter y/n/s")
                
                matches.append({
                    'match_type': 'inclusion',
                    'parent_code': code,
                    'parent_description': description,
                    'parent_level': level,
                    'naics_version': version,
                    'subordinate_codes': subordinate_codes,
                    'total_codes': len(subordinate_codes),
                    'approved': approved,
                    'approval_reason': approval_reason
                })
        
        return matches
    
    def _find_subordinate_codes(self, parent_code: str, level: str, version: str) -> List[Dict]:
        """Find all codes that fall under a parent code"""
        subordinate_codes = []
        
        # Select the right code dictionary based on version
        if version == '2017':
            code_descriptions = self.code_descriptions_2017
        else:  # 2022
            code_descriptions = self.code_descriptions_2022
        
        if level == 'sector':  # 2-digit sectors (like "48-49")
            # Handle special sector codes with dashes
            if '-' in parent_code:
                # Extract start and end range (e.g., "48-49" -> 48, 49)
                start_code, end_code = parent_code.split('-')
                start_num = int(start_code)
                end_num = int(end_code)
                
                # Find all codes that start with numbers in this range
                for code, desc in code_descriptions.items():
                    if len(code) > 2 and code.isdigit():
                        code_prefix = int(code[:2])
                        if start_num <= code_prefix <= end_num:
                            subordinate_codes.append({
                                'code': code,
                                'description': desc,
                                'level': self._get_code_level(code),
                                'naics_version': version
                            })
            else:
                # Handle regular 2-digit sectors
                for code, desc in code_descriptions.items():
                    if len(code) > 2 and code.startswith(parent_code):
                        subordinate_codes.append({
                            'code': code,
                            'description': desc,
                            'level': self._get_code_level(code),
                            'naics_version': version
                        })
        
        elif level == 'subsector':  # 3-digit
            # Find all codes starting with this 3-digit prefix
            for code, desc in code_descriptions.items():
                if len(code) > 3 and code.startswith(parent_code):
                    subordinate_codes.append({
                        'code': code,
                        'description': desc,
                        'level': self._get_code_level(code),
                        'naics_version': version
                    })
        
        elif level == 'industry_group':  # 4-digit
            # Find all codes starting with this 4-digit prefix
            for code, desc in code_descriptions.items():
                if len(code) > 4 and code.startswith(parent_code):
                    subordinate_codes.append({
                        'code': code,
                        'description': desc,
                        'level': self._get_code_level(code),
                        'naics_version': version
                    })
        
        elif level == 'industry':  # 5-digit
            # Find all codes starting with this 5-digit prefix
            for code, desc in code_descriptions.items():
                if len(code) == 6 and code.startswith(parent_code):
                    subordinate_codes.append({
                        'code': code,
                        'description': desc,
                        'level': self._get_code_level(code),
                        'naics_version': version
                    })
        
        elif level == 'us_industry':  # 6-digit
            # This IS the final level, so just return itself
            subordinate_codes.append({
                'code': parent_code,
                'description': code_descriptions.get(parent_code, ''),
                'level': 'us_industry',
                'naics_version': version
            })
        
        return subordinate_codes
    
    def _get_code_level(self, code: str) -> str:
        """Determine the hierarchy level of a code"""
        if '-' in code or len(code) == 2:
            return 'sector'
        elif len(code) == 3:
            return 'subsector'
        elif len(code) == 4:
            return 'industry_group'
        elif len(code) == 5:
            return 'industry'
        elif len(code) == 6:
            return 'us_industry'
        else:
            return 'unknown'
    
    def process_all_labels(self):
        """Process all 220 insurance labels with interactive approval"""
        labels = self.load_all_insurance_labels()
        all_results = []
        
        print(f"\n{'='*80}")
        print(f"STARTING ENHANCED HIERARCHICAL MAPPING")
        print(f"Processing ALL {len(labels)} insurance taxonomy labels")
        print(f"Using both 2017 and 2022 NAICS data with 'except' rule filtering")
        print(f"{'='*80}")
        
        for i, label in enumerate(labels, 1):
            print(f"\n[{i}/{len(labels)}] Processing: {label}")
            
            # Find matches with interactive approval
            label_results = self.find_exact_and_inclusion_matches(label)
            
            # Only keep results with approved matches
            if label_results['approved_matches']:
                all_results.append(label_results)
                
                # Print summary for this label
                total_codes = sum(match['total_codes'] for match in label_results['approved_matches'])
                print(f"\n📊 LABEL SUMMARY:")
                print(f"   - Approved matches: {len(label_results['approved_matches'])}")
                print(f"   - Total NAICS codes unlocked: {total_codes}")
                
                # Show version breakdown
                version_counts = {}
                for match in label_results['approved_matches']:
                    version = match['naics_version']
                    version_counts[version] = version_counts.get(version, 0) + 1
                
                print(f"   - Version breakdown: {version_counts}")
            else:
                print(f"\n📊 LABEL SUMMARY: No approved matches found")
        
        return all_results
    
    def save_results(self, results: List[Dict], filename: str):
        """Save results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"FINAL RESULTS SAVED TO: {filename}")
        print(f"{'='*80}")
        
        # Print final summary statistics
        total_labels_with_matches = len(results)
        total_approved_matches = sum(len(result['approved_matches']) for result in results)
        total_codes_unlocked = sum(
            sum(match['total_codes'] for match in result['approved_matches'])
            for result in results
        )
        
        print(f"\n🎉 ENHANCED HIERARCHICAL MAPPING COMPLETE!")
        print(f"   - Labels with approved matches: {total_labels_with_matches}")
        print(f"   - Total approved hierarchical matches: {total_approved_matches}")
        print(f"   - Total NAICS codes unlocked: {total_codes_unlocked}")
        
        # Show breakdown by match type and version
        exact_matches = sum(
            len([m for m in result['approved_matches'] if m['match_type'] == 'exact'])
            for result in results
        )
        inclusion_matches = sum(
            len([m for m in result['approved_matches'] if m['match_type'] == 'inclusion'])
            for result in results
        )
        
        version_2017 = sum(
            len([m for m in result['approved_matches'] if m['naics_version'] == '2017'])
            for result in results
        )
        version_2022 = sum(
            len([m for m in result['approved_matches'] if m['naics_version'] == '2022'])
            for result in results
        )
        
        print(f"\n📈 DETAILED BREAKDOWN:")
        print(f"   - Exact matches (auto-approved): {exact_matches}")
        print(f"   - Inclusion matches (user-approved): {inclusion_matches}")
        print(f"   - 2017 NAICS matches: {version_2017}")
        print(f"   - 2022 NAICS matches: {version_2022}")

def main():
    print("=== ENHANCED HIERARCHICAL NAICS MAPPING ===")
    print("🎯 Strategy: 2017+2022 data, exact auto-approval, inclusion manual approval")
    print("🚫 'Except' rule filtering to avoid bad matches")
    print("📋 Clear version indicators for all mappings")
    print()
    
    mapper = HierarchicalNAICSMapper()
    
    # Load both 2017 and 2022 NAICS data
    if not mapper.load_naics_data():
        print("Failed to load NAICS data. Exiting.")
        return
    
    # Process all labels with interactive approval
    results = mapper.process_all_labels()
    
    # Save results
    output_file = 'data/processed/enhanced_hierarchical_mappings.json'
    mapper.save_results(results, output_file)

if __name__ == "__main__":
    main() 