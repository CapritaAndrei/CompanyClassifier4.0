#!/usr/bin/env python3
"""
Embedding-Based NAICS Mapping Script with Caching

This script finds semantic matches using sentence embeddings at different NAICS hierarchy levels
using both 2017 and 2022 NAICS data for maximum coverage.

Key features:
- Perfect embedding matches (1.0) are auto-approved (very rare)
- All other matches above threshold require manual approval  
- "Except" rule filters out bad matches
- Shows which NAICS version(s) matched
- Uses sentence embeddings for semantic understanding
- CACHES embeddings for faster subsequent runs
"""

import pandas as pd
import json
import re
import numpy as np
import pickle
import os
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingNAICSMapper:
    def __init__(self, similarity_threshold=0.75):
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
        
        # Embedding model and threshold
        print("Loading sentence transformer model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and good quality
        self.similarity_threshold = similarity_threshold
        
        # Precomputed embeddings for efficiency
        self.embeddings_2017 = {}
        self.embeddings_2022 = {}
        self.label_embeddings = {}
        
        # Cache file paths
        self.cache_dir = 'data/processed/embedding_cache'
        self.naics_2017_cache = os.path.join(self.cache_dir, 'naics_2017_embeddings.pkl')
        self.naics_2022_cache = os.path.join(self.cache_dir, 'naics_2022_embeddings.pkl')
        self.labels_cache = os.path.join(self.cache_dir, 'insurance_labels_embeddings.pkl')
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
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
            
            # Load or compute embeddings for efficiency
            self._load_or_compute_embeddings()
            
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
    
    def _load_or_compute_embeddings(self):
        """Load embeddings from cache or compute them if cache doesn't exist"""
        print("\n💾 Checking embedding cache...")
        
        # Check if all cache files exist
        cache_exists = (
            os.path.exists(self.naics_2017_cache) and 
            os.path.exists(self.naics_2022_cache) and 
            os.path.exists(self.labels_cache)
        )
        
        if cache_exists:
            print("✅ Found cached embeddings, loading from cache...")
            self._load_embeddings_from_cache()
        else:
            print("🔄 No cache found, computing embeddings...")
            self._compute_and_cache_embeddings()
    
    def _load_embeddings_from_cache(self):
        """Load precomputed embeddings from cache files"""
        try:
            print("Loading 2017 NAICS embeddings from cache...")
            with open(self.naics_2017_cache, 'rb') as f:
                cache_data = pickle.load(f)
                self.embeddings_2017 = cache_data['embeddings']
                cached_descriptions = cache_data['descriptions']
                
            # Verify cache is still valid
            if cached_descriptions != self.code_descriptions_2017:
                print("⚠️  2017 cache is outdated, will recompute...")
                raise ValueError("Cache outdated")
            
            print("Loading 2022 NAICS embeddings from cache...")
            with open(self.naics_2022_cache, 'rb') as f:
                cache_data = pickle.load(f)
                self.embeddings_2022 = cache_data['embeddings']
                cached_descriptions = cache_data['descriptions']
                
            # Verify cache is still valid
            if cached_descriptions != self.code_descriptions_2022:
                print("⚠️  2022 cache is outdated, will recompute...")
                raise ValueError("Cache outdated")
            
            print("Loading insurance label embeddings from cache...")
            with open(self.labels_cache, 'rb') as f:
                cache_data = pickle.load(f)
                self.label_embeddings = cache_data['embeddings']
                cached_labels = cache_data['labels']
                
            # Load current labels for verification
            current_labels = self.load_all_insurance_labels()
            if cached_labels != current_labels:
                print("⚠️  Label cache is outdated, will recompute...")
                raise ValueError("Cache outdated")
            
            print(f"✅ Successfully loaded cached embeddings:")
            print(f"   - 2017 NAICS: {len(self.embeddings_2017)} codes")
            print(f"   - 2022 NAICS: {len(self.embeddings_2022)} codes") 
            print(f"   - Insurance labels: {len(self.label_embeddings)} labels")
            
        except Exception as e:
            print(f"❌ Error loading cache: {e}")
            print("🔄 Will compute embeddings from scratch...")
            self._compute_and_cache_embeddings()
    
    def _compute_and_cache_embeddings(self):
        """Compute embeddings and save them to cache"""
        print("Computing embeddings (this may take a few minutes)...")
        
        # Compute NAICS embeddings
        print("Computing 2017 NAICS embeddings...")
        all_descriptions_2017 = list(self.code_descriptions_2017.values())
        embeddings_2017_values = self.model.encode(all_descriptions_2017, batch_size=32, show_progress_bar=True)
        
        print("Computing 2022 NAICS embeddings...")
        all_descriptions_2022 = list(self.code_descriptions_2022.values())
        embeddings_2022_values = self.model.encode(all_descriptions_2022, batch_size=32, show_progress_bar=True)
        
        # Store embeddings with code as key
        for i, (code, desc) in enumerate(self.code_descriptions_2017.items()):
            self.embeddings_2017[code] = embeddings_2017_values[i]
            
        for i, (code, desc) in enumerate(self.code_descriptions_2022.items()):
            self.embeddings_2022[code] = embeddings_2022_values[i]
        
        # Compute insurance label embeddings
        print("Computing insurance label embeddings...")
        insurance_labels = self.load_all_insurance_labels()
        label_embeddings_values = self.model.encode(insurance_labels, batch_size=32, show_progress_bar=True)
        
        for i, label in enumerate(insurance_labels):
            self.label_embeddings[label] = label_embeddings_values[i]
        
        # Save to cache
        print("💾 Saving embeddings to cache...")
        
        # Save 2017 NAICS embeddings
        with open(self.naics_2017_cache, 'wb') as f:
            pickle.dump({
                'embeddings': self.embeddings_2017,
                'descriptions': self.code_descriptions_2017
            }, f)
        
        # Save 2022 NAICS embeddings
        with open(self.naics_2022_cache, 'wb') as f:
            pickle.dump({
                'embeddings': self.embeddings_2022,
                'descriptions': self.code_descriptions_2022
            }, f)
        
        # Save insurance label embeddings
        with open(self.labels_cache, 'wb') as f:
            pickle.dump({
                'embeddings': self.label_embeddings,
                'labels': insurance_labels
            }, f)
        
        print(f"✅ Embeddings computed and cached:")
        print(f"   - 2017 NAICS: {len(self.embeddings_2017)} codes")
        print(f"   - 2022 NAICS: {len(self.embeddings_2022)} codes")
        print(f"   - Insurance labels: {len(self.label_embeddings)} labels")
        print(f"   - Cache saved to: {self.cache_dir}")
    
    def load_all_insurance_labels(self):
        """Load all 220 insurance taxonomy labels"""
        labels = []
        
        # Load from the main insurance taxonomy file
        df = pd.read_csv('data/input/insurance_taxonomy - insurance_taxonomy.csv')
        labels = df['label'].tolist()
        
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
    
    def find_embedding_matches(self, label: str) -> Dict:
        """Find semantic matches using embeddings in both 2017 and 2022 data"""
        print(f"\n{'='*70}")
        print(f"PROCESSING LABEL: {label}")
        print(f"SIMILARITY THRESHOLD: {self.similarity_threshold}")
        print(f"{'='*70}")
        
        matches_found = {
            'label': label,
            'approved_matches': [],
            'rejected_matches': []
        }
        
        # Get cached embedding for the input label
        if label in self.label_embeddings:
            label_embedding = self.label_embeddings[label]
        else:
            # Fallback: compute on the fly if not in cache
            print(f"⚠️  Label '{label}' not in cache, computing embedding...")
            label_embedding = self.model.encode([label])[0]
        
        # Search both versions
        versions = [
            ('2017', {
                'sector': self.sector_codes_2017,
                'subsector': self.subsector_codes_2017, 
                'industry_group': self.industry_group_codes_2017,
                'industry': self.industry_codes_2017,
                'us_industry': self.us_industry_codes_2017
            }, self.embeddings_2017),
            ('2022', {
                'sector': self.sector_codes_2022,
                'subsector': self.subsector_codes_2022,
                'industry_group': self.industry_group_codes_2022, 
                'industry': self.industry_codes_2022,
                'us_industry': self.us_industry_codes_2022
            }, self.embeddings_2022)
        ]
        
        for version, hierarchy_levels, embeddings in versions:
            print(f"\n🔍 Searching {version} NAICS data...")
            
            for level_name, codes_dict in hierarchy_levels.items():
                level_display = f"{level_name} ({version})"
                print(f"  Checking {level_display}...")
                
                level_matches = self._search_level_embedding_similarity(
                    label, label_embedding, codes_dict, level_name, version, embeddings
                )
                
                for match in level_matches:
                    if match['approved']:
                        matches_found['approved_matches'].append(match)
                    else:
                        matches_found['rejected_matches'].append(match)
        
        return matches_found
    
    def _search_level_embedding_similarity(self, original_label: str, label_embedding: np.ndarray, 
                                         codes_dict: Dict, level: str, version: str, embeddings: Dict) -> List[Dict]:
        """Search for embedding similarity matches at a specific level"""
        matches = []
        
        # Collect all similarities for this level
        similarities = []
        for code, description in codes_dict.items():
            if code in embeddings:
                # Apply "except" rule filter first
                if self._has_except_exclusion(original_label, description):
                    continue
                
                # Calculate similarity
                desc_embedding = embeddings[code]
                similarity = cosine_similarity([label_embedding], [desc_embedding])[0][0]
                
                if similarity >= self.similarity_threshold:
                    similarities.append((similarity, code, description))
        
        # Sort by similarity (highest first)
        similarities.sort(reverse=True)
        
        # Process top matches
        for similarity, code, description in similarities:
            # Check for perfect match (auto-approve)
            if similarity >= 0.999:  # Essentially perfect match
                print(f"    ✅ PERFECT MATCH - AUTO APPROVED ({version}) - Similarity: {similarity:.4f}")
                print(f"       Label: '{original_label}'")
                print(f"       NAICS: {code} - {description}")
                
                subordinate_codes = self._find_subordinate_codes(code, level, version)
                print(f"       → Expands to {len(subordinate_codes)} codes")
                
                matches.append({
                    'match_type': 'perfect_embedding',
                    'parent_code': code,
                    'parent_description': description,
                    'parent_level': level,
                    'naics_version': version,
                    'similarity_score': float(similarity),
                    'subordinate_codes': subordinate_codes,
                    'total_codes': len(subordinate_codes),
                    'approved': True,
                    'approval_reason': f'perfect_embedding_auto_approved_{version}'
                })
            
            # Similarity match (requires approval)
            else:
                print(f"    🔍 SIMILARITY MATCH - NEEDS APPROVAL ({version}) - Similarity: {similarity:.4f}")
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
                        approval_reason = f'similarity_match_user_approved_{version}'
                        break
                    elif response in ['n', 'no']:
                        print(f"       ❌ REJECTED by user ({version})")
                        approved = False
                        approval_reason = f'similarity_match_user_rejected_{version}'
                        break
                    elif response in ['s', 'skip']:
                        print(f"       ⏭️  SKIPPED by user ({version})")
                        approved = False
                        approval_reason = f'similarity_match_user_skipped_{version}'
                        break
                    else:
                        print("       Please enter y/n/s")
                
                matches.append({
                    'match_type': 'similarity',
                    'parent_code': code,
                    'parent_description': description,
                    'parent_level': level,
                    'naics_version': version,
                    'similarity_score': float(similarity),
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
        print(f"STARTING EMBEDDING-BASED NAICS MAPPING")
        print(f"Processing ALL {len(labels)} insurance taxonomy labels")
        print(f"Using similarity threshold: {self.similarity_threshold}")
        print(f"Using cached embeddings for fast processing")
        print(f"{'='*80}")
        
        for i, label in enumerate(labels, 1):
            print(f"\n[{i}/{len(labels)}] Processing: {label}")
            
            # Find matches with interactive approval
            label_results = self.find_embedding_matches(label)
            
            # Only keep results with approved matches
            if label_results['approved_matches']:
                all_results.append(label_results)
                
                # Print summary for this label
                total_codes = sum(match['total_codes'] for match in label_results['approved_matches'])
                print(f"\n📊 LABEL SUMMARY:")
                print(f"   - Approved matches: {len(label_results['approved_matches'])}")
                print(f"   - Total NAICS codes unlocked: {total_codes}")
                
                # Show version and similarity breakdown
                version_counts = {}
                avg_similarity = 0
                for match in label_results['approved_matches']:
                    version = match['naics_version']
                    version_counts[version] = version_counts.get(version, 0) + 1
                    avg_similarity += match['similarity_score']
                
                avg_similarity /= len(label_results['approved_matches'])
                print(f"   - Version breakdown: {version_counts}")
                print(f"   - Average similarity: {avg_similarity:.4f}")
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
        
        print(f"\n🎉 EMBEDDING-BASED MAPPING COMPLETE!")
        print(f"   - Labels with approved matches: {total_labels_with_matches}")
        print(f"   - Total approved embedding matches: {total_approved_matches}")
        print(f"   - Total NAICS codes unlocked: {total_codes_unlocked}")
        
        # Show breakdown by match type and version
        perfect_matches = sum(
            len([m for m in result['approved_matches'] if m['match_type'] == 'perfect_embedding'])
            for result in results
        )
        similarity_matches = sum(
            len([m for m in result['approved_matches'] if m['match_type'] == 'similarity'])
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
        
        # Calculate average similarity
        all_similarities = []
        for result in results:
            for match in result['approved_matches']:
                all_similarities.append(match['similarity_score'])
        
        avg_similarity = np.mean(all_similarities) if all_similarities else 0
        
        print(f"\n📈 DETAILED BREAKDOWN:")
        print(f"   - Perfect matches (auto-approved): {perfect_matches}")
        print(f"   - Similarity matches (user-approved): {similarity_matches}")
        print(f"   - 2017 NAICS matches: {version_2017}")
        print(f"   - 2022 NAICS matches: {version_2022}")
        print(f"   - Average similarity score: {avg_similarity:.4f}")

def main():
    print("=== EMBEDDING-BASED NAICS MAPPING WITH CACHING ===")
    print("🧠 Strategy: Semantic similarity using sentence embeddings")
    print("🎯 Perfect matches auto-approved, similarity matches need approval")
    print("🚫 'Except' rule filtering to avoid bad matches")
    print("💾 Smart caching for faster subsequent runs")
    print("📋 Clear similarity scores and version indicators")
    print()
    
    # You can adjust the threshold here
    similarity_threshold = 0.75  # Start with 0.75, can be modified manually
    mapper = EmbeddingNAICSMapper(similarity_threshold=similarity_threshold)
    
    # Load both 2017 and 2022 NAICS data and load/compute embeddings
    if not mapper.load_naics_data():
        print("Failed to load NAICS data. Exiting.")
        return
    
    # Process all labels with interactive approval
    results = mapper.process_all_labels()
    
    # Save results
    output_file = 'data/processed/embedding_naics_mappings.json'
    mapper.save_results(results, output_file)

if __name__ == "__main__":
    main() 