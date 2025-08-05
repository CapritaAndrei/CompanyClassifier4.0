"""
BEACON Dataset Strategic Mapper
===============================

Analyzes 40k BEACON training dataset to identify high-frequency unmapped NAICS codes
and provides interactive mapping interface with embedding similarity ordering.

Strategy:
1. Load 40k BEACON training examples
2. Count frequency of each NAICS code
3. Identify unmapped codes (not in Master Map)
4. Present mapping candidates with embedding similarity
5. Save progress and allow resume
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import Counter
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class BeaconDatasetMapper:
    """
    Strategic mapper for expanding Master Map using BEACON training dataset
    """
    
    def __init__(self, 
                 beacon_dataset_path: str = "BEACON/example_data_2017.txt",
                 master_map_path: str = "data/processed/master_insurance_to_naics_mapping_simplified.json",
                 taxonomy_path: str = "data/input/insurance_taxonomy - insurance_taxonomy.csv",
                 progress_file: str = "data/processed/beacon_mapping_progress.json"):
        
        self.beacon_dataset_path = beacon_dataset_path
        self.master_map_path = master_map_path
        self.taxonomy_path = taxonomy_path
        self.progress_file = progress_file
        
        # Initialize embedding model
        print("🔄 Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load data
        self.beacon_data = None
        self.master_map = {}
        self.taxonomy_labels = []
        self.naics_frequency = {}
        self.unmapped_naics = {}
        self.progress = self._load_progress()
        
        # Embedding cache
        self.embedding_cache_file = "data/processed/taxonomy_embeddings_cache.pkl"
        self.taxonomy_embeddings = self._load_or_create_taxonomy_embeddings()
    
    def _load_progress(self) -> Dict:
        """Load existing progress or create new"""
        if Path(self.progress_file).exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            "mapped_naics": {},
            "skipped_naics": [],
            "session_stats": {"total_mapped": 0, "total_skipped": 0}
        }
    
    def _save_progress(self):
        """Save current progress"""
        Path(self.progress_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
        print(f"✅ Progress saved to {self.progress_file}")
    
    def _load_or_create_taxonomy_embeddings(self) -> np.ndarray:
        """Load or create embeddings for all 220 insurance labels"""
        if Path(self.embedding_cache_file).exists():
            print("📥 Loading cached taxonomy embeddings...")
            with open(self.embedding_cache_file, 'rb') as f:
                return pickle.load(f)
        
        print("🔄 Creating taxonomy embeddings...")
        # Load taxonomy
        taxonomy_df = pd.read_csv(self.taxonomy_path)
        self.taxonomy_labels = taxonomy_df['label'].tolist()
        
        # Create embeddings
        embeddings = self.embedding_model.encode(self.taxonomy_labels)
        
        # Cache embeddings
        Path(self.embedding_cache_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.embedding_cache_file, 'wb') as f:
            pickle.dump(embeddings, f)
        
        return embeddings
    
    def load_beacon_dataset(self):
        """Load and analyze BEACON training dataset"""
        print("📊 Loading BEACON training dataset...")
        
        # Try different possible file locations/names
        possible_paths = [
            self.beacon_dataset_path,
            "BEACON/example_data_2017.txt",
            "BEACON/example_data_2022.txt",
            "data/beacon_training_data.csv",
            "data/processed/beacon_training_data.csv"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                print(f"📥 Found dataset at: {path}")
                
                # Handle different file formats
                if path.endswith('.txt'):
                    # BEACON format: TEXT|NAICS|SAMPLE_WEIGHT
                    self.beacon_data = pd.read_csv(path, sep='|')
                    # Rename columns to standard format
                    self.beacon_data.columns = ['company_description', 'naics_code', 'sample_weight']
                else:
                    # CSV format
                    self.beacon_data = pd.read_csv(path)
                break
        
        if self.beacon_data is None:
            print("❌ BEACON dataset not found. Please ensure it's available.")
            print("Expected columns: company_description, naics_code")
            return False
        
        print(f"✅ Loaded {len(self.beacon_data)} BEACON training examples")
        print(f"   Columns: {list(self.beacon_data.columns)}")
        
        return True
    
    def load_master_map(self):
        """Load current Master Map"""
        print("📥 Loading Master Map...")
        
        with open(self.master_map_path, 'r') as f:
            self.master_map = json.load(f)
        
        # Extract all mapped NAICS codes
        mapped_codes = set()
        for label, codes in self.master_map.items():
            for code_info in codes:
                mapped_codes.add(code_info['naics_code'])
        
        print(f"✅ Master Map loaded:")
        print(f"   Insurance labels: {len(self.master_map)}")
        print(f"   Mapped NAICS codes: {len(mapped_codes)}")
        
        return mapped_codes
    
    def analyze_naics_frequency(self, mapped_codes: set):
        """Analyze frequency of NAICS codes in BEACON dataset"""
        print("🔍 Analyzing NAICS code frequency...")
        
        # Count frequency of each NAICS code
        naics_column = 'naics_code'  # Adjust if column name is different
        if naics_column not in self.beacon_data.columns:
            print(f"❌ Column '{naics_column}' not found in dataset")
            print(f"Available columns: {list(self.beacon_data.columns)}")
            return
        
        # Count frequencies
        self.naics_frequency = Counter(self.beacon_data[naics_column].astype(str))
        
        # Identify unmapped codes
        self.unmapped_naics = {
            code: count for code, count in self.naics_frequency.items()
            if code not in mapped_codes and code not in self.progress["mapped_naics"]
            and code not in self.progress["skipped_naics"]
        }
        
        print(f"📊 NAICS Analysis Results:")
        print(f"   Total unique NAICS codes: {len(self.naics_frequency)}")
        print(f"   Already mapped: {len(self.naics_frequency) - len(self.unmapped_naics)}")
        print(f"   Unmapped candidates: {len(self.unmapped_naics)}")
        
        # Show top unmapped codes
        top_unmapped = sorted(self.unmapped_naics.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n🔝 Top 10 Unmapped NAICS Codes:")
        for code, count in top_unmapped:
            print(f"   {code}: {count} examples")
    
    def get_naics_description(self, naics_code: str) -> str:
        """Get description for a NAICS code"""
        # Load NAICS lookup data if available
        # For now, return placeholder
        return f"NAICS Code {naics_code}"
    
    def get_sample_companies(self, naics_code: str, limit: int = 15) -> List[str]:
        """Get sample company descriptions for a NAICS code"""
        samples = self.beacon_data[self.beacon_data['naics_code'].astype(str) == naics_code]
        description_column = 'company_description'  # Should be standardized now
        
        if description_column not in samples.columns:
            # Try common alternatives
            for alt_col in ['description', 'business_description', 'company_desc', 'TEXT']:
                if alt_col in samples.columns:
                    description_column = alt_col
                    break
        
        if description_column in samples.columns:
            # Get unique descriptions to avoid duplicates
            unique_descriptions = samples[description_column].unique()
            return unique_descriptions[:limit].tolist()
        return ["Sample descriptions not available"]
    
    def get_similarity_ranked_labels(self, naics_code: str, sample_descriptions: List[str]) -> List[Tuple[str, float]]:
        """Get insurance labels ranked by similarity to NAICS code + samples"""
        
        # Create text for similarity comparison
        naics_desc = self.get_naics_description(naics_code)
        sample_text = " ".join(sample_descriptions[:2])  # Use first 2 samples
        combined_text = f"{naics_desc} {sample_text}"
        
        # Get embedding for combined text
        query_embedding = self.embedding_model.encode([combined_text])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.taxonomy_embeddings)[0]
        
        # Rank labels by similarity
        ranked_labels = [(self.taxonomy_labels[i], similarities[i]) 
                        for i in range(len(self.taxonomy_labels))]
        ranked_labels.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_labels
    
    def get_custom_search_ranked_labels(self, custom_search_term: str) -> List[Tuple[str, float]]:
        """Get insurance labels ranked by similarity to custom search term"""
        
        # Get embedding for custom search term
        query_embedding = self.embedding_model.encode([custom_search_term])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.taxonomy_embeddings)[0]
        
        # Rank labels by similarity
        ranked_labels = [(self.taxonomy_labels[i], similarities[i]) 
                        for i in range(len(self.taxonomy_labels))]
        ranked_labels.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_labels
    
    def _handle_custom_search(self, naics_code: str, custom_term: str, custom_ranked: List[Tuple[str, float]], count: int) -> str:
        """Handle custom search interface and return action taken"""
        
        print(f"\n🎯 Insurance Labels (ranked by similarity to '{custom_term}'):")
        
        # Show top 15 results
        for j, (label, similarity) in enumerate(custom_ranked[:15], 1):
            print(f"   {j:2d}. {label:<40} (similarity: {similarity:.4f})")
        
        print(f"\n" + "-"*60)
        print(f"🔽 Custom Search Options:")
        print(f"   16. Show all 220 labels (custom search)")
        print(f"   18. Back to original rankings")
        print(f"   s.  Skip this NAICS code")
        print(f"   q.  Quit and save progress")
        
        # Custom search interaction loop
        while True:
            custom_choice = input(f"\n👉 Select option (1-16, 18, s, q): ").strip().lower()
            
            if custom_choice == 'q':
                print("💾 Saving progress and quitting...")
                self._save_progress()
                return 'quit'
            
            elif custom_choice == 's':
                print(f"⏭️  Skipping NAICS code {naics_code}")
                self.progress["skipped_naics"].append(naics_code)
                self.progress["session_stats"]["total_skipped"] += 1
                return 'skip'
            
            elif custom_choice == '16':
                print(f"\n📜 All 220 Insurance Labels (custom search for '{custom_term}'):")
                for j, (label, similarity) in enumerate(custom_ranked, 1):
                    print(f"   {j:3d}. {label:<40} (similarity: {similarity:.4f})")
                print(f"\n" + "-"*60)
                continue
            
            elif custom_choice == '18':
                print(f"🔄 Returning to original rankings...")
                return 'back'
            
            elif custom_choice.isdigit():
                choice_num = int(custom_choice)
                if 1 <= choice_num <= len(custom_ranked):
                    selected_label = custom_ranked[choice_num - 1][0]
                    similarity_score = custom_ranked[choice_num - 1][1]
                    
                    print(f"✅ Mapping: {naics_code} → {selected_label}")
                    print(f"   Similarity: {similarity_score:.4f}")
                    print(f"   Training examples: {count:,}")
                    print(f"   Search term: '{custom_term}'")
                    
                    # Save mapping
                    self.progress["mapped_naics"][naics_code] = {
                        "insurance_label": selected_label,
                        "similarity_score": float(similarity_score),
                        "training_examples": count,
                        "search_term": custom_term,
                        "mapped_at": datetime.now().isoformat()
                    }
                    self.progress["session_stats"]["total_mapped"] += 1
                    
                    # Auto-save progress
                    self._save_progress()
                    return 'mapped'
                else:
                    print(f"❌ Invalid choice. Please enter 1-{len(custom_ranked)}, 16, 18, s, or q")
            else:
                print(f"❌ Invalid choice. Please enter 1-16, 18, s, or q")
    
    def interactive_mapping_session(self):
        """Interactive session for mapping unmapped NAICS codes"""
        print("\n" + "="*70)
        print("🎯 BEACON DATASET STRATEGIC MAPPING SESSION")
        print("="*70)
        
        if not self.unmapped_naics:
            print("🎉 No unmapped NAICS codes found! All codes are already mapped.")
            return
        
        print(f"📊 Session Overview:")
        print(f"   Unmapped NAICS codes: {len(self.unmapped_naics)}")
        print(f"   Previously mapped: {self.progress['session_stats']['total_mapped']}")
        print(f"   Previously skipped: {self.progress['session_stats']['total_skipped']}")
        
        print(f"\n💡 Instructions:")
        print(f"   • Review each NAICS code with samples")
        print(f"   • Choose best insurance label (ranked by similarity)")
        print(f"   • Enter number to select, 's' to skip, 'q' to quit & save")
        
        # Sort unmapped codes by frequency (most frequent first)
        sorted_unmapped = sorted(self.unmapped_naics.items(), key=lambda x: x[1], reverse=True)
        
        for i, (naics_code, count) in enumerate(sorted_unmapped, 1):
            print(f"\n" + "="*70)
            print(f"🔍 MAPPING CANDIDATE {i}/{len(sorted_unmapped)}")
            print(f"   NAICS Code: {naics_code}")
            print(f"   Frequency: {count:,} examples in dataset")
            print(f"   Description: {self.get_naics_description(naics_code)}")
            print("-" * 60)
            
            # Get sample companies and additional info
            samples = self.get_sample_companies(naics_code)
            naics_examples = self.beacon_data[self.beacon_data['naics_code'].astype(str) == naics_code]
            
            # Show sample weights if available
            if 'sample_weight' in naics_examples.columns:
                sample_weights = naics_examples['sample_weight'].unique()
                print(f"   💰 Sample weights: {sample_weights}")
            
            print(f"\n📋 Sample Company Types ({len(samples)} unique descriptions):")
            for j, sample in enumerate(samples, 1):
                print(f"   {j:2d}. {sample}")
            
            # Get similarity-ranked labels
            ranked_labels = self.get_similarity_ranked_labels(naics_code, samples)
            
            print(f"\n🎯 Insurance Labels (ranked by similarity to samples):")
            print(f"   📝 Based on: '{self.get_naics_description(naics_code)} {' '.join(samples[:2])}'")
            for j, (label, similarity) in enumerate(ranked_labels[:15], 1):
                print(f"   {j:2d}. {label:<40} (similarity: {similarity:.4f})")
            
            print(f"\n" + "-"*60)
            print(f"🔽 More options:")
            print(f"   16. Show all 220 labels")
            print(f"   17. Custom search term")
            print(f"   s.  Skip this NAICS code")
            print(f"   q.  Quit and save progress")
            
            # Get user choice
            while True:
                choice = input(f"\n👉 Select option (1-17, s, q): ").strip().lower()
                
                if choice == 'q':
                    print("💾 Saving progress and quitting...")
                    self._save_progress()
                    return
                
                elif choice == 's':
                    print(f"⏭️  Skipping NAICS code {naics_code}")
                    self.progress["skipped_naics"].append(naics_code)
                    self.progress["session_stats"]["total_skipped"] += 1
                    break
                
                elif choice == '16':
                    print(f"\n📜 All 220 Insurance Labels:")
                    for j, (label, similarity) in enumerate(ranked_labels, 1):
                        print(f"   {j:3d}. {label:<40} (similarity: {similarity:.4f})")
                    print(f"\n" + "-"*60)
                    continue
                
                elif choice == '17':
                    print(f"\n🔍 Custom Search Term")
                    print(f"💡 Enter your own search term to find similar insurance labels")
                    print(f"   Examples: 'metal manufacturing', 'fabricated metal products', 'custom metalwork'")
                    
                    custom_term = input(f"\n📝 Enter search term: ").strip()
                    
                    if custom_term:
                        custom_ranked = self.get_custom_search_ranked_labels(custom_term)
                        
                        # Handle custom search results
                        mapping_made = self._handle_custom_search(naics_code, custom_term, custom_ranked, count)
                        
                        if mapping_made == 'quit':
                            return
                        elif mapping_made == 'skip':
                            break  # Move to next NAICS
                        elif mapping_made == 'mapped':
                            break  # Move to next NAICS
                        # If mapping_made == 'back', continue to original rankings
                    else:
                        print(f"❌ No search term entered. Returning to original rankings.")
                    
                    continue
                
                elif choice.isdigit():
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(ranked_labels):
                        selected_label = ranked_labels[choice_num - 1][0]
                        similarity_score = ranked_labels[choice_num - 1][1]
                        
                        print(f"✅ Mapping: {naics_code} → {selected_label}")
                        print(f"   Similarity: {similarity_score:.4f}")
                        print(f"   Training examples: {count:,}")
                        
                        # Save mapping
                        self.progress["mapped_naics"][naics_code] = {
                            "insurance_label": selected_label,
                            "similarity_score": float(similarity_score),
                            "training_examples": count,
                            "mapped_at": datetime.now().isoformat()
                        }
                        self.progress["session_stats"]["total_mapped"] += 1
                        
                        # Auto-save progress
                        self._save_progress()
                        break
                    else:
                        print(f"❌ Invalid choice. Please enter 1-{len(ranked_labels)}, s, or q")
                else:
                    print(f"❌ Invalid choice. Please enter 1-17, s, or q")
        
        print(f"\n🎉 Mapping session complete!")
        print(f"   Total mapped: {self.progress['session_stats']['total_mapped']}")
        print(f"   Total skipped: {self.progress['session_stats']['total_skipped']}")
        self._save_progress()
    
    def generate_expanded_training_set(self):
        """Generate expanded training set from mapped NAICS codes"""
        print("\n🚀 Generating expanded training set...")
        
        if not self.progress["mapped_naics"]:
            print("❌ No new mappings found. Complete mapping session first.")
            return
        
        # Combine original master map with new mappings
        expanded_examples = []
        
        # From new mappings
        for naics_code, mapping_info in self.progress["mapped_naics"].items():
            examples = self.beacon_data[self.beacon_data['naics_code'].astype(str) == naics_code]
            
            for _, row in examples.iterrows():
                expanded_examples.append({
                    'company_description': row.get('company_description', row.get('TEXT', '')),
                    'naics_code': naics_code,
                    'insurance_label': mapping_info['insurance_label'],
                    'source': 'beacon_strategic_mapping',
                    'similarity_score': mapping_info['similarity_score']
                })
        
        print(f"✅ Generated {len(expanded_examples)} new training examples")
        print(f"   From {len(self.progress['mapped_naics'])} mapped NAICS codes")
        
        # Save expanded training set
        output_path = "data/processed/beacon_expanded_training_set.csv"
        pd.DataFrame(expanded_examples).to_csv(output_path, index=False)
        print(f"💾 Saved to: {output_path}")
        
        return expanded_examples
    
    def run_complete_workflow(self):
        """Run the complete strategic mapping workflow"""
        print("🎯 BEACON Dataset Strategic Mapping - Complete Workflow")
        print("=" * 70)
        
        # Step 1: Load data
        if not self.load_beacon_dataset():
            return
        
        # Step 2: Load master map
        mapped_codes = self.load_master_map()
        
        # Step 3: Analyze frequencies
        self.analyze_naics_frequency(mapped_codes)
        
        # Step 4: Interactive mapping
        self.interactive_mapping_session()
        
        # Step 5: Generate training set
        self.generate_expanded_training_set()
        
        print("\n🎉 Strategic mapping workflow complete!")


def main():
    """Main execution"""
    mapper = BeaconDatasetMapper()
    mapper.run_complete_workflow()


if __name__ == "__main__":
    main()