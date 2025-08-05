"""
Hierarchical TF-IDF SIC Classifier
Matches step-by-step through SIC hierarchy: Division → Major Group → Industry Group → SIC Code
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from ..preprocessing.text_utils import preprocess_text


class HierarchicalTFIDFClassifier:
    """Hierarchical TF-IDF classifier following SIC code structure"""
    
    def __init__(self):
        self.sic_data = None
        self.hierarchy = {}
        self.vectorizers = {}
        self.tfidf_matrices = {}
        self.level_data = {}
        
    def load_and_process_data(self, sic_file='data/input/osha_sic_rich.csv'):
        """Load and process the rich OSHA SIC dataset"""
        print("📥 Loading rich OSHA SIC dataset for hierarchical matching...")
        
        # Read and consolidate the data (reuse logic from previous classifier)
        raw_data = pd.read_csv(sic_file, dtype=str)
        print(f"✅ Loaded {len(raw_data)} raw entries")
        
        # Group by SIC code and consolidate examples
        sic_consolidated = []
        current_sic = None
        current_data = None
        examples = []
        
        for _, row in raw_data.iterrows():
            sic_code = row['sic_code']
            
            if pd.notna(sic_code) and sic_code != current_sic:
                # Save previous SIC if it exists
                if current_sic is not None:
                    current_data['examples'] = '; '.join(examples)
                    sic_consolidated.append(current_data)
                
                # Start new SIC
                current_sic = sic_code
                current_data = {
                    'sic_code': sic_code,
                    'sic_name': row['sic_name'],
                    'group_code': row['group_code'], 
                    'group_name': row['group_name'],
                    'division_code': row['division_code'],
                    'division_name': row['division_name'],
                    'description': row['sic_description'] if pd.notna(row['sic_description']) else '',
                    'examples': ''
                }
                examples = []
                
                if pd.notna(row['sic_examples']) and row['sic_examples'].strip():
                    examples.append(row['sic_examples'].strip())
            else:
                if pd.notna(row['sic_examples']) and row['sic_examples'].strip():
                    examples.append(row['sic_examples'].strip())
        
        # Don't forget the last SIC
        if current_sic is not None:
            current_data['examples'] = '; '.join(examples)
            sic_consolidated.append(current_data)
        
        self.sic_data = pd.DataFrame(sic_consolidated)
        print(f"✅ Consolidated to {len(self.sic_data)} unique SIC codes")
        
        # Build hierarchy structure
        self._build_hierarchy()
        
    def _build_hierarchy(self):
        """Build the hierarchical structure from SIC data"""
        print("🏗️ Building SIC hierarchy...")
        
        # Build nested hierarchy
        self.hierarchy = {}
        
        for _, row in self.sic_data.iterrows():
            div_code = row['division_code']
            div_name = row['division_name']
            mg_code = row['group_code']
            mg_name = row['group_name']
            sic_code = row['sic_code']
            sic_name = row['sic_name']
            
            # Get first 3 digits for industry group (assuming 4-digit SIC codes)
            if len(str(sic_code)) >= 3:
                ig_code = str(sic_code)[:3]
            else:
                ig_code = str(sic_code)
                
            # Build hierarchy structure
            if div_code not in self.hierarchy:
                self.hierarchy[div_code] = {
                    'name': div_name,
                    'major_groups': {}
                }
            
            if mg_code not in self.hierarchy[div_code]['major_groups']:
                self.hierarchy[div_code]['major_groups'][mg_code] = {
                    'name': mg_name,
                    'industry_groups': {}
                }
            
            if ig_code not in self.hierarchy[div_code]['major_groups'][mg_code]['industry_groups']:
                self.hierarchy[div_code]['major_groups'][mg_code]['industry_groups'][ig_code] = {
                    'sic_codes': {}
                }
            
            self.hierarchy[div_code]['major_groups'][mg_code]['industry_groups'][ig_code]['sic_codes'][sic_code] = {
                'name': sic_name,
                'description': row['description'],
                'examples': row['examples']
            }
        
        # Print hierarchy stats
        n_divisions = len(self.hierarchy)
        n_major_groups = sum(len(div['major_groups']) for div in self.hierarchy.values())
        n_industry_groups = sum(
            len(mg['industry_groups']) 
            for div in self.hierarchy.values() 
            for mg in div['major_groups'].values()
        )
        n_sic_codes = sum(
            len(ig['sic_codes'])
            for div in self.hierarchy.values()
            for mg in div['major_groups'].values() 
            for ig in mg['industry_groups'].values()
        )
        
        print(f"✅ Hierarchy: {n_divisions} divisions → {n_major_groups} major groups → {n_industry_groups} industry groups → {n_sic_codes} SIC codes")
        
    def _create_text_representation(self, items, text_fields):
        """Create comprehensive text representation for TF-IDF"""
        texts = []
        codes = []
        
        for code, data in items.items():
            text_parts = []
            
            for field, weight in text_fields.items():
                if field in data and pd.notna(data[field]) and data[field]:
                    text_content = str(data[field]).strip().strip('"')
                    # Repeat content based on weight for TF-IDF importance
                    text_parts.extend([text_content] * weight)
            
            if text_parts:
                full_text = ' '.join(text_parts)
                processed_text = preprocess_text(full_text, to_lower=True, punct_remove=True)
                texts.append(processed_text)
                codes.append(code)
        
        return texts, codes
    
    def create_hierarchical_vectorizers(self):
        """Create TF-IDF vectorizers for each level of the hierarchy"""
        print("🚀 Creating hierarchical TF-IDF vectorizers...")
        
        # Level 1: Divisions
        print("   📂 Level 1: Divisions")
        division_items = {
            code: {'name': data['name']} 
            for code, data in self.hierarchy.items()
        }
        
        div_texts, div_codes = self._create_text_representation(
            division_items, 
            {'name': 3}  # Weight division names heavily
        )
        
        self._create_vectorizer('divisions', div_texts, div_codes)
        
        # Level 2: Major Groups (per division)
        print("   📂 Level 2: Major Groups")
        for div_code, div_data in self.hierarchy.items():
            mg_items = {
                code: {'name': data['name']}
                for code, data in div_data['major_groups'].items()
            }
            
            if mg_items:
                mg_texts, mg_codes = self._create_text_representation(
                    mg_items,
                    {'name': 3}
                )
                self._create_vectorizer(f'major_groups_{div_code}', mg_texts, mg_codes)
        
        # Level 3: Industry Groups (per major group)
        print("   📂 Level 3: Industry Groups") 
        for div_code, div_data in self.hierarchy.items():
            for mg_code, mg_data in div_data['major_groups'].items():
                # For industry groups, we use the SIC codes within them to build text
                ig_items = {}
                for ig_code, ig_data in mg_data['industry_groups'].items():
                    # Combine all SIC names/descriptions within this industry group
                    sic_texts = []
                    for sic_code, sic_data in ig_data['sic_codes'].items():
                        if sic_data['name']:
                            sic_texts.append(sic_data['name'])
                        if sic_data['description']:
                            sic_texts.append(sic_data['description'][:200])  # Truncate long descriptions
                    
                    ig_items[ig_code] = {
                        'combined_text': ' '.join(sic_texts)
                    }
                
                if ig_items:
                    ig_texts, ig_codes = self._create_text_representation(
                        ig_items,
                        {'combined_text': 1}
                    )
                    self._create_vectorizer(f'industry_groups_{div_code}_{mg_code}', ig_texts, ig_codes)
        
        # Level 4: SIC Codes (per industry group)
        print("   📂 Level 4: SIC Codes")
        for div_code, div_data in self.hierarchy.items():
            for mg_code, mg_data in div_data['major_groups'].items():
                for ig_code, ig_data in mg_data['industry_groups'].items():
                    sic_items = ig_data['sic_codes']
                    
                    if sic_items:
                        sic_texts, sic_codes = self._create_text_representation(
                            sic_items,
                            {'name': 3, 'description': 1, 'examples': 2}
                        )
                        self._create_vectorizer(f'sic_codes_{div_code}_{mg_code}_{ig_code}', sic_texts, sic_codes)
        
        print(f"✅ Created {len(self.vectorizers)} hierarchical TF-IDF vectorizers")
    
    def _create_vectorizer(self, key, texts, codes):
        """Create and store a TF-IDF vectorizer for a specific level"""
        if not texts:
            return
        
        # Adjust parameters based on corpus size
        corpus_size = len(texts)
        
        # Dynamic parameter adjustment
        if corpus_size <= 5:
            min_df = 1
            max_df = 1.0  # Allow all terms for very small corpora
            max_features = min(1000, corpus_size * 100)
        elif corpus_size <= 20:
            min_df = 1
            max_df = 0.95
            max_features = min(2000, corpus_size * 50)
        else:
            min_df = 1
            max_df = 0.9
            max_features = min(5000, corpus_size * 25)
            
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=max_features,
            ngram_range=(1, 2),  # Smaller ngrams for focused matching
            min_df=min_df,
            max_df=max_df,
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            self.vectorizers[key] = vectorizer
            self.tfidf_matrices[key] = tfidf_matrix
            self.level_data[key] = {
                'texts': texts,
                'codes': codes
            }
            
        except ValueError as e:
            print(f"   ⚠️  Skipping {key} due to insufficient vocabulary (corpus size: {corpus_size})")
            # For very small corpora, we can fall back to exact string matching or skip
    
    def _match_at_level(self, label, level_key, top_k=3):
        """Match label at a specific hierarchy level"""
        if level_key not in self.vectorizers:
            return []
            
        # Preprocess label
        processed_label = preprocess_text(label, to_lower=True, punct_remove=True)
        
        # Transform to TF-IDF
        vectorizer = self.vectorizers[level_key]
        label_tfidf = vectorizer.transform([processed_label])
        
        # Calculate similarities
        tfidf_matrix = self.tfidf_matrices[level_key]
        similarities = linear_kernel(label_tfidf, tfidf_matrix)[0]
        
        # Get top matches
        best_indices = np.argsort(similarities)[::-1][:top_k]
        codes = self.level_data[level_key]['codes']
        
        matches = []
        for idx in best_indices:
            if similarities[idx] > 0.05:  # Very low threshold for hierarchical matching
                matches.append({
                    'code': codes[idx],
                    'similarity': float(similarities[idx])
                })
        
        return matches
    
    def classify_hierarchical(self, label, max_divisions=2, max_major_groups=3, max_industry_groups=3):
        """Classify a label through the SIC hierarchy step by step"""
        result = {
            'label': label,
            'path': [],
            'final_matches': []
        }
        
        # Level 1: Match divisions
        division_matches = self._match_at_level(label, 'divisions', max_divisions)
        if not division_matches:
            return result
        
        result['path'].append(('divisions', division_matches))
        
        # Level 2: Match major groups within selected divisions
        mg_matches = []
        for div_match in division_matches:
            div_code = div_match['code']
            mg_level_key = f'major_groups_{div_code}'
            
            level_mg_matches = self._match_at_level(label, mg_level_key, max_major_groups)
            for mg_match in level_mg_matches:
                mg_match['division'] = div_code
                mg_match['div_similarity'] = div_match['similarity']
                # Combined similarity (you could use different combination strategies)
                mg_match['combined_similarity'] = mg_match['similarity'] * div_match['similarity']
                mg_matches.append(mg_match)
        
        # Sort by combined similarity and take top matches
        mg_matches.sort(key=lambda x: x['combined_similarity'], reverse=True)
        mg_matches = mg_matches[:max_major_groups]
        
        if not mg_matches:
            return result
            
        result['path'].append(('major_groups', mg_matches))
        
        # Level 3: Match industry groups within selected major groups
        ig_matches = []
        for mg_match in mg_matches:
            div_code = mg_match['division']
            mg_code = mg_match['code']
            ig_level_key = f'industry_groups_{div_code}_{mg_code}'
            
            level_ig_matches = self._match_at_level(label, ig_level_key, max_industry_groups)
            for ig_match in level_ig_matches:
                ig_match['division'] = div_code
                ig_match['major_group'] = mg_code
                ig_match['mg_similarity'] = mg_match['combined_similarity']
                ig_match['combined_similarity'] = ig_match['similarity'] * mg_match['combined_similarity']
                ig_matches.append(ig_match)
        
        ig_matches.sort(key=lambda x: x['combined_similarity'], reverse=True)
        ig_matches = ig_matches[:max_industry_groups]
        
        if not ig_matches:
            return result
            
        result['path'].append(('industry_groups', ig_matches))
        
        # Level 4: Match SIC codes within selected industry groups
        sic_matches = []
        for ig_match in ig_matches:
            div_code = ig_match['division']
            mg_code = ig_match['major_group']
            ig_code = ig_match['code']
            sic_level_key = f'sic_codes_{div_code}_{mg_code}_{ig_code}'
            
            level_sic_matches = self._match_at_level(label, sic_level_key, 5)
            for sic_match in level_sic_matches:
                sic_match['division'] = div_code
                sic_match['major_group'] = mg_code
                sic_match['industry_group'] = ig_code
                sic_match['ig_similarity'] = ig_match['combined_similarity']
                sic_match['final_similarity'] = sic_match['similarity'] * ig_match['combined_similarity']
                
                # Get full SIC information
                sic_info = self.hierarchy[div_code]['major_groups'][mg_code]['industry_groups'][ig_code]['sic_codes'][sic_match['code']]
                sic_match['name'] = sic_info['name']
                sic_matches.append(sic_match)
        
        sic_matches.sort(key=lambda x: x['final_similarity'], reverse=True)
        result['final_matches'] = sic_matches[:5]
        
        return result
    
    def display_hierarchical_result(self, result):
        """Display the hierarchical classification result"""
        print(f"\n🎯 HIERARCHICAL CLASSIFICATION: {result['label']}")
        print("=" * 70)
        
        # Show the path through hierarchy
        for level_name, matches in result['path']:
            print(f"\n📂 {level_name.upper()}:")
            for i, match in enumerate(matches[:3]):  # Show top 3
                print(f"   {i+1}. {match['code']} (similarity: {match.get('combined_similarity', match['similarity']):.3f})")
        
        # Show final SIC code matches
        if result['final_matches']:
            print(f"\n🎯 FINAL SIC CODE MATCHES:")
            for i, match in enumerate(result['final_matches'][:5]):
                path = f"{match['division']}-{match['major_group']}-{match['industry_group']}-{match['code']}"
                print(f"   {i+1}. SIC {match['code']}: {match['name']}")
                print(f"      Path: {path}")
                print(f"      Final similarity: {match['final_similarity']:.4f}")
        else:
            print("\n❌ No final SIC code matches found")


def main():
    """Test the hierarchical TF-IDF classifier"""
    
    # Test labels
    test_labels = [
        "Landscaping Services",
        "Tree Services - Pruning / Removal", 
        "Sheet Metal Services",
        "Real Estate Services",
        "Software Development Services",
        "Rubber Manufacturing"
    ]
    
    print("🎯 Testing Hierarchical TF-IDF SIC Classifier")
    print("=" * 60)
    
    # Initialize classifier
    classifier = HierarchicalTFIDFClassifier()
    
    # Load and process data
    classifier.load_and_process_data()
    
    # Create hierarchical vectorizers
    classifier.create_hierarchical_vectorizers()
    
    # Test each label
    for label in test_labels:
        result = classifier.classify_hierarchical(label)
        classifier.display_hierarchical_result(result)
        print("\n" + "="*70)
    
    print("\n✅ Hierarchical TF-IDF classification complete!")


if __name__ == "__main__":
    main() 