"""
TF-IDF SIC Classifier using rich OSHA dataset
Maps insurance labels to SIC codes using TF-IDF similarity
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import re
from ..preprocessing.text_utils import preprocess_text


class TFIDFSICClassifier:
    """TF-IDF classifier for mapping insurance labels to SIC codes"""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.sic_data = None
        self.sic_texts = None
        self.sic_codes = None
        # Sector-specific data
        self.sector_vectorizers = {}
        self.sector_matrices = {}
        self.sector_data = {}
        
    def load_and_process_osha_data(self, file_path='data/input/osha_sic_rich.csv'):
        """Load and process the rich OSHA SIC dataset"""
        print("📥 Loading rich OSHA SIC dataset...")
        
        # Read the CSV with proper handling of newlines in examples
        raw_data = pd.read_csv(file_path, dtype=str)
        print(f"✅ Loaded {len(raw_data)} raw entries")
        
        # Group by SIC code and consolidate
        sic_consolidated = []
        
        current_sic = None
        current_data = None
        examples = []
        
        for _, row in raw_data.iterrows():
            sic_code = row['sic_code']
            
            # If this is a new SIC code or the first row
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
                
                # Add example if present
                if pd.notna(row['sic_examples']) and row['sic_examples'].strip():
                    examples.append(row['sic_examples'].strip())
            else:
                # This is a continuation line with more examples
                if pd.notna(row['sic_examples']) and row['sic_examples'].strip():
                    examples.append(row['sic_examples'].strip())
        
        # Don't forget the last SIC
        if current_sic is not None:
            current_data['examples'] = '; '.join(examples)
            sic_consolidated.append(current_data)
        
        self.sic_data = pd.DataFrame(sic_consolidated)
        print(f"✅ Consolidated to {len(self.sic_data)} unique SIC codes")
        
        # Create comprehensive text for each SIC code
        self.sic_texts = []
        self.sic_codes = []
        
        for _, row in self.sic_data.iterrows():
            # Combine all text fields for TF-IDF
            text_parts = []
            
            # SIC name (high weight by repeating)
            sic_name = row['sic_name'] if pd.notna(row['sic_name']) else ''
            if sic_name:
                text_parts.extend([sic_name] * 3)  # Repeat 3 times for higher weight
            
            # Description
            description = row['description'] if pd.notna(row['description']) else ''
            if description:
                # Clean description (remove quotes, extra spaces)
                description = description.strip('"').strip()
                text_parts.append(description)
            
            # Examples (high weight by repeating)
            examples = row['examples'] if pd.notna(row['examples']) else ''
            if examples:
                # Split examples and add each one twice for higher weight
                example_list = [ex.strip() for ex in examples.split(';') if ex.strip()]
                text_parts.extend(example_list * 2)  # Repeat examples twice
            
            # Group and division names for context
            group_name = row['group_name'] if pd.notna(row['group_name']) else ''
            division_name = row['division_name'] if pd.notna(row['division_name']) else ''
            
            if group_name:
                text_parts.append(group_name)
            if division_name:
                text_parts.append(division_name)
            
            # Join all parts
            full_text = ' '.join(text_parts)
            
            # Preprocess the text
            processed_text = preprocess_text(full_text, to_lower=True, punct_remove=True)
            
            self.sic_texts.append(processed_text)
            self.sic_codes.append(row['sic_code'])
        
        print(f"✅ Created {len(self.sic_texts)} TF-IDF text representations")
        
    def create_tfidf_vectorizer(self):
        """Create and fit TF-IDF vectorizer on SIC texts"""
        print("🚀 Creating TF-IDF vectorizer...")
        
        # Use TF-IDF with good parameters for business text
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=10000,  # Increase for better vocabulary coverage
            ngram_range=(1, 3),  # Include unigrams, bigrams, and trigrams
            min_df=2,           # Ignore terms that appear in fewer than 2 documents
            max_df=0.8,         # Ignore terms that appear in more than 80% of documents
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'  # Include alphanumeric tokens
        )
        
        # Fit on SIC texts
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.sic_texts)
        print(f"✅ TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        
        # Create sector-specific vectorizers
        self._create_sector_vectorizers()
        
    def _create_sector_vectorizers(self):
        """Create TF-IDF vectorizers for specific sectors"""
        print("🎯 Creating sector-specific TF-IDF vectorizers...")
        
        # Define sector mappings
        sector_divisions = {
            'services': ['I'],  # Services
            'finance_insurance': ['H'],  # Finance, Insurance, Real Estate  
            'manufacturing': ['D'],  # Manufacturing
            'construction': ['C'],  # Construction
            'agriculture': ['A'],  # Agriculture, Forestry, Fishing
            'mining': ['B'],  # Mining
            'trade': ['F', 'G'],  # Wholesale Trade, Retail Trade
            'transportation': ['E']  # Transportation, Communications, Utilities
        }
        
        for sector_name, divisions in sector_divisions.items():
            # Filter SIC data for this sector
            sector_mask = self.sic_data['division_code'].isin(divisions)
            sector_indices = sector_mask[sector_mask].index.tolist()
            
            if len(sector_indices) > 10:  # Only create if we have enough data
                sector_texts = [self.sic_texts[i] for i in sector_indices]
                sector_codes = [self.sic_codes[i] for i in sector_indices]
                
                # Create sector-specific vectorizer
                vectorizer = TfidfVectorizer(
                    stop_words='english',
                    max_features=5000,  # Smaller for focused sectors
                    ngram_range=(1, 3),
                    min_df=1,  # Lower threshold for smaller datasets
                    max_df=0.9,
                    lowercase=True,
                    token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'
                )
                
                sector_matrix = vectorizer.fit_transform(sector_texts)
                
                self.sector_vectorizers[sector_name] = vectorizer
                self.sector_matrices[sector_name] = sector_matrix
                self.sector_data[sector_name] = {
                    'indices': sector_indices,
                    'texts': sector_texts,
                    'codes': sector_codes
                }
                
                print(f"   ✅ {sector_name}: {len(sector_codes)} SIC codes")
        
        print(f"✅ Created {len(self.sector_vectorizers)} sector-specific vectorizers")
    
    def _determine_sectors(self, label):
        """Determine multiple plausible sectors for an insurance label, ordered by likelihood"""
        label_lower = label.lower()
        sectors = []
        
        # Agriculture indicators - specific activities that are clearly agricultural
        if any(word in label_lower for word in ['farming', 'agriculture', 'forestry', 'fishing', 'ranch', 'landscaping', 'landscape', 'tree services', 'tree removal', 'tree trimming', 'ornamental', 'nursery', 'lawn', 'garden', 'irrigation', 'crop', 'livestock']):
            sectors.append('agriculture')
        
        # Manufacturing indicators - production and fabrication
        if any(word in label_lower for word in ['manufacturing', 'production', 'fabrication', 'assembly', 'rubber manufacturing', 'metal', 'steel', 'plastic', 'chemical', 'textile']):
            sectors.append('manufacturing')
            
        # Construction indicators - building and infrastructure
        if any(word in label_lower for word in ['construction', 'building', 'contractor', 'roofing', 'concrete', 'electrical work', 'plumbing', 'hvac', 'carpentry', 'masonry']):
            sectors.append('construction')
        
        # Finance/Insurance indicators - financial services
        if any(word in label_lower for word in ['insurance', 'financial', 'banking', 'real estate', 'investment', 'mortgage', 'credit', 'loan']):
            sectors.append('finance_insurance')
            
        # Mining indicators - extraction activities
        if any(word in label_lower for word in ['mining', 'extraction', 'quarry', 'drilling', 'oil', 'gas', 'coal', 'gravel', 'sand mining']):
            sectors.append('mining')
        
        # Trade indicators - buying/selling activities
        if any(word in label_lower for word in ['wholesale', 'retail', 'sales', 'distribution', 'trading', 'dealer', 'supplier']):
            sectors.append('trade')
            
        # Transportation indicators - moving goods/people
        if any(word in label_lower for word in ['transportation', 'shipping', 'logistics', 'trucking', 'delivery', 'freight', 'courier', 'moving']):
            sectors.append('transportation')
        
        # Services indicators - but be more specific about service types
        service_indicators = ['service', 'services', 'consulting', 'management', 'repair', 'maintenance', 'cleaning', 'security', 'legal', 'accounting', 'advertising', 'consulting']
        if any(word in label_lower for word in service_indicators):
            # Only add services if it's not already covered by a more specific sector
            if not sectors:  # No specific sector identified yet
                sectors.append('services')
            else:
                # Add services as secondary option for labels that could be both
                # e.g., "Landscaping Services" could be agriculture OR services
                if 'services' not in sectors:
                    sectors.append('services')
        
        # Special cases where multiple sectors are very likely
        if 'landscaping' in label_lower or 'landscape' in label_lower:
            # Landscaping could be agriculture (actual work) or services (consulting)
            if 'agriculture' not in sectors:
                sectors.append('agriculture')
            if 'services' not in sectors:
                sectors.append('services')
                
        if 'real estate' in label_lower:
            # Real estate is primarily finance_insurance but could involve construction services
            if 'finance_insurance' not in sectors:
                sectors.append('finance_insurance')
            if 'construction' not in sectors:
                sectors.append('construction')
        
        return sectors if sectors else None
    
    def _match_within_sectors(self, label, sectors):
        """Try matching within multiple sectors, return best overall match"""
        if not sectors:
            return None
            
        all_matches = []
        
        for sector in sectors:
            if sector not in self.sector_vectorizers:
                continue
                
            # Preprocess label
            processed_label = preprocess_text(label, to_lower=True, punct_remove=True)
            
            # Transform using sector-specific vectorizer
            vectorizer = self.sector_vectorizers[sector]
            label_tfidf = vectorizer.transform([processed_label])
            
            # Calculate similarities within sector
            sector_matrix = self.sector_matrices[sector]
            similarities = linear_kernel(label_tfidf, sector_matrix)[0]
            
            # Get best matches within this sector
            best_indices = np.argsort(similarities)[::-1][:3]  # Top 3 per sector
            sector_codes = self.sector_data[sector]['codes']
            sector_indices = self.sector_data[sector]['indices']
            
            for idx in best_indices:
                if similarities[idx] > 0.1:  # Lower threshold for sector searches
                    original_idx = sector_indices[idx]
                    all_matches.append({
                        'sic_code': sector_codes[idx],
                        'sic_name': self.sic_data.iloc[original_idx]['sic_name'],
                        'similarity': float(similarities[idx]),
                        'sector': sector
                    })
        
        # Sort all matches by similarity and return top 5
        all_matches.sort(key=lambda x: x['similarity'], reverse=True)
        return all_matches[:5] if all_matches else None
    
    def map_insurance_labels_to_sic(self, insurance_labels, use_sector_aware=True):
        """Map insurance labels to best matching SIC codes using TF-IDF"""
        print(f"🎯 Mapping {len(insurance_labels)} insurance labels to SIC codes...")
        if use_sector_aware:
            print("🎯 Using sector-aware matching for improved accuracy")
        
        mappings = []
        sector_stats = {}
        
        for label in insurance_labels:
            # Determine sectors for this label
            sectors = self._determine_sectors(label) if use_sector_aware else None
            
            best_matches = None
            matching_method = "full_corpus"
            
            # Try sector-specific matching first
            if sectors and use_sector_aware:
                sector_matches = self._match_within_sectors(label, sectors)
                if sector_matches and sector_matches[0]['similarity'] > 0.25:  # Lower threshold
                    best_matches = sector_matches
                    matching_method = f"sector_{','.join(sectors)}"
                    sector_stats[','.join(sectors)] = sector_stats.get(','.join(sectors), 0) + 1
            
            # Fall back to full corpus matching if sector matching failed
            if best_matches is None:
                # Preprocess insurance label
                processed_label = preprocess_text(label, to_lower=True, punct_remove=True)
                
                # Transform to TF-IDF vector
                label_tfidf = self.tfidf_vectorizer.transform([processed_label])
                
                # Calculate cosine similarities
                similarities = linear_kernel(label_tfidf, self.tfidf_matrix)[0]
                
                # Get best matches
                best_indices = np.argsort(similarities)[::-1][:5]  # Top 5 matches
                
                best_matches = [
                    {
                        'sic_code': self.sic_codes[idx],
                        'sic_name': self.sic_data.iloc[idx]['sic_name'],
                        'similarity': float(similarities[idx])
                    }
                    for idx in best_indices
                ]
            
            mappings.append({
                'insurance_label': label,
                'best_sic_code': best_matches[0]['sic_code'],
                'best_sic_name': best_matches[0]['sic_name'],
                'best_similarity': best_matches[0]['similarity'],
                'matching_method': matching_method,
                'sectors': sectors,
                'top_5_matches': best_matches
            })
        
        if use_sector_aware and sector_stats:
            print(f"📊 Sector matching statistics: {sector_stats}")
        
        return mappings
    
    def save_mappings(self, mappings, output_file='data/output/insurance_sic_mappings.csv'):
        """Save the insurance label to SIC mappings"""
        mapping_df = pd.DataFrame([
            {
                'insurance_label': m['insurance_label'],
                'sic_code': m['best_sic_code'],
                'sic_name': m['best_sic_name'],
                'similarity_score': m['best_similarity'],
                'matching_method': m['matching_method'],
                'sectors': m.get('sectors', '')
            }
            for m in mappings
        ])
        
        mapping_df.to_csv(output_file, index=False)
        print(f"💾 Saved mappings to {output_file}")
        
        return mapping_df
    
    def display_mapping_results(self, mappings, top_n=10):
        """Display the top mapping results"""
        print(f"\n📊 TOP {top_n} INSURANCE LABEL → SIC MAPPINGS:")
        print("=" * 80)
        
        # Sort by similarity score
        sorted_mappings = sorted(mappings, key=lambda x: x['best_similarity'], reverse=True)
        
        for i, mapping in enumerate(sorted_mappings[:top_n]):
            method_str = f"[{mapping['matching_method']}]" if 'matching_method' in mapping else ""
            print(f"\n{i+1:2d}. {mapping['insurance_label']} {method_str}")
            print(f"    → SIC {mapping['best_sic_code']}: {mapping['best_sic_name']}")
            print(f"    → Similarity: {mapping['best_similarity']:.4f}")
            
            # Show top 3 alternatives
            print("    → Alternatives:")
            for j, alt in enumerate(mapping['top_5_matches'][1:4]):  # Skip first (best match)
                print(f"       {j+2}. SIC {alt['sic_code']}: {alt['sic_name']} ({alt['similarity']:.4f})")
        
        print(f"\n📈 SUMMARY STATISTICS:")
        similarities = [m['best_similarity'] for m in mappings]
        print(f"   Average similarity: {np.mean(similarities):.4f}")
        print(f"   Median similarity: {np.median(similarities):.4f}")
        print(f"   Min similarity: {np.min(similarities):.4f}")
        print(f"   Max similarity: {np.max(similarities):.4f}")
        print(f"   High confidence (>0.5): {sum(1 for s in similarities if s > 0.5)}/{len(similarities)}")
        
        # Method breakdown
        if 'matching_method' in mappings[0]:
            method_counts = {}
            for m in mappings:
                method = m['matching_method']
                method_counts[method] = method_counts.get(method, 0) + 1
            print(f"   Matching methods: {method_counts}")


def main():
    """Test the TF-IDF SIC classifier"""
    
    # Load insurance labels
    insurance_bridge = pd.read_csv('data/cache_backup/naics_mappings/insurance_naics_bridge.csv')
    insurance_labels = insurance_bridge['insurance_label'].tolist()
    
    print(f"🎯 Testing TF-IDF SIC Classifier with {len(insurance_labels)} insurance labels")
    print("=" * 60)
    
    # Initialize classifier
    classifier = TFIDFSICClassifier()
    
    # Load and process OSHA data
    classifier.load_and_process_osha_data()
    
    # Create TF-IDF vectorizer
    classifier.create_tfidf_vectorizer()
    
    # Map insurance labels to SIC codes (with sector awareness)
    mappings = classifier.map_insurance_labels_to_sic(insurance_labels, use_sector_aware=True)
    
    # Display results
    classifier.display_mapping_results(mappings, top_n=15)
    
    # Save mappings
    mapping_df = classifier.save_mappings(mappings)
    
    print(f"\n✅ TF-IDF SIC classification complete!")
    return mappings


if __name__ == "__main__":
    main() 