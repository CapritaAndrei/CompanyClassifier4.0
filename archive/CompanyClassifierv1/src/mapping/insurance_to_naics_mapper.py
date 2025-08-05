"""
Insurance Taxonomy to NAICS Mapper
Maps the 220 insurance taxonomy labels to NAICS codes using semantic similarity.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle


class InsuranceToNAICSMapper:
    """Maps insurance taxonomy labels to NAICS codes using semantic similarity."""
    
    def __init__(self):
        self.model = None
        self.naics_data = None
        self.naics_embeddings = None
        self.insurance_labels = None
        self.cache_dir = "data/cache/naics_mappings"
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def load_naics_data(self, naics_file_path="NAICS/2022_NAICS_Index_File.xlsx"):
        """Load NAICS index data."""
        print("Loading NAICS data...")
        self.naics_data = pd.read_excel(naics_file_path)
        print(f"Loaded {len(self.naics_data)} NAICS entries")
        return self.naics_data
    
    def load_insurance_taxonomy(self, taxonomy_file="data/input/insurance_taxonomy.xlsx"):
        """Load insurance taxonomy labels."""
        print("Loading insurance taxonomy...")
        self.insurance_labels = pd.read_excel(taxonomy_file)
        print(f"Loaded {len(self.insurance_labels)} insurance labels")
        return self.insurance_labels
    
    def load_embedding_model(self, model_name="all-MiniLM-L6-v2"):
        """Load sentence transformer model for semantic similarity."""
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        return self.model
    
    def load_naics_embeddings(self):
        """Load pre-computed NAICS embeddings."""
        cache_file = os.path.join(self.cache_dir, "naics_embeddings.pkl")
        
        if os.path.exists(cache_file):
            print("Loading cached NAICS embeddings...")
            with open(cache_file, 'rb') as f:
                self.naics_embeddings = pickle.load(f)
            return self.naics_embeddings
        else:
            print("NAICS embeddings not found. Creating them...")
            return self.create_naics_embeddings()
    
    def create_naics_embeddings(self):
        """Create embeddings for all NAICS descriptions."""
        print("Creating NAICS embeddings...")
        descriptions = self.naics_data['INDEX ITEM DESCRIPTION'].fillna('').tolist()
        self.naics_embeddings = self.model.encode(descriptions, show_progress_bar=True)
        
        # Cache the embeddings
        cache_file = os.path.join(self.cache_dir, "naics_embeddings.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(self.naics_embeddings, f)
        print(f"Cached NAICS embeddings to {cache_file}")
        
        return self.naics_embeddings
    
    def find_best_naics_matches_for_label(self, label_info, top_k=5):
        """Find best NAICS matches for an insurance label."""
        # Combine label, definition, and keywords for better matching
        description_parts = [label_info['label']]
        
        if pd.notna(label_info['Definition']):
            description_parts.append(label_info['Definition'])
        
        if pd.notna(label_info['Keywords']):
            description_parts.append(label_info['Keywords'])
        
        # Create comprehensive description
        full_description = ' '.join(description_parts)
        
        # Create embedding for the description
        desc_embedding = self.model.encode([full_description])
        
        # Calculate similarities
        similarities = cosine_similarity(desc_embedding, self.naics_embeddings)[0]
        
        # Get top matches
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        matches = []
        for idx in top_indices:
            matches.append({
                'naics_code': self.naics_data.iloc[idx]['NAICS22'],
                'naics_description': self.naics_data.iloc[idx]['INDEX ITEM DESCRIPTION'],
                'similarity_score': similarities[idx]
            })
        
        return matches
    
    def map_all_insurance_labels_to_naics(self):
        """Map all insurance taxonomy labels to NAICS codes."""
        print(f"Mapping {len(self.insurance_labels)} insurance labels to NAICS codes...")
        
        results = []
        for idx, row in self.insurance_labels.iterrows():
            if idx % 20 == 0:
                print(f"Processing {idx}/{len(self.insurance_labels)}...")
            
            # Find best matches
            matches = self.find_best_naics_matches_for_label(row, top_k=3)
            
            # Store results
            result = {
                'insurance_label': row['label'],
                'insurance_definition': row['Definition'] if pd.notna(row['Definition']) else '',
                'insurance_keywords': row['Keywords'] if pd.notna(row['Keywords']) else '',
                'best_naics_code': matches[0]['naics_code'],
                'best_naics_description': matches[0]['naics_description'],
                'best_similarity_score': matches[0]['similarity_score'],
                'second_naics_code': matches[1]['naics_code'] if len(matches) > 1 else None,
                'second_naics_description': matches[1]['naics_description'] if len(matches) > 1 else None,
                'second_similarity_score': matches[1]['similarity_score'] if len(matches) > 1 else None,
                'third_naics_code': matches[2]['naics_code'] if len(matches) > 2 else None,
                'third_naics_description': matches[2]['naics_description'] if len(matches) > 2 else None,
                'third_similarity_score': matches[2]['similarity_score'] if len(matches) > 2 else None,
            }
            results.append(result)
        
        results_df = pd.DataFrame(results)
        return results_df
    
    def save_insurance_naics_mappings(self, mappings_df, filename="insurance_to_naics_mappings.csv"):
        """Save the insurance to NAICS mappings."""
        filepath = os.path.join(self.cache_dir, filename)
        mappings_df.to_csv(filepath, index=False)
        print(f"Saved insurance to NAICS mappings to {filepath}")
        return filepath
    
    def analyze_mapping_quality(self, mappings_df):
        """Analyze the quality of insurance to NAICS mappings."""
        print("\n=== Insurance to NAICS Mapping Quality ===")
        print(f"Total insurance labels mapped: {len(mappings_df)}")
        print(f"Average similarity score: {mappings_df['best_similarity_score'].mean():.3f}")
        
        # Confidence level distribution
        high_confidence = (mappings_df['best_similarity_score'] > 0.7).sum()
        medium_confidence = ((mappings_df['best_similarity_score'] >= 0.5) & 
                           (mappings_df['best_similarity_score'] <= 0.7)).sum()
        low_confidence = (mappings_df['best_similarity_score'] < 0.5).sum()
        
        print(f"High confidence mappings (>0.7): {high_confidence} ({high_confidence/len(mappings_df)*100:.1f}%)")
        print(f"Medium confidence mappings (0.5-0.7): {medium_confidence} ({medium_confidence/len(mappings_df)*100:.1f}%)")
        print(f"Low confidence mappings (<0.5): {low_confidence} ({low_confidence/len(mappings_df)*100:.1f}%)")
        
        # Show some examples
        print("\n=== Top 10 Best Matches ===")
        top_matches = mappings_df.nlargest(10, 'best_similarity_score')
        for _, row in top_matches.iterrows():
            print(f"'{row['insurance_label']}' → {row['best_naics_code']}: '{row['best_naics_description'][:60]}...' (Score: {row['best_similarity_score']:.3f})")
        
        print("\n=== Lowest 10 Matches (Need Review) ===")
        low_matches = mappings_df.nsmallest(10, 'best_similarity_score')
        for _, row in low_matches.iterrows():
            print(f"'{row['insurance_label']}' → {row['best_naics_code']}: '{row['best_naics_description'][:60]}...' (Score: {row['best_similarity_score']:.3f})")


def create_insurance_to_naics_mappings():
    """Main function to create insurance taxonomy to NAICS mappings."""
    mapper = InsuranceToNAICSMapper()
    
    # Load data and model
    mapper.load_naics_data()
    mapper.load_insurance_taxonomy()
    mapper.load_embedding_model()
    
    # Load or create NAICS embeddings
    mapper.load_naics_embeddings()
    
    # Create mappings
    mappings = mapper.map_all_insurance_labels_to_naics()
    
    # Analyze quality
    mapper.analyze_mapping_quality(mappings)
    
    # Save results
    mapper.save_insurance_naics_mappings(mappings)
    
    return mappings


if __name__ == "__main__":
    mappings = create_insurance_to_naics_mappings() 