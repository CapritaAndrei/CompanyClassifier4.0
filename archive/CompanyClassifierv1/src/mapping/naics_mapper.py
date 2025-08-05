"""
NAICS Mapping Module
Maps company Sector → Category → Niche combinations to NAICS codes using semantic similarity.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle


class NAICSMapper:
    """Maps company industry descriptions to NAICS codes using semantic similarity."""
    
    def __init__(self):
        self.model = None
        self.naics_data = None
        self.naics_embeddings = None
        self.cache_dir = "data/cache/naics_mappings"
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def load_naics_data(self, naics_file_path="NAICS/2022_NAICS_Index_File.xlsx"):
        """Load NAICS index data."""
        print("Loading NAICS data...")
        self.naics_data = pd.read_excel(naics_file_path)
        print(f"Loaded {len(self.naics_data)} NAICS entries")
        return self.naics_data
    
    def load_embedding_model(self, model_name="all-MiniLM-L6-v2"):
        """Load sentence transformer model for semantic similarity."""
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        return self.model
    
    def create_naics_embeddings(self, force_recreate=False):
        """Create embeddings for all NAICS descriptions."""
        cache_file = os.path.join(self.cache_dir, "naics_embeddings.pkl")
        
        if not force_recreate and os.path.exists(cache_file):
            print("Loading cached NAICS embeddings...")
            with open(cache_file, 'rb') as f:
                self.naics_embeddings = pickle.load(f)
            return self.naics_embeddings
        
        print("Creating NAICS embeddings...")
        descriptions = self.naics_data['INDEX ITEM DESCRIPTION'].fillna('').tolist()
        self.naics_embeddings = self.model.encode(descriptions, show_progress_bar=True)
        
        # Cache the embeddings
        with open(cache_file, 'wb') as f:
            pickle.dump(self.naics_embeddings, f)
        print(f"Cached NAICS embeddings to {cache_file}")
        
        return self.naics_embeddings
    
    def find_best_naics_matches(self, description, top_k=5):
        """Find best NAICS matches for a given description."""
        # Create embedding for the description
        desc_embedding = self.model.encode([description])
        
        # Calculate similarities
        similarities = cosine_similarity(desc_embedding, self.naics_embeddings)[0]
        
        # Get top matches
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        matches = []
        for idx in top_indices:
            matches.append({
                'naics_code': self.naics_data.iloc[idx]['NAICS22'],
                'description': self.naics_data.iloc[idx]['INDEX ITEM DESCRIPTION'],
                'similarity_score': similarities[idx]
            })
        
        return matches
    
    def map_sector_category_niche_combinations(self, combinations_file="sector_category_niche_combinations.csv"):
        """Map all Sector → Category → Niche combinations to NAICS codes."""
        print("Loading combinations...")
        combinations = pd.read_csv(combinations_file)
        
        print(f"Mapping {len(combinations)} combinations to NAICS codes...")
        
        results = []
        for idx, row in combinations.iterrows():
            if idx % 100 == 0:
                print(f"Processing {idx}/{len(combinations)}...")
            
            # Use the niche as primary description, with category as context
            description = f"{row['niche']} {row['category']}"
            
            # Find best matches
            matches = self.find_best_naics_matches(description, top_k=3)
            
            # Store results
            result = {
                'sector': row['sector'],
                'category': row['category'], 
                'niche': row['niche'],
                'best_naics_code': matches[0]['naics_code'],
                'best_naics_description': matches[0]['description'],
                'best_similarity_score': matches[0]['similarity_score'],
                'second_naics_code': matches[1]['naics_code'] if len(matches) > 1 else None,
                'second_similarity_score': matches[1]['similarity_score'] if len(matches) > 1 else None,
                'third_naics_code': matches[2]['naics_code'] if len(matches) > 2 else None,
                'third_similarity_score': matches[2]['similarity_score'] if len(matches) > 2 else None,
            }
            results.append(result)
        
        results_df = pd.DataFrame(results)
        return results_df
    
    def save_mappings(self, mappings_df, filename="naics_mappings.csv"):
        """Save the mappings to a CSV file."""
        filepath = os.path.join(self.cache_dir, filename)
        mappings_df.to_csv(filepath, index=False)
        print(f"Saved NAICS mappings to {filepath}")
        return filepath


def create_naics_mappings():
    """Main function to create NAICS mappings for all combinations."""
    mapper = NAICSMapper()
    
    # Load data and model
    mapper.load_naics_data()
    mapper.load_embedding_model()
    
    # Create embeddings
    mapper.create_naics_embeddings()
    
    # Create mappings
    mappings = mapper.map_sector_category_niche_combinations()
    
    # Save results
    mapper.save_mappings(mappings)
    
    # Display summary
    print("\n=== NAICS Mapping Summary ===")
    print(f"Total combinations mapped: {len(mappings)}")
    print(f"Average similarity score: {mappings['best_similarity_score'].mean():.3f}")
    print(f"High confidence mappings (>0.7): {(mappings['best_similarity_score'] > 0.7).sum()}")
    print(f"Medium confidence mappings (0.5-0.7): {((mappings['best_similarity_score'] >= 0.5) & (mappings['best_similarity_score'] <= 0.7)).sum()}")
    print(f"Low confidence mappings (<0.5): {(mappings['best_similarity_score'] < 0.5).sum()}")
    
    return mappings


if __name__ == "__main__":
    mappings = create_naics_mappings() 