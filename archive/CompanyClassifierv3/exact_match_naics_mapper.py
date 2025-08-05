"""
Exact Match NAICS Mapper
Creates NAICS → Insurance Label mappings using exact substring matching
with preprocessing (lowercase + stemming)
"""

import pandas as pd
import numpy as np
import json
import re
import ast
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import warnings
warnings.filterwarnings('ignore')

class SimpleStemmer:
    """Simple stemming implementation without NLTK dependency"""
    
    def __init__(self):
        # Common English stop words
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'or', 'but', 'if', 'this', 'they',
            'we', 'you', 'your', 'their', 'them', 'than', 'can', 'could'
        }
        
        # Basic stemming rules (suffix removal)
        self.suffix_rules = [
            ('ingness', 'ing'),
            ('ational', 'ate'),
            ('tional', 'tion'),
            ('encies', 'ency'),
            ('ically', 'ic'),
            ('ations', 'ate'),
            ('fulness', 'ful'),
            ('ousness', 'ous'),
            ('iveness', 'ive'),
            ('ement', 'e'),
            ('ance', ''),
            ('ence', ''),
            ('ing', ''),
            ('tion', ''),
            ('ness', ''),
            ('ment', ''),
            ('able', ''),
            ('ible', ''),
            ('ive', ''),
            ('ous', ''),
            ('ful', ''),
            ('less', ''),
            ('ed', ''),
            ('er', ''),
            ('est', ''),
            ('ly', ''),
            ('s', ''),
        ]
    
    def stem(self, word):
        """Simple stemming function"""
        word = word.lower().strip()
        
        if len(word) <= 3:
            return word
            
        # Apply suffix rules
        for suffix, replacement in self.suffix_rules:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)] + replacement
        
        return word


class ExactMatchNAICSMapper:
    """
    Creates NAICS → Insurance Label mappings using exact substring matching
    """
    
    def __init__(self):
        self.stemmer = SimpleStemmer()
        self.stop_words = self.stemmer.stop_words
        
        # Load data
        self.insurance_labels = self._load_insurance_taxonomy()
        self.naics_data = self._load_naics_data()
        
        # Store mappings
        self.naics_to_insurance = defaultdict(list)
        self.insurance_to_naics = defaultdict(list)
        self.match_confidence = {}
        
    def _load_insurance_taxonomy(self) -> List[str]:
        """Load the 220 insurance taxonomy labels"""
        try:
            df = pd.read_csv('data/input/insurance_taxonomy - insurance_taxonomy.csv')
            labels = df['label'].tolist()
            print(f"✅ Loaded {len(labels)} insurance taxonomy labels")
            return labels
        except Exception as e:
            print(f"❌ Error loading insurance taxonomy: {e}")
            return []
    
    def _load_naics_data(self) -> Dict[str, Dict]:
        """Load NAICS titles and descriptions"""
        naics_data = {}
        
        try:
            # Load 2022 NAICS codes (official titles)
            df_codes = pd.read_excel('data/input/6-digit_2022_Codes.xlsx')
            
            # Load 2022 NAICS index (activity descriptions)  
            df_index = pd.read_excel('data/input/2022_NAICS_Index_File.xlsx')
            
            print(f"✅ Loaded {len(df_codes)} NAICS codes and {len(df_index)} activity descriptions")
            
            # Process official titles
            for _, row in df_codes.iterrows():
                naics_code = str(row.iloc[0]).strip()  # First column is NAICS code
                title = str(row.iloc[1]).strip() if len(row) > 1 else ""  # Second column is title
                
                if naics_code and title and naics_code != 'nan':
                    naics_data[naics_code] = {
                        'title': title,
                        'descriptions': []
                    }
            
            # Process activity descriptions
            for _, row in df_index.iterrows():
                if len(row) >= 2:
                    description = str(row.iloc[0]).strip()  # First column is description
                    naics_code = str(row.iloc[1]).strip()   # Second column is NAICS code
                    
                    if naics_code in naics_data and description and description != 'nan':
                        naics_data[naics_code]['descriptions'].append(description)
            
            print(f"✅ Processed NAICS data for {len(naics_data)} codes")
            return naics_data
            
        except Exception as e:
            print(f"❌ Error loading NAICS data: {e}")
            return {}
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text: lowercase + stemming
        """
        if not text or pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove punctuation and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize, remove stop words, and stem
        words = text.split()
        processed_words = []
        
        for word in words:
            if word and len(word) > 2:  # Skip very short words
                stemmed = self.stemmer.stem(word)
                processed_words.append(stemmed)
        
        return ' '.join(processed_words)
    
    def find_exact_matches(self, insurance_label: str) -> List[Tuple[str, str, str]]:
        """
        Find exact substring matches for an insurance label in NAICS data
        
        Returns:
            List of (naics_code, match_text, match_type) tuples
        """
        processed_label = self.preprocess_text(insurance_label)
        matches = []
        
        if not processed_label:
            return matches
        
        for naics_code, data in self.naics_data.items():
            # Check title
            if data['title']:
                processed_title = self.preprocess_text(data['title'])
                if processed_label in processed_title:
                    matches.append((naics_code, data['title'], 'title'))
            
            # Check descriptions
            for description in data['descriptions']:
                processed_desc = self.preprocess_text(description)
                if processed_label in processed_desc:
                    matches.append((naics_code, description, 'description'))
        
        return matches
    
    def create_mappings(self) -> Dict:
        """
        Create NAICS → Insurance Label mappings for all labels
        """
        print(f"\n🔍 Finding exact matches for {len(self.insurance_labels)} insurance labels...")
        
        mapping_results = {
            'successful_mappings': {},
            'failed_mappings': [],
            'statistics': {}
        }
        
        successful_count = 0
        total_matches = 0
        
        for i, label in enumerate(self.insurance_labels):
            matches = self.find_exact_matches(label)
            
            if matches:
                successful_count += 1
                total_matches += len(matches)
                
                # Store all matches for this label
                mapping_results['successful_mappings'][label] = []
                
                for naics_code, match_text, match_type in matches:
                    mapping_results['successful_mappings'][label].append({
                        'naics_code': naics_code,
                        'match_text': match_text,
                        'match_type': match_type
                    })
                    
                    # Create reverse mapping
                    self.naics_to_insurance[naics_code].append(label)
                    self.insurance_to_naics[label].append(naics_code)
                
                # Calculate confidence based on number of matches
                confidence = min(1.0, len(matches) / 3)  # Max confidence at 3+ matches
                self.match_confidence[label] = confidence
                
                if i % 20 == 0 or len(matches) > 5:
                    print(f"   {label}: {len(matches)} matches (confidence: {confidence:.2f})")
            else:
                mapping_results['failed_mappings'].append(label)
        
        # Statistics
        mapping_results['statistics'] = {
            'total_labels': len(self.insurance_labels),
            'successful_mappings': successful_count,
            'failed_mappings': len(mapping_results['failed_mappings']),
            'success_rate': successful_count / len(self.insurance_labels),
            'total_matches': total_matches,
            'average_matches_per_label': total_matches / successful_count if successful_count > 0 else 0
        }
        
        return mapping_results
    
    def apply_mappings_to_companies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply NAICS → Insurance mappings to company dataset
        """
        print(f"\n🏷️ Applying mappings to {len(df)} companies...")
        
        # We'll use BEACON to predict NAICS codes first
        from BEACON.beacon import BeaconModel, load_naics_data
        
        # Load and train BEACON
        print("   Loading BEACON model...")
        X_2017, y_2017, sw_2017 = load_naics_data('2017')
        X_2022, y_2022, sw_2022 = load_naics_data('2022')
        
        X_beacon = np.concatenate([X_2017, X_2022])
        y_beacon = np.concatenate([y_2017, y_2022])
        sw_beacon = np.concatenate([sw_2017, sw_2022])
        
        beacon = BeaconModel(verbose=0)
        beacon.fit(X_beacon, y_beacon, sw_beacon)
        
        # Predict NAICS codes for companies
        print("   Predicting NAICS codes...")
        descriptions = df['description'].fillna('').astype(str)
        predicted_naics = beacon.predict(descriptions)
        
        # Apply insurance label mappings
        print("   Mapping to insurance labels...")
        company_labels = []
        mapping_confidence = []
        
        for naics_code in predicted_naics:
            if naics_code in self.naics_to_insurance:
                labels = self.naics_to_insurance[naics_code]
                # Get confidence scores for these labels
                confidences = [self.match_confidence.get(label, 0) for label in labels]
                company_labels.append(labels)
                mapping_confidence.append(max(confidences) if confidences else 0)
            else:
                company_labels.append([])
                mapping_confidence.append(0)
        
        # Add to dataframe
        df_result = df.copy()
        df_result['predicted_naics'] = predicted_naics
        df_result['insurance_labels'] = company_labels
        df_result['mapping_confidence'] = mapping_confidence
        
        # Filter for high-confidence mappings only
        high_confidence = df_result[df_result['mapping_confidence'] >= 0.7]
        
        print(f"   📊 Results:")
        print(f"      Total companies: {len(df_result)}")
        print(f"      With insurance labels: {len(df_result[df_result['insurance_labels'].apply(len) > 0])}")
        print(f"      High confidence (≥0.7): {len(high_confidence)}")
        
        return df_result, high_confidence
    
    def save_results(self, mapping_results: Dict, output_file: str = "exact_match_mappings.json"):
        """Save mapping results to file"""
        with open(output_file, 'w') as f:
            json.dump(mapping_results, f, indent=2)
        print(f"💾 Saved mapping results to {output_file}")

def main():
    """Main execution pipeline"""
    print("🎯 NAICS → Insurance Label Exact Match Mapper")
    print("=" * 60)
    
    # Initialize mapper
    mapper = ExactMatchNAICSMapper()
    
    if not mapper.insurance_labels or not mapper.naics_data:
        print("❌ Failed to load required data")
        return
    
    # Create mappings
    mapping_results = mapper.create_mappings()
    
    # Print summary
    stats = mapping_results['statistics']
    print(f"\n📊 MAPPING SUMMARY:")
    print(f"   Total insurance labels: {stats['total_labels']}")
    print(f"   Successfully mapped: {stats['successful_mappings']}")
    print(f"   Failed to map: {stats['failed_mappings']}")
    print(f"   Success rate: {stats['success_rate']:.1%}")
    print(f"   Total matches found: {stats['total_matches']}")
    print(f"   Average matches per label: {stats['average_matches_per_label']:.1f}")
    
    # Show some examples
    print(f"\n🔍 EXAMPLE MAPPINGS:")
    for label, matches in list(mapping_results['successful_mappings'].items())[:5]:
        print(f"   {label}:")
        for match in matches[:2]:  # Show first 2 matches
            print(f"      → {match['naics_code']}: {match['match_text'][:60]}...")
    
    # Save results
    mapper.save_results(mapping_results)
    
    # Apply to companies
    try:
        print(f"\n📁 Loading company dataset...")
        df = pd.read_csv('data/input/ml_insurance_challenge.csv')
        
        df_labeled, df_high_conf = mapper.apply_mappings_to_companies(df)
        
        # Save training data
        df_high_conf.to_csv('data/processed/high_confidence_training_data.csv', index=False)
        print(f"💾 Saved high-confidence training data: {len(df_high_conf)} companies")
        
        return mapper, mapping_results, df_high_conf
        
    except Exception as e:
        print(f"⚠️ Could not apply to company dataset: {e}")
        return mapper, mapping_results, None

if __name__ == "__main__":
    main() 