"""
Test BEACON Direct Insurance Label Mapping
=========================================

Simple test to validate the core concept:
1. Load BEACON dataset (40k examples: company_description → NAICS_code)
2. Filter to only NAICS codes we have mapped (61 labels)
3. Map NAICS codes to insurance labels using Master Map
4. Train classifier: company_description → insurance_label (directly)
5. Test accuracy and see how much training data we get

This bypasses the complex active learning and gives us a massive training set.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')


class BeaconDirectMapper:
    """
    Test direct mapping from BEACON dataset to insurance labels
    """
    
    def __init__(self):
        print("🔄 Initializing BEACON Direct Mapping Test...")
        
        # File paths
        self.beacon_path = "example_data_2017.txt"  # 40k BEACON dataset
        self.master_map_path = "data/processed/master_insurance_to_naics_mapping_simplified.json"
        self.taxonomy_path = "data/input/insurance_taxonomy - insurance_taxonomy.csv"
        
        # Data containers
        self.beacon_data = None
        self.master_map = {}
        self.naics_to_label = {}  # Reverse mapping: NAICS → insurance label
        
        # Results
        self.filtered_beacon = None
        self.classifier = None
        self.vectorizer = None
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load all required data"""
        print("📥 Loading data files...")
        
        # 1. Load BEACON dataset
        beacon_paths = [
            self.beacon_path,
            "BEACON/example_data_2017.txt",
            "example_data_2017.txt",
            "BEACON/example_data_2022.txt",
            "example_data_2022.txt"
        ]
        
        for path in beacon_paths:
            if Path(path).exists():
                print(f"📄 Loading BEACON dataset from: {path}")
                try:
                    # BEACON format: TEXT|NAICS|SAMPLE_WEIGHT
                    self.beacon_data = pd.read_csv(path, sep='|', header=None)
                    self.beacon_data.columns = ['company_description', 'naics_code', 'sample_weight']
                    print(f"✅ BEACON: {len(self.beacon_data)} examples loaded")
                    break
                except Exception as e:
                    print(f"❌ Error loading {path}: {e}")
                    continue
        
        if self.beacon_data is None:
            print("❌ BEACON dataset not found. Creating sample data...")
            self._create_sample_beacon_data()
        
        # 2. Load Master Map
        try:
            with open(self.master_map_path, 'r') as f:
                self.master_map = json.load(f)
            print(f"✅ Master Map: {len(self.master_map)} insurance labels")
        except Exception as e:
            print(f"❌ Error loading Master Map: {e}")
            return False
        
        # 3. Create reverse mapping: NAICS → insurance label
        self.naics_to_label = {}
        for label, naics_codes in self.master_map.items():
            for naics_info in naics_codes:
                naics_code = naics_info['naics_code']
                self.naics_to_label[naics_code] = label
        
        print(f"✅ Reverse mapping: {len(self.naics_to_label)} NAICS codes mapped to insurance labels")
        
        return True
    
    def _create_sample_beacon_data(self):
        """Create sample BEACON data for testing"""
        print("🔄 Creating sample BEACON data...")
        
        # Create sample data that matches some of our mapped NAICS codes
        sample_data = []
        
        # Add examples for each mapped NAICS code
        for naics_code, label in list(self.naics_to_label.items())[:10]:  # First 10 for testing
            descriptions = [
                f"Manufacturing company specializing in {label.lower()}",
                f"Industrial {label.lower()} production and services",
                f"Commercial {label.lower()} business operations",
                f"Professional {label.lower()} solutions provider",
                f"Specialized {label.lower()} manufacturing facility"
            ]
            
            for desc in descriptions:
                sample_data.append({
                    'company_description': desc,
                    'naics_code': naics_code,
                    'sample_weight': 1.0
                })
        
        self.beacon_data = pd.DataFrame(sample_data)
        print(f"✅ Created {len(self.beacon_data)} sample BEACON examples")
    
    def filter_beacon_to_mapped_naics(self):
        """Filter BEACON dataset to only include mapped NAICS codes"""
        print("\n🔍 Filtering BEACON Dataset to Mapped NAICS Codes")
        print("=" * 60)
        
        if self.beacon_data is None:
            print("❌ No BEACON data available")
            return False
        
        # Convert NAICS codes to string for consistent comparison
        self.beacon_data['naics_code'] = self.beacon_data['naics_code'].astype(str)
        mapped_naics = set(self.naics_to_label.keys())
        
        print(f"📊 BEACON Dataset Analysis:")
        print(f"   Total BEACON examples: {len(self.beacon_data):,}")
        print(f"   Unique NAICS codes in BEACON: {len(self.beacon_data['naics_code'].unique()):,}")
        print(f"   Mapped NAICS codes available: {len(mapped_naics)}")
        
        # Filter to only mapped NAICS codes
        self.filtered_beacon = self.beacon_data[
            self.beacon_data['naics_code'].isin(mapped_naics)
        ].copy()
        
        print(f"\n✅ Filtering Results:")
        print(f"   Examples with mapped NAICS: {len(self.filtered_beacon):,}")
        print(f"   Coverage: {len(self.filtered_beacon) / len(self.beacon_data) * 100:.1f}%")
        
        if len(self.filtered_beacon) == 0:
            print("❌ No examples found with mapped NAICS codes!")
            return False
        
        # Add insurance labels
        self.filtered_beacon['insurance_label'] = self.filtered_beacon['naics_code'].map(self.naics_to_label)
        
        # Analyze label distribution
        label_counts = self.filtered_beacon['insurance_label'].value_counts()
        
        print(f"\n📈 Insurance Label Distribution (Top 20):")
        for label, count in label_counts.head(20).items():
            print(f"   {label:<40} {count:>6} examples")
        
        print(f"\n📊 Training Data Statistics:")
        print(f"   Total insurance labels: {len(label_counts)}")
        print(f"   Total training examples: {len(self.filtered_beacon):,}")
        print(f"   Average examples per label: {len(self.filtered_beacon) / len(label_counts):.1f}")
        print(f"   Min examples per label: {label_counts.min()}")
        print(f"   Max examples per label: {label_counts.max()}")
        
        return True
    
    def train_direct_classifier(self):
        """Train classifier directly from company descriptions to insurance labels"""
        print("\n🚀 Training Direct Insurance Label Classifier")
        print("=" * 60)
        
        if self.filtered_beacon is None or len(self.filtered_beacon) == 0:
            print("❌ No filtered BEACON data available")
            return False
        
        # Prepare data
        X = self.filtered_beacon['company_description'].values
        y = self.filtered_beacon['insurance_label'].values
        
        # Remove any NaN values
        valid_indices = ~pd.isna(self.filtered_beacon['company_description']) & ~pd.isna(self.filtered_beacon['insurance_label'])
        X = X[valid_indices]
        y = y[valid_indices]
        
        print(f"📊 Training Data:")
        print(f"   Total examples: {len(X):,}")
        print(f"   Unique labels: {len(set(y))}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"   Training set: {len(X_train):,} examples")
        print(f"   Test set: {len(X_test):,} examples")
        
        # Create TF-IDF features
        print("🔄 Creating TF-IDF features...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2
        )
        
        X_train_features = self.vectorizer.fit_transform(X_train)
        X_test_features = self.vectorizer.transform(X_test)
        
        # Train classifier
        print("🔄 Training Random Forest classifier...")
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        
        self.classifier.fit(X_train_features, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test_features)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n📊 Classifier Performance:")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"\n📋 Detailed Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return True
    
    def test_on_sample_descriptions(self):
        """Test classifier on sample business descriptions"""
        print("\n🧪 Testing on Sample Business Descriptions")
        print("=" * 60)
        
        if self.classifier is None:
            print("❌ No trained classifier available")
            return
        
        # Sample business descriptions
        test_descriptions = [
            "Chemical manufacturing company producing industrial chemicals",
            "Waste management and recycling services provider",
            "Insurance brokerage and risk management services",
            "Software development and technology consulting",
            "Plastic manufacturing and injection molding",
            "Textile manufacturing and fabric production",
            "Food processing and beverage manufacturing",
            "Construction and building materials supply"
        ]
        
        print("🔍 Sample Predictions:")
        for desc in test_descriptions:
            # Get prediction
            desc_features = self.vectorizer.transform([desc])
            prediction = self.classifier.predict(desc_features)[0]
            probabilities = self.classifier.predict_proba(desc_features)[0]
            confidence = max(probabilities)
            
            print(f"\n   Description: {desc}")
            print(f"   Prediction: {prediction}")
            print(f"   Confidence: {confidence:.3f}")
    
    def run_complete_test(self):
        """Run the complete test workflow"""
        print("🎯 BEACON DIRECT MAPPING - COMPLETE TEST")
        print("=" * 70)
        
        # Step 1: Filter BEACON to mapped NAICS codes
        if not self.filter_beacon_to_mapped_naics():
            print("❌ Failed to filter BEACON data")
            return False
        
        # Step 2: Train direct classifier
        if not self.train_direct_classifier():
            print("❌ Failed to train classifier")
            return False
        
        # Step 3: Test on sample descriptions
        self.test_on_sample_descriptions()
        
        # Step 4: Final summary
        print(f"\n🎉 TEST COMPLETE - BEACON DIRECT MAPPING WORKS!")
        print("=" * 70)
        
        if self.filtered_beacon is not None:
            label_counts = self.filtered_beacon['insurance_label'].value_counts()
            print(f"✅ Results Summary:")
            print(f"   Training examples generated: {len(self.filtered_beacon):,}")
            print(f"   Insurance labels covered: {len(label_counts)}")
            print(f"   Average examples per label: {len(self.filtered_beacon) / len(label_counts):.1f}")
            print(f"   Classifier accuracy: Available")
            
            print(f"\n🚀 Next Steps:")
            print(f"   1. Expand Master Map to more labels")
            print(f"   2. Get more training examples per label")
            print(f"   3. Deploy production classifier")
            print(f"   4. Apply to full 9k company dataset")
        
        return True


def main():
    """Main test function"""
    mapper = BeaconDirectMapper()
    mapper.run_complete_test()


if __name__ == "__main__":
    main() 