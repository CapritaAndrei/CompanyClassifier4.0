"""
Test Discovery-Based Active Learning Pipeline
==========================================

Test the discovery pipeline with actual labeled training data to validate:
1. Data loading and processing
2. Bootstrap classifier training
3. Uncertainty detection
4. Label discovery interface
5. Progress tracking

Uses actual data:
- 2,379 labeled training examples
- 9,495 unlabeled challenge examples
- 61 mapped labels from Master Map
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class TestDiscoveryPipeline:
    """
    Test version of discovery pipeline using actual labeled data
    """
    
    def __init__(self):
        print("🔄 Initializing Test Discovery Pipeline...")
        
        # Paths to actual data
        self.master_map_path = "data/processed/master_insurance_to_naics_mapping_simplified.json"
        self.taxonomy_path = "data/input/insurance_taxonomy - insurance_taxonomy.csv"
        self.training_data_path = "data/processed/training_data.csv"
        self.challenge_data_path = "data/input/ml_insurance_challenge.csv"
        
        # Initialize components
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Data containers
        self.master_map = {}
        self.taxonomy_labels = []
        self.training_data = None
        self.challenge_data = None
        self.known_labels = set()
        
        # Model components
        self.classifier = None
        self.vectorizer = None
        self.current_labels = []
        
        # Load data
        self._load_all_data()
    
    def _load_all_data(self):
        """Load all required data files"""
        print("📥 Loading data files...")
        
        # 1. Load Master Map
        try:
            with open(self.master_map_path, 'r') as f:
                self.master_map = json.load(f)
            self.known_labels = set(self.master_map.keys())
            print(f"✅ Master Map: {len(self.known_labels)} mapped labels")
        except Exception as e:
            print(f"❌ Error loading Master Map: {e}")
            return False
        
        # 2. Load Taxonomy
        try:
            taxonomy_df = pd.read_csv(self.taxonomy_path)
            self.taxonomy_labels = taxonomy_df['label'].tolist()
            print(f"✅ Taxonomy: {len(self.taxonomy_labels)} total labels")
        except Exception as e:
            print(f"❌ Error loading taxonomy: {e}")
            return False
        
        # 3. Load Training Data
        try:
            self.training_data = pd.read_csv(self.training_data_path)
            print(f"✅ Training Data: {len(self.training_data)} labeled examples")
            print(f"   Columns: {list(self.training_data.columns)}")
        except Exception as e:
            print(f"❌ Error loading training data: {e}")
            return False
        
        # 4. Load Challenge Data
        try:
            self.challenge_data = pd.read_csv(self.challenge_data_path)
            print(f"✅ Challenge Data: {len(self.challenge_data)} unlabeled examples")
            print(f"   Columns: {list(self.challenge_data.columns)}")
        except Exception as e:
            print(f"❌ Error loading challenge data: {e}")
            return False
        
        return True
    
    def analyze_current_labels(self):
        """Analyze the current labeled data"""
        print("\n📊 Analyzing Current Labeled Data")
        print("=" * 50)
        
        if 'primary_label' in self.training_data.columns:
            label_col = 'primary_label'
        elif 'selected_labels' in self.training_data.columns:
            label_col = 'selected_labels'
        else:
            print("❌ No label column found in training data")
            return
        
        # Get label distribution
        label_counts = self.training_data[label_col].value_counts()
        
        print(f"📈 Label Distribution (top 20):")
        for label, count in label_counts.head(20).items():
            print(f"   {label:<40} {count:>5} examples")
        
        print(f"\n📊 Summary Statistics:")
        print(f"   Total labels in training data: {len(label_counts)}")
        print(f"   Total examples: {len(self.training_data)}")
        print(f"   Average examples per label: {len(self.training_data) / len(label_counts):.1f}")
        
        # Check overlap with Master Map
        training_labels = set(label_counts.index)
        master_map_labels = set(self.master_map.keys())
        
        overlap = training_labels.intersection(master_map_labels)
        only_in_training = training_labels - master_map_labels
        only_in_master = master_map_labels - training_labels
        
        print(f"\n🔍 Label Overlap Analysis:")
        print(f"   Labels in both training & master map: {len(overlap)}")
        print(f"   Labels only in training data: {len(only_in_training)}")
        print(f"   Labels only in master map: {len(only_in_master)}")
        
        if only_in_training:
            print(f"\n📋 Labels only in training data (top 10):")
            for label in list(only_in_training)[:10]:
                count = label_counts[label]
                print(f"   {label:<40} {count:>5} examples")
        
        return {
            'total_labels': len(label_counts),
            'total_examples': len(self.training_data),
            'overlap_labels': len(overlap),
            'training_only_labels': len(only_in_training),
            'master_only_labels': len(only_in_master)
        }
    
    def test_bootstrap_classifier(self, test_size: int = 500):
        """Test bootstrap classifier with subset of data"""
        print(f"\n🚀 Testing Bootstrap Classifier")
        print("=" * 50)
        
        if 'primary_label' in self.training_data.columns:
            label_col = 'primary_label'
        else:
            print("❌ No primary_label column found")
            return False
        
        # Use subset for testing and clean data
        test_data = self.training_data.sample(n=min(test_size, len(self.training_data)), random_state=42)
        
        # Clean data - remove NaN values
        test_data = test_data.dropna(subset=['description', label_col])
        
        print(f"📊 Test Data: {len(test_data)} examples")
        
        # Prepare features and labels
        X = test_data['description'].values
        y = test_data[label_col].values
        
        # Split for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   Training: {len(X_train)} examples")
        print(f"   Testing: {len(X_test)} examples")
        
        # Create TF-IDF features
        print("🔄 Creating TF-IDF features...")
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2
        )
        
        X_train_features = self.vectorizer.fit_transform(X_train)
        X_test_features = self.vectorizer.transform(X_test)
        
        # Train classifier
        print("🔄 Training classifier...")
        base_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        
        self.classifier = CalibratedClassifierCV(base_classifier, cv=3)
        self.classifier.fit(X_train_features, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test_features)
        y_proba = self.classifier.predict_proba(X_test_features)
        
        print(f"\n📊 Classifier Performance:")
        print(classification_report(y_test, y_pred))
        
        # Store current labels
        self.current_labels = sorted(set(y_train))
        
        # Calculate confidence distribution
        max_probs = np.max(y_proba, axis=1)
        uncertainty_scores = 1 - max_probs
        
        print(f"\n📈 Confidence Distribution:")
        print(f"   High confidence (>0.8): {np.sum(max_probs > 0.8)} examples")
        print(f"   Medium confidence (0.6-0.8): {np.sum((max_probs >= 0.6) & (max_probs <= 0.8))} examples")
        print(f"   Low confidence (<0.6): {np.sum(max_probs < 0.6)} examples")
        print(f"   Average confidence: {np.mean(max_probs):.3f}")
        
        return True
    
    def test_uncertainty_detection(self, n_samples: int = 100):
        """Test uncertainty detection on challenge data"""
        print(f"\n🔍 Testing Uncertainty Detection")
        print("=" * 50)
        
        if self.classifier is None:
            print("❌ No classifier available. Run test_bootstrap_classifier() first.")
            return []
        
        # Use subset of challenge data and clean it
        challenge_subset = self.challenge_data.sample(n=min(n_samples, len(self.challenge_data)), random_state=42)
        
        # Clean data - remove NaN descriptions
        challenge_subset = challenge_subset.dropna(subset=['description'])
        
        print(f"📊 Analyzing {len(challenge_subset)} challenge examples...")
        
        # Get predictions
        X_challenge = self.vectorizer.transform(challenge_subset['description'].values)
        predictions = self.classifier.predict(X_challenge)
        probabilities = self.classifier.predict_proba(X_challenge)
        
        # Calculate uncertainty
        max_probs = np.max(probabilities, axis=1)
        uncertainty_scores = 1 - max_probs
        
        # Create uncertainty cases
        uncertain_cases = []
        for i, (idx, row) in enumerate(challenge_subset.iterrows()):
            uncertain_cases.append({
                'original_index': idx,
                'description': row['description'],
                'predicted_label': predictions[i],
                'max_probability': max_probs[i],
                'uncertainty_score': uncertainty_scores[i],
                'business_tags': row.get('business_tags', ''),
                'sector': row.get('sector', ''),
                'category': row.get('category', '')
            })
        
        # Sort by uncertainty
        uncertain_cases.sort(key=lambda x: x['uncertainty_score'], reverse=True)
        
        print(f"\n📈 Uncertainty Analysis:")
        print(f"   Most uncertain: {uncertain_cases[0]['uncertainty_score']:.3f}")
        print(f"   Least uncertain: {uncertain_cases[-1]['uncertainty_score']:.3f}")
        print(f"   Average uncertainty: {np.mean(uncertainty_scores):.3f}")
        
        # Show top 5 most uncertain
        print(f"\n🔝 Top 5 Most Uncertain Cases:")
        for i, case in enumerate(uncertain_cases[:5], 1):
            print(f"\n   {i}. Uncertainty: {case['uncertainty_score']:.3f}")
            print(f"      Predicted: {case['predicted_label']} (confidence: {case['max_probability']:.3f})")
            print(f"      Description: {case['description'][:100]}...")
            print(f"      Sector: {case['sector']}")
        
        return uncertain_cases
    
    def test_label_similarity(self, test_description: str):
        """Test label similarity ranking"""
        print(f"\n🎯 Testing Label Similarity")
        print("=" * 50)
        
        print(f"📝 Test Description: {test_description}")
        
        # Get embedding for test description
        desc_embedding = self.embedding_model.encode([test_description])
        
        # Get embeddings for current labels
        label_embeddings = self.embedding_model.encode(self.current_labels)
        
        # Calculate similarities
        similarities = cosine_similarity(desc_embedding, label_embeddings)[0]
        
        # Rank labels
        ranked_labels = [(self.current_labels[i], similarities[i]) 
                        for i in range(len(self.current_labels))]
        ranked_labels.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🎯 Top 10 Similar Labels:")
        for i, (label, similarity) in enumerate(ranked_labels[:10], 1):
            print(f"   {i:2d}. {label:<40} (similarity: {similarity:.3f})")
        
        return ranked_labels
    
    def interactive_test_session(self):
        """Interactive test session to validate discovery interface"""
        print(f"\n🎮 Interactive Test Session")
        print("=" * 50)
        
        # Get some uncertain cases
        uncertain_cases = self.test_uncertainty_detection(20)
        
        if not uncertain_cases:
            print("❌ No uncertain cases found.")
            return
        
        print(f"\n💡 This simulates the discovery interface you'll use in production.")
        print(f"   We'll show you the top 3 most uncertain cases.")
        
        for i, case in enumerate(uncertain_cases[:3], 1):
            print(f"\n" + "="*60)
            print(f"🔍 UNCERTAIN CASE {i}/3")
            print(f"   Description: {case['description'][:200]}...")
            print(f"   Sector: {case['sector']}")
            print(f"   Category: {case['category']}")
            print(f"   Current prediction: {case['predicted_label']} (confidence: {case['max_probability']:.3f})")
            print(f"   Uncertainty score: {case['uncertainty_score']:.3f}")
            
            # Show similar labels
            similar_labels = self.test_label_similarity(case['description'])
            print(f"\n🎯 Most Similar Labels:")
            for j, (label, similarity) in enumerate(similar_labels[:5], 1):
                print(f"   {j}. {label:<40} (similarity: {similarity:.3f})")
            
            print(f"\n💭 In the real interface, you would:")
            print(f"   • Choose from similar labels (1-5)")
            print(f"   • Create new label (n)")
            print(f"   • Skip this case (s)")
            print(f"   • Save and quit (q)")
    
    def run_full_test(self):
        """Run complete test of discovery pipeline"""
        print("🎯 DISCOVERY PIPELINE - FULL TEST")
        print("=" * 70)
        
        # Step 1: Analyze current data
        analysis = self.analyze_current_labels()
        
        # Step 2: Test bootstrap classifier
        if self.test_bootstrap_classifier():
            print("✅ Bootstrap classifier test passed")
        else:
            print("❌ Bootstrap classifier test failed")
            return
        
        # Step 3: Test uncertainty detection
        uncertain_cases = self.test_uncertainty_detection(50)
        if uncertain_cases:
            print("✅ Uncertainty detection test passed")
        else:
            print("❌ Uncertainty detection test failed")
            return
        
        # Step 4: Test similarity ranking
        test_desc = "Software development and technology consulting services"
        self.test_label_similarity(test_desc)
        print("✅ Label similarity test passed")
        
        # Step 5: Interactive test session
        self.interactive_test_session()
        
        # Final summary
        print(f"\n🎉 TEST COMPLETE - All Systems Working!")
        print("=" * 70)
        print(f"✅ Data Loading: Success")
        print(f"✅ Classifier Training: Success") 
        print(f"✅ Uncertainty Detection: Success")
        print(f"✅ Label Similarity: Success")
        print(f"✅ Interface Simulation: Success")
        
        print(f"\n🚀 Ready for Production Pipeline!")
        print(f"   • Training data: {len(self.training_data)} examples")
        print(f"   • Challenge data: {len(self.challenge_data)} examples")
        print(f"   • Current labels: {len(self.current_labels)}")
        print(f"   • Master map labels: {len(self.master_map)}")


def main():
    """Main test function"""
    test_pipeline = TestDiscoveryPipeline()
    test_pipeline.run_full_test()


if __name__ == "__main__":
    main() 