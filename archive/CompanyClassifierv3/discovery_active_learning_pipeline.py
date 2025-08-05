"""
Discovery-Based Active Learning Pipeline
=======================================

Intelligent active learning system that:
1. Starts with 61 known insurance labels mapped to NAICS codes
2. Trains initial classifier on companies that map to known labels
3. Applies to full 9k dataset, identifies uncertainty/low confidence cases
4. Presents unclear cases to user for new label discovery
5. Iteratively expands label coverage and retrains model
6. Tracks progress and quality metrics

Strategy:
- Bootstrap with known labels (61/220)
- Discover new labels based on real data needs
- Prioritize by frequency and business importance
- Incremental expansion rather than upfront mapping
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime
import pickle
from collections import Counter, defaultdict
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

# Import existing tools
import sys
sys.path.append('.')
from deepseek_naics_classifier import DeepSeekNAICSClassifier


class DiscoveryActiveLearningPipeline:
    """
    Discovery-based active learning system for insurance label classification
    """
    
    def __init__(self, 
                 master_map_path: str = "data/processed/master_insurance_to_naics_mapping_simplified.json",
                 taxonomy_path: str = "data/input/insurance_taxonomy - insurance_taxonomy.csv",
                 companies_data_path: str = "data/input/nine_k_companies_with_naics.csv",
                 progress_file: str = "data/processed/active_learning_progress.json"):
        
        self.master_map_path = master_map_path
        self.taxonomy_path = taxonomy_path
        self.companies_data_path = companies_data_path
        self.progress_file = progress_file
        
        # Initialize components
        print("🔄 Initializing Discovery-Based Active Learning Pipeline...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.naics_classifier = DeepSeekNAICSClassifier()
        
        # Data containers
        self.master_map = {}
        self.companies_data = None
        self.taxonomy_labels = []
        self.known_labels = set()
        self.discovered_labels = {}  # New labels discovered during process
        self.label_examples = defaultdict(list)  # Training examples per label
        
        # Model components
        self.classifier = None
        self.vectorizer = None
        self.label_encoder = None
        self.confidence_threshold = 0.6  # Below this = uncertain
        
        # Progress tracking
        self.progress = self._load_progress()
        
        # Load initial data
        self._load_master_map()
        self._load_taxonomy()
        self._load_companies_data()
    
    def _load_progress(self) -> Dict:
        """Load existing progress"""
        if Path(self.progress_file).exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            "discovered_labels": {},
            "manual_mappings": {},
            "iteration_history": [],
            "current_iteration": 0,
            "total_labels": 61,  # Starting with known labels
            "training_examples": 0,
            "uncertain_examples_reviewed": 0,
            "model_performance": {}
        }
    
    def _save_progress(self):
        """Save current progress"""
        Path(self.progress_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
        print(f"💾 Progress saved to {self.progress_file}")
    
    def _load_master_map(self):
        """Load Master Map with 61 known labels"""
        print("📥 Loading Master Map...")
        with open(self.master_map_path, 'r') as f:
            self.master_map = json.load(f)
        
        self.known_labels = set(self.master_map.keys())
        print(f"✅ Loaded {len(self.known_labels)} known insurance labels")
    
    def _load_taxonomy(self):
        """Load full insurance taxonomy (220 labels)"""
        print("📥 Loading insurance taxonomy...")
        taxonomy_df = pd.read_csv(self.taxonomy_path)
        self.taxonomy_labels = taxonomy_df['label'].tolist()
        print(f"✅ Loaded {len(self.taxonomy_labels)} total insurance labels")
    
    def _load_companies_data(self):
        """Load 9k companies dataset"""
        print("📥 Loading companies dataset...")
        
        # Try different possible paths
        possible_paths = [
            self.companies_data_path,
            "data/input/nine_k_companies_with_naics.csv",
            "data/input/companies_with_naics.csv",
            "data/processed/companies_with_naics.csv",
            "nine_k_companies_with_naics.csv"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                self.companies_data = pd.read_csv(path)
                print(f"✅ Loaded {len(self.companies_data)} companies from {path}")
                break
        
        if self.companies_data is None:
            print("❌ Companies dataset not found. Will create sample data.")
            self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample data for testing"""
        print("🔄 Creating sample companies data...")
        
        # Create sample companies with various business descriptions
        sample_companies = [
            "Manufacturing steel products and metal fabrication",
            "Waste management and recycling services",
            "Chemical manufacturing and industrial chemicals",
            "Insurance brokerage and risk management",
            "Software development and technology consulting",
            "Construction and building materials",
            "Healthcare services and medical equipment",
            "Food processing and beverage manufacturing",
            "Transportation and logistics services",
            "Retail sales and customer service"
        ] * 100  # 1000 sample companies
        
        self.companies_data = pd.DataFrame({
            'company_description': sample_companies,
            'company_id': range(len(sample_companies))
        })
        
        print(f"✅ Created {len(self.companies_data)} sample companies")
    
    def bootstrap_initial_classifier(self):
        """Train initial classifier on companies that map to known labels"""
        print("\n🚀 Phase 1: Bootstrap Initial Classifier")
        print("=" * 50)
        
        # Step 1: Apply BEACON to get NAICS predictions for all companies
        print("🔄 Step 1: Getting NAICS predictions from BEACON...")
        
        # Get NAICS predictions for all companies
        companies_with_naics = []
        batch_size = 100
        
        for i in range(0, len(self.companies_data), batch_size):
            batch = self.companies_data.iloc[i:i+batch_size]
            print(f"   Processing batch {i//batch_size + 1}/{(len(self.companies_data)-1)//batch_size + 1}")
            
            for _, row in batch.iterrows():
                try:
                    # Get NAICS prediction
                    naics_prediction = self.naics_classifier.classify_business(
                        row['company_description']
                    )
                    
                    companies_with_naics.append({
                        'company_id': row.get('company_id', f"company_{i}"),
                        'company_description': row['company_description'],
                        'predicted_naics': naics_prediction.get('naics_code', 'unknown'),
                        'naics_confidence': naics_prediction.get('confidence', 0.0)
                    })
                    
                except Exception as e:
                    print(f"   ⚠️ Error processing company: {e}")
                    companies_with_naics.append({
                        'company_id': row.get('company_id', f"company_{i}"),
                        'company_description': row['company_description'],
                        'predicted_naics': 'unknown',
                        'naics_confidence': 0.0
                    })
        
        # Create DataFrame with predictions
        self.companies_with_naics = pd.DataFrame(companies_with_naics)
        
        # Step 2: Map NAICS to known insurance labels
        print("🔄 Step 2: Mapping NAICS to known insurance labels...")
        
        # Create reverse mapping: NAICS -> Insurance Label
        naics_to_label = {}
        for label, naics_codes in self.master_map.items():
            for naics_info in naics_codes:
                naics_code = naics_info['naics_code']
                naics_to_label[naics_code] = label
        
        # Map companies to insurance labels
        training_data = []
        unmapped_companies = []
        
        for _, row in self.companies_with_naics.iterrows():
            predicted_naics = row['predicted_naics']
            
            if predicted_naics in naics_to_label:
                # Company maps to known label
                insurance_label = naics_to_label[predicted_naics]
                training_data.append({
                    'company_id': row['company_id'],
                    'company_description': row['company_description'],
                    'insurance_label': insurance_label,
                    'naics_code': predicted_naics,
                    'naics_confidence': row['naics_confidence']
                })
            else:
                # Company doesn't map to known label
                unmapped_companies.append(row)
        
        # Create training dataset
        self.training_data = pd.DataFrame(training_data)
        self.unmapped_companies = pd.DataFrame(unmapped_companies)
        
        print(f"✅ Mapping Results:")
        print(f"   Companies with known labels: {len(self.training_data)}")
        print(f"   Companies without known labels: {len(self.unmapped_companies)}")
        
        # Step 3: Train initial classifier
        if len(self.training_data) > 0:
            print("🔄 Step 3: Training initial classifier...")
            self._train_classifier(self.training_data)
        else:
            print("❌ No training data available. Cannot train classifier.")
            return False
        
        return True
    
    def _train_classifier(self, training_data: pd.DataFrame):
        """Train classifier on available training data"""
        
        # Prepare features and labels
        X = training_data['company_description'].values
        y = training_data['insurance_label'].values
        
        # Create TF-IDF features
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=2
            )
            X_features = self.vectorizer.fit_transform(X)
        else:
            X_features = self.vectorizer.transform(X)
        
        # Train classifier with probability calibration
        base_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        
        # Use calibrated classifier for better probability estimates
        self.classifier = CalibratedClassifierCV(base_classifier, cv=3)
        self.classifier.fit(X_features, y)
        
        # Store unique labels
        self.current_labels = sorted(set(y))
        
        print(f"✅ Classifier trained on {len(training_data)} examples")
        print(f"   Number of labels: {len(self.current_labels)}")
        
        # Evaluate if we have enough data
        if len(training_data) > 100:
            X_train, X_test, y_train, y_test = train_test_split(
                X_features, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train on split data
            self.classifier.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.classifier.predict(X_test)
            print("\n📊 Initial Classifier Performance:")
            print(classification_report(y_test, y_pred))
    
    def identify_uncertain_cases(self, n_cases: int = 20) -> List[Dict]:
        """Identify companies with uncertain predictions for label discovery"""
        print(f"\n🔍 Phase 2: Identifying {n_cases} Most Uncertain Cases")
        print("=" * 50)
        
        if self.classifier is None:
            print("❌ No classifier trained yet. Run bootstrap_initial_classifier() first.")
            return []
        
        # Get predictions for unmapped companies
        uncertain_cases = []
        
        if len(self.unmapped_companies) > 0:
            print(f"🔄 Analyzing {len(self.unmapped_companies)} unmapped companies...")
            
            # Get predictions and confidence scores
            X_unmapped = self.vectorizer.transform(self.unmapped_companies['company_description'].values)
            
            # Get prediction probabilities
            pred_probs = self.classifier.predict_proba(X_unmapped)
            predicted_labels = self.classifier.predict(X_unmapped)
            
            # Calculate uncertainty (1 - max probability)
            max_probs = np.max(pred_probs, axis=1)
            uncertainty_scores = 1 - max_probs
            
            # Create uncertainty cases
            for i, (_, row) in enumerate(self.unmapped_companies.iterrows()):
                uncertain_cases.append({
                    'company_id': row['company_id'],
                    'company_description': row['company_description'],
                    'predicted_label': predicted_labels[i],
                    'max_probability': max_probs[i],
                    'uncertainty_score': uncertainty_scores[i],
                    'naics_code': row['predicted_naics'],
                    'naics_confidence': row['naics_confidence']
                })
        
        # Sort by uncertainty (highest first)
        uncertain_cases.sort(key=lambda x: x['uncertainty_score'], reverse=True)
        
        # Return top N cases
        top_cases = uncertain_cases[:n_cases]
        
        print(f"✅ Found {len(top_cases)} uncertain cases for review")
        if len(top_cases) > 0:
            print(f"   Highest uncertainty: {top_cases[0]['uncertainty_score']:.3f}")
            print(f"   Lowest uncertainty: {top_cases[-1]['uncertainty_score']:.3f}")
        
        return top_cases
    
    def interactive_label_discovery(self, uncertain_cases: List[Dict]):
        """Interactive session for discovering new labels"""
        print(f"\n🎯 Phase 3: Interactive Label Discovery")
        print("=" * 50)
        
        if not uncertain_cases:
            print("❌ No uncertain cases to review.")
            return
        
        print(f"📊 Review Session Overview:")
        print(f"   Cases to review: {len(uncertain_cases)}")
        print(f"   Current known labels: {len(self.current_labels)}")
        print(f"   Available taxonomy labels: {len(self.taxonomy_labels)}")
        
        print(f"\n💡 Instructions:")
        print(f"   • Review each uncertain company")
        print(f"   • Choose: existing label, new label, or skip")
        print(f"   • 'q' to quit and save progress")
        
        new_mappings = []
        
        for i, case in enumerate(uncertain_cases, 1):
            print(f"\n" + "="*60)
            print(f"🔍 UNCERTAIN CASE {i}/{len(uncertain_cases)}")
            print(f"   Company: {case['company_description']}")
            print(f"   Predicted NAICS: {case['naics_code']}")
            print(f"   Current best guess: {case['predicted_label']} (confidence: {case['max_probability']:.3f})")
            print(f"   Uncertainty score: {case['uncertainty_score']:.3f}")
            print("-" * 60)
            
            # Show top similar existing labels
            similar_labels = self._get_similar_labels(case['company_description'])
            print(f"🎯 Most Similar Existing Labels:")
            for j, (label, similarity) in enumerate(similar_labels[:5], 1):
                print(f"   {j}. {label} (similarity: {similarity:.3f})")
            
            # Show options
            print(f"\n🔽 Options:")
            print(f"   1-5: Select similar existing label")
            print(f"   n:   Create new label")
            print(f"   s:   Skip this case")
            print(f"   q:   Quit and save progress")
            
            while True:
                choice = input(f"\n👉 Your choice: ").strip().lower()
                
                if choice == 'q':
                    print("💾 Saving progress and quitting...")
                    self._save_discovery_progress(new_mappings)
                    return new_mappings
                
                elif choice == 's':
                    print("⏭️ Skipping this case")
                    break
                
                elif choice == 'n':
                    # Create new label
                    new_label = self._create_new_label_interface(case)
                    if new_label:
                        new_mappings.append({
                            'company_id': case['company_id'],
                            'company_description': case['company_description'],
                            'insurance_label': new_label,
                            'naics_code': case['naics_code'],
                            'discovery_method': 'new_label_creation',
                            'uncertainty_score': case['uncertainty_score']
                        })
                        print(f"✅ Created new label: {new_label}")
                    break
                
                elif choice.isdigit() and 1 <= int(choice) <= 5:
                    # Select existing label
                    selected_label = similar_labels[int(choice) - 1][0]
                    similarity = similar_labels[int(choice) - 1][1]
                    
                    new_mappings.append({
                        'company_id': case['company_id'],
                        'company_description': case['company_description'],
                        'insurance_label': selected_label,
                        'naics_code': case['naics_code'],
                        'discovery_method': 'existing_label_assignment',
                        'similarity_score': similarity,
                        'uncertainty_score': case['uncertainty_score']
                    })
                    print(f"✅ Mapped to existing label: {selected_label}")
                    break
                
                else:
                    print("❌ Invalid choice. Please enter 1-5, n, s, or q")
        
        self._save_discovery_progress(new_mappings)
        return new_mappings
    
    def _get_similar_labels(self, company_description: str) -> List[Tuple[str, float]]:
        """Get existing labels ranked by similarity to company description"""
        
        # Get embedding for company description
        company_embedding = self.embedding_model.encode([company_description])
        
        # Get embeddings for current labels
        if not hasattr(self, 'label_embeddings'):
            self.label_embeddings = self.embedding_model.encode(self.current_labels)
        
        # Calculate similarities
        similarities = cosine_similarity(company_embedding, self.label_embeddings)[0]
        
        # Rank labels by similarity
        ranked_labels = [(self.current_labels[i], similarities[i]) 
                        for i in range(len(self.current_labels))]
        ranked_labels.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_labels
    
    def _create_new_label_interface(self, case: Dict) -> Optional[str]:
        """Interface for creating new insurance labels"""
        print(f"\n🆕 Create New Label")
        print(f"   Company: {case['company_description']}")
        print(f"   NAICS: {case['naics_code']}")
        
        # Option 1: Select from unused taxonomy labels
        unused_labels = [label for label in self.taxonomy_labels 
                        if label not in self.current_labels]
        
        if unused_labels:
            print(f"\n📋 Unused Taxonomy Labels (showing top 10):")
            # Find most similar unused labels
            similar_unused = self._get_similar_unused_labels(case['company_description'], unused_labels)
            
            for i, (label, similarity) in enumerate(similar_unused[:10], 1):
                print(f"   {i:2d}. {label} (similarity: {similarity:.3f})")
            
            print(f"\n🔽 Options:")
            print(f"   1-10: Select unused taxonomy label")
            print(f"   c:    Create completely custom label")
            print(f"   b:    Back to previous menu")
            
            while True:
                choice = input(f"\n👉 Your choice: ").strip().lower()
                
                if choice == 'b':
                    return None
                
                elif choice == 'c':
                    # Create custom label
                    custom_label = input(f"Enter custom label name: ").strip()
                    if custom_label:
                        return custom_label
                    
                elif choice.isdigit() and 1 <= int(choice) <= 10:
                    # Select unused taxonomy label
                    selected_label = similar_unused[int(choice) - 1][0]
                    return selected_label
                
                else:
                    print("❌ Invalid choice. Please enter 1-10, c, or b")
        
        else:
            # No unused labels, create custom
            custom_label = input(f"Enter custom label name: ").strip()
            if custom_label:
                return custom_label
        
        return None
    
    def _get_similar_unused_labels(self, company_description: str, unused_labels: List[str]) -> List[Tuple[str, float]]:
        """Get unused taxonomy labels ranked by similarity"""
        
        # Get embedding for company description
        company_embedding = self.embedding_model.encode([company_description])
        
        # Get embeddings for unused labels
        unused_embeddings = self.embedding_model.encode(unused_labels)
        
        # Calculate similarities
        similarities = cosine_similarity(company_embedding, unused_embeddings)[0]
        
        # Rank labels by similarity
        ranked_labels = [(unused_labels[i], similarities[i]) 
                        for i in range(len(unused_labels))]
        ranked_labels.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_labels
    
    def _save_discovery_progress(self, new_mappings: List[Dict]):
        """Save progress from label discovery session"""
        
        # Update progress
        self.progress["manual_mappings"].update({
            f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}": new_mappings
        })
        
        self.progress["uncertain_examples_reviewed"] += len(new_mappings)
        self.progress["current_iteration"] += 1
        
        # Save to file
        self._save_progress()
        
        print(f"✅ Saved {len(new_mappings)} new mappings")
    
    def retrain_with_new_labels(self, new_mappings: List[Dict]):
        """Retrain classifier with newly discovered labels"""
        print(f"\n🔄 Phase 4: Retraining with New Labels")
        print("=" * 50)
        
        if not new_mappings:
            print("❌ No new mappings to retrain with.")
            return
        
        # Combine original training data with new mappings
        new_training_data = pd.DataFrame(new_mappings)
        
        # Combine with existing training data
        combined_training = pd.concat([
            self.training_data,
            new_training_data[['company_id', 'company_description', 'insurance_label', 'naics_code']]
        ], ignore_index=True)
        
        print(f"📊 Training Data Summary:")
        print(f"   Original examples: {len(self.training_data)}")
        print(f"   New examples: {len(new_training_data)}")
        print(f"   Total examples: {len(combined_training)}")
        
        # Update current labels
        self.current_labels = sorted(set(combined_training['insurance_label']))
        print(f"   Total labels: {len(self.current_labels)}")
        
        # Retrain classifier
        self._train_classifier(combined_training)
        
        # Update training data
        self.training_data = combined_training
        
        print(f"✅ Classifier retrained with expanded dataset")
    
    def run_discovery_iteration(self, n_uncertain_cases: int = 10):
        """Run one complete iteration of discovery-based active learning"""
        print(f"\n🚀 Discovery-Based Active Learning Iteration {self.progress['current_iteration'] + 1}")
        print("=" * 70)
        
        # Step 1: Identify uncertain cases
        uncertain_cases = self.identify_uncertain_cases(n_uncertain_cases)
        
        if not uncertain_cases:
            print("🎉 No uncertain cases found. Model is confident on all predictions!")
            return
        
        # Step 2: Interactive label discovery
        new_mappings = self.interactive_label_discovery(uncertain_cases)
        
        if new_mappings:
            # Step 3: Retrain with new labels
            self.retrain_with_new_labels(new_mappings)
            
            # Step 4: Update progress
            self.progress["training_examples"] = len(self.training_data)
            self.progress["total_labels"] = len(self.current_labels)
            self._save_progress()
            
            print(f"\n✅ Iteration Complete!")
            print(f"   New mappings: {len(new_mappings)}")
            print(f"   Total examples: {len(self.training_data)}")
            print(f"   Total labels: {len(self.current_labels)}")
        else:
            print("⏭️ No new mappings created in this iteration.")
    
    def run_complete_pipeline(self, max_iterations: int = 5):
        """Run the complete discovery-based active learning pipeline"""
        print("🎯 DISCOVERY-BASED ACTIVE LEARNING PIPELINE")
        print("=" * 70)
        
        # Phase 1: Bootstrap
        print("Starting with 61 known labels...")
        if not self.bootstrap_initial_classifier():
            print("❌ Bootstrap failed. Cannot proceed.")
            return
        
        # Phase 2: Iterative Discovery
        for iteration in range(max_iterations):
            print(f"\n🔄 Starting Iteration {iteration + 1}/{max_iterations}")
            
            # Check if we have enough uncertain cases
            uncertain_cases = self.identify_uncertain_cases(20)
            
            if len(uncertain_cases) < 5:
                print(f"🎉 Pipeline Complete! Less than 5 uncertain cases remaining.")
                break
            
            # Run discovery iteration
            self.run_discovery_iteration(10)
            
            # Check stopping criteria
            if self.progress["total_labels"] >= 150:  # Stop at 150 labels
                print(f"🎯 Target reached! {self.progress['total_labels']} labels discovered.")
                break
        
        # Final summary
        self._print_final_summary()
    
    def _print_final_summary(self):
        """Print final pipeline summary"""
        print(f"\n🎉 DISCOVERY-BASED ACTIVE LEARNING COMPLETE")
        print("=" * 70)
        print(f"📊 Final Results:")
        print(f"   Starting labels: 61")
        print(f"   Final labels: {self.progress['total_labels']}")
        print(f"   Label expansion: +{self.progress['total_labels'] - 61}")
        print(f"   Training examples: {self.progress['training_examples']}")
        print(f"   Iterations completed: {self.progress['current_iteration']}")
        print(f"   Uncertain cases reviewed: {self.progress['uncertain_examples_reviewed']}")
        
        print(f"\n✅ Ready for Production!")
        print(f"   Coverage: {self.progress['total_labels']}/220 labels ({self.progress['total_labels']/220*100:.1f}%)")
        print(f"   Training data: {self.progress['training_examples']} examples")
        print(f"   Model: Trained and calibrated")


def main():
    """Main execution function"""
    pipeline = DiscoveryActiveLearningPipeline()
    
    # Run complete pipeline
    pipeline.run_complete_pipeline(max_iterations=3)


if __name__ == "__main__":
    main() 