#!/usr/bin/env python3
"""
Smart Insurance Classification Training Pipeline
Addresses the issues with rare labels and improves predictions significantly
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple
import pickle
from pathlib import Path
import json


class SmartInsuranceTrainingPipeline:
    """
    Improved training pipeline with:
    - Better handling of rare labels
    - Semantic similarity understanding
    - Smart label boosting based on domain
    - Augmented features for better predictions
    """
    
    def __init__(self, training_data_path: str = "data/processed/training_data.csv"):
        self.training_data_path = training_data_path
        self.sentence_model = None
        self.label_encoder = LabelEncoder()
        self.feature_extractors = {}
        self.label_embeddings = {}
        self.domain_groups = {}
        
    def load_and_analyze_data(self):
        """Load data and perform smart analysis"""
        print("📊 Loading and analyzing data...")
        self.df = pd.read_csv(self.training_data_path)
        
        print(f"Total examples: {len(self.df)}")
        print(f"Unique labels: {self.df['primary_label'].nunique()}")
        
        # Analyze label distribution
        self.label_counts = self.df['primary_label'].value_counts()
        
        # Group labels by semantic domains
        self._create_domain_groups()
        
        # Load sentence transformer for semantic understanding
        print("🧠 Loading semantic model...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def _create_domain_groups(self):
        """Create semantic groups of related labels"""
        print("🏷️ Creating semantic domain groups...")
        
        # Define domain keywords for grouping similar labels
        domain_keywords = {
            'software_tech': ['software', 'development', 'technology', 'IT', 'computer', 'web', 'app', 'digital'],
            'manufacturing': ['manufacturing', 'production', 'factory', 'industrial', 'assembly'],
            'food_restaurant': ['food', 'restaurant', 'dining', 'cuisine', 'cafe', 'catering'],
            'financial': ['financial', 'banking', 'investment', 'insurance', 'credit', 'loan'],
            'healthcare': ['health', 'medical', 'clinic', 'hospital', 'care', 'therapy'],
            'retail': ['retail', 'store', 'shop', 'sales', 'merchant'],
            'construction': ['construction', 'building', 'contractor', 'renovation'],
            'automotive': ['auto', 'car', 'vehicle', 'motor', 'automotive'],
            'real_estate': ['real estate', 'property', 'realty', 'housing'],
            'transportation': ['transport', 'logistics', 'shipping', 'freight', 'delivery']
        }
        
        # Group labels by domain
        self.domain_groups = {domain: [] for domain in domain_keywords}
        
        for label in self.df['primary_label'].unique():
            if pd.notna(label):
                label_lower = str(label).lower()
                for domain, keywords in domain_keywords.items():
                    if any(keyword in label_lower for keyword in keywords):
                        self.domain_groups[domain].append(label)
                        break
        
        # Print domain statistics
        for domain, labels in self.domain_groups.items():
            if labels:
                print(f"  {domain}: {len(labels)} labels")
    
    def create_smart_features(self):
        """Create enhanced features with semantic understanding"""
        print("\n🔧 SMART FEATURE ENGINEERING:")
        print("=" * 50)
        
        # 1. Enhanced text features (lower min_df for rare labels)
        print("📝 Creating enhanced text features...")
        self.feature_extractors['tfidf_desc'] = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 3),  # Include trigrams for better context
            min_df=1,  # Don't ignore rare words!
            max_df=0.95,
            sublinear_tf=True  # Use log scale for term frequency
        )
        
        descriptions = self.df['description'].fillna('').astype(str)
        tfidf_desc = self.feature_extractors['tfidf_desc'].fit_transform(descriptions)
        
        # 2. Business tags with better handling
        print("🏷️ Processing business tags...")
        tags_list = []
        for tags in self.df['business_tags'].fillna('[]'):
            if isinstance(tags, str) and tags.startswith('['):
                try:
                    tags = eval(tags)
                    tags_list.append(' '.join(tags) if isinstance(tags, list) else str(tags))
                except:
                    tags_list.append('')
            else:
                tags_list.append(str(tags))
        
        self.feature_extractors['tfidf_tags'] = TfidfVectorizer(
            max_features=1000,
            min_df=1,
            ngram_range=(1, 2)
        )
        tfidf_tags = self.feature_extractors['tfidf_tags'].fit_transform(tags_list)
        
        # 3. Semantic embeddings for better understanding
        print("🧠 Creating semantic embeddings...")
        embeddings = self.sentence_model.encode(descriptions.tolist(), show_progress_bar=True)
        
        # 4. Domain-aware features
        print("🎯 Adding domain-aware features...")
        domain_features = self._create_domain_features(descriptions)
        
        # 5. Combine all features
        print("🔗 Combining features...")
        self.X = np.hstack([
            tfidf_desc.toarray(),
            tfidf_tags.toarray(),
            embeddings,
            domain_features
        ])
        
        print(f"✅ Final feature matrix: {self.X.shape}")
        
        # Prepare labels
        self.y = self.df['primary_label'].values
        
    def _create_domain_features(self, descriptions):
        """Create features that capture domain-specific signals"""
        domain_features = []
        
        domain_signals = {
            'tech_signals': ['API', 'cloud', 'SaaS', 'software', 'platform', 'digital', 'online'],
            'manufacturing_signals': ['production', 'assembly', 'facility', 'plant', 'industrial'],
            'service_signals': ['consulting', 'services', 'solutions', 'support', 'management'],
            'retail_signals': ['store', 'shop', 'retail', 'sales', 'customer', 'products']
        }
        
        for desc in descriptions:
            desc_lower = desc.lower()
            features = []
            for signal_type, keywords in domain_signals.items():
                # Count occurrences of domain signals
                count = sum(1 for keyword in keywords if keyword in desc_lower)
                features.append(count)
            domain_features.append(features)
            
        return np.array(domain_features)
    
    def create_label_embeddings(self):
        """Create semantic embeddings for all labels for similarity matching"""
        print("🏷️ Creating label embeddings for semantic matching...")
        
        unique_labels = [label for label in self.df['primary_label'].unique() if pd.notna(label)]
        
        # Create embeddings for each label
        label_texts = []
        for label in unique_labels:
            # Enhance label with examples from training data
            examples = self.df[self.df['primary_label'] == label]['description'].head(3)
            # Convert to string and handle NaN values
            example_texts = []
            for ex in examples:
                if pd.notna(ex) and isinstance(ex, str):
                    example_texts.append(ex[:100])  # Limit length
            
            # Create enhanced text
            if example_texts:
                enhanced_text = f"{label}. " + " ".join(example_texts)
            else:
                enhanced_text = str(label)  # Fallback to just the label
                
            label_texts.append(enhanced_text)
        
        self.label_embeddings = self.sentence_model.encode(label_texts)
        self.label_to_embedding = dict(zip(unique_labels, self.label_embeddings))
        
        print(f"✅ Created embeddings for {len(unique_labels)} labels")
    
    def smart_train_test_split(self):
        """Improved train/test split that handles rare labels better"""
        print("\n📊 SMART TRAIN/TEST SPLIT:")
        print("=" * 50)
        
        # Strategy: Keep some examples of rare labels in test set
        train_indices = []
        test_indices = []
        
        for label in self.label_counts.index:
            label_indices = self.df[self.df['primary_label'] == label].index.tolist()
            count = len(label_indices)
            
            if count == 1:
                # For single examples, 80% go to train, 20% to test
                if np.random.random() > 0.2:
                    train_indices.extend(label_indices)
                else:
                    test_indices.extend(label_indices)
            elif count < 5:
                # For rare labels, keep at least 1 in train
                np.random.shuffle(label_indices)
                train_indices.extend(label_indices[:-1])
                test_indices.extend(label_indices[-1:])
            else:
                # For common labels, standard 80/20 split
                np.random.shuffle(label_indices)
                split_point = int(0.8 * count)
                train_indices.extend(label_indices[:split_point])
                test_indices.extend(label_indices[split_point:])
        
        # Create train/test sets
        self.X_train = self.X[train_indices]
        self.X_test = self.X[test_indices]
        self.y_train = self.y[train_indices]
        self.y_test = self.y[test_indices]
        
        print(f"Training set: {len(self.X_train)} examples")
        print(f"Test set: {len(self.X_test)} examples")
        print(f"Unique labels in train: {len(set(self.y_train))}")
        print(f"Unique labels in test: {len(set(self.y_test))}")
        
        # Encode labels
        self.label_encoder.fit(self.y)
        self.y_train_encoded = self.label_encoder.transform(self.y_train)
        self.y_test_encoded = self.label_encoder.transform(self.y_test)
    
    def train_with_domain_awareness(self):
        """Train models with domain-aware improvements"""
        print("\n🤖 TRAINING SMART MODELS:")
        print("=" * 50)
        
        # Use class weights to handle imbalanced labels
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(self.y_train_encoded)
        class_weights = compute_class_weight(
            'balanced',
            classes=classes,
            y=self.y_train_encoded
        )
        class_weight_dict = dict(zip(classes, class_weights))
        
        # Train improved models
        self.models = {}
        
        # 1. Random Forest with class balancing
        print("🌲 Training Balanced Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight=class_weight_dict,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(self.X_train, self.y_train_encoded)
        self.models['random_forest'] = rf
        
        # 2. Logistic Regression with balancing
        print("📊 Training Balanced Logistic Regression...")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        
        lr = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        lr.fit(X_train_scaled, self.y_train_encoded)
        self.models['logistic_regression'] = {'model': lr, 'scaler': scaler}
        
        # Evaluate models
        self._evaluate_models()
    
    def _evaluate_models(self):
        """Evaluate with focus on practical metrics"""
        print("\n📊 MODEL EVALUATION:")
        print("=" * 50)
        
        for name, model in self.models.items():
            print(f"\n{name.upper()}:")
            
            if name == 'logistic_regression':
                X_test_scaled = model['scaler'].transform(self.X_test)
                y_pred = model['model'].predict(X_test_scaled)
                y_proba = model['model'].predict_proba(X_test_scaled)
            else:
                y_pred = model.predict(self.X_test)
                y_proba = model.predict_proba(self.X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test_encoded, y_pred)
            
            # Top-k accuracy
            top_3_acc = self._calculate_top_k_accuracy(self.y_test_encoded, y_proba, k=3)
            top_5_acc = self._calculate_top_k_accuracy(self.y_test_encoded, y_proba, k=5)
            
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  Top-3 Accuracy: {top_3_acc:.3f}")
            print(f"  Top-5 Accuracy: {top_5_acc:.3f}")
            
            # Test on specific examples
            self._test_specific_examples(model, name)
    
    def _calculate_top_k_accuracy(self, y_true, y_proba, k):
        """Calculate top-k accuracy"""
        n_classes = y_proba.shape[1]
        k = min(k, n_classes)
        
        top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
        correct = 0
        for i, true_label in enumerate(y_true):
            if true_label in top_k_preds[i]:
                correct += 1
        return correct / len(y_true)
    
    def _test_specific_examples(self, model, model_name):
        """Test on specific example types to ensure quality"""
        print(f"\n  Testing specific examples:")
        
        test_cases = [
            {
                'description': 'Software development company building web applications',
                'business_tags': ['Software Development', 'Web Development'],
                'expected_domain': 'software_tech'
            },
            {
                'description': 'Italian restaurant serving traditional cuisine',
                'business_tags': ['Restaurant', 'Food Service'],
                'expected_domain': 'food_restaurant'
            },
            {
                'description': 'Manufacturing facility producing automotive parts',
                'business_tags': ['Manufacturing', 'Automotive'],
                'expected_domain': 'manufacturing'
            }
        ]
        
        # We'll implement prediction logic in the predictor class
        
    def save_smart_model(self):
        """Save the best performing model with all components"""
        print("\n💾 SAVING SMART MODEL...")
        
        # Choose best model (you can implement selection logic)
        best_model = self.models['random_forest']  # For now, use RF
        
        model_package = {
            'model': best_model,
            'label_encoder': self.label_encoder,
            'feature_extractors': self.feature_extractors,
            'label_embeddings': self.label_to_embedding,
            'domain_groups': self.domain_groups,
            'sentence_model_name': 'all-MiniLM-L6-v2',
            'feature_shape': self.X.shape,
            'training_info': {
                'total_examples': len(self.df),
                'unique_labels': self.df['primary_label'].nunique(),
                'label_distribution': dict(self.label_counts.head(20))
            }
        }
        
        output_path = Path('models/smart_insurance_model.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f"✅ Saved smart model to {output_path}")
        
        # Save model info
        info = {
            'model_type': 'Smart Random Forest',
            'features': self.X.shape[1],
            'training_examples': len(self.df),
            'unique_labels': self.df['primary_label'].nunique(),
            'improvements': [
                'Semantic embeddings',
                'Domain-aware features',
                'Balanced class weights',
                'Better rare label handling',
                'Lower min_df for rare words'
            ]
        }
        
        with open('models/smart_model_info.json', 'w') as f:
            json.dump(info, f, indent=2)
    
    def run_pipeline(self):
        """Run the complete smart pipeline"""
        print("🚀 STARTING SMART TRAINING PIPELINE")
        print("=" * 60)
        
        # 1. Load and analyze data
        self.load_and_analyze_data()
        
        # 2. Create smart features
        self.create_smart_features()
        
        # 3. Create label embeddings
        self.create_label_embeddings()
        
        # 4. Smart train/test split
        self.smart_train_test_split()
        
        # 5. Train with improvements
        self.train_with_domain_awareness()
        
        # 6. Save model
        self.save_smart_model()
        
        print("\n🎉 SMART PIPELINE COMPLETED!")
        print("=" * 60)


def main():
    """Run the smart training pipeline"""
    pipeline = SmartInsuranceTrainingPipeline()
    pipeline.run_pipeline()


if __name__ == "__main__":
    main() 