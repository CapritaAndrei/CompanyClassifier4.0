"""
Custom Insurance Classification Model Training Pipeline
Uses unified training data (288 examples) to train various ML models
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, top_k_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier

# Deep Learning
try:
    from sentence_transformers import SentenceTransformer
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    print("⚠️ Deep learning libraries not available. Install with: pip install sentence-transformers")
    DEEP_LEARNING_AVAILABLE = False


class InsuranceClassificationTrainingPipeline:
    """
    Comprehensive training pipeline for insurance classification models
    """
    
    def __init__(self, 
                 training_data_path: str = "data/unified_training_data.csv",
                 taxonomy_path: str = "data/input/insurance_taxonomy - insurance_taxonomy.csv",
                 output_dir: str = "models/"):
        """
        Initialize training pipeline
        
        Args:
            training_data_path: Path to unified training data CSV
            taxonomy_path: Path to insurance taxonomy CSV  
            output_dir: Directory to save trained models
        """
        self.training_data_path = training_data_path
        self.taxonomy_path = taxonomy_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.training_df = None
        self.taxonomy_df = None
        self.label_encoder = LabelEncoder()
        self.multilabel_binarizer = MultiLabelBinarizer()
        self.feature_extractors = {}
        self.models = {}
        self.results = {}
        
        # Load sentence transformer if available
        if DEEP_LEARNING_AVAILABLE:
            print("🧠 Loading sentence transformer model...")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.sentence_model = None
            
    def load_data(self):
        """Load and preprocess training data"""
        print("📊 Loading training data...")
        
        # Load training data
        self.training_df = pd.read_csv(self.training_data_path)
        print(f"✅ Loaded {len(self.training_df)} training examples")
        
        # Load taxonomy
        if Path(self.taxonomy_path).exists():
            self.taxonomy_df = pd.read_csv(self.taxonomy_path)
            print(f"✅ Loaded {len(self.taxonomy_df)} taxonomy labels")
        else:
            print("⚠️ Taxonomy file not found, using labels from training data")
            
        # Data quality analysis
        self._analyze_data_quality()
        
    def _analyze_data_quality(self):
        """Analyze training data quality and distribution"""
        print("\n📈 DATA QUALITY ANALYSIS:")
        print("=" * 50)
        
        # Basic statistics
        print(f"Total examples: {len(self.training_df)}")
        print(f"Unique primary labels: {self.training_df['primary_label'].nunique()}")
        print(f"Labels with multiple examples: {self.training_df['primary_label'].value_counts().gt(1).sum()}")
        
        # Data sources
        print(f"\nData sources:")
        for source, count in self.training_df['source'].value_counts().items():
            print(f"  • {source}: {count} examples")
            
        # Confidence distribution
        print(f"\nConfidence distribution:")
        for conf, count in self.training_df['confidence'].value_counts().items():
            print(f"  • {conf}: {count} examples ({count/len(self.training_df):.1%})")
            
        # Label distribution analysis
        label_counts = self.training_df['primary_label'].value_counts()
        print(f"\nLabel distribution:")
        print(f"  • Labels with 1 example: {(label_counts == 1).sum()}")
        print(f"  • Labels with 2-5 examples: {((label_counts >= 2) & (label_counts <= 5)).sum()}")
        print(f"  • Labels with 6+ examples: {(label_counts >= 6).sum()}")
        
        # Top labels
        print(f"\n🏆 Top 10 most frequent labels:")
        for label, count in label_counts.head(10).items():
            print(f"  • {label}: {count} examples")
            
    def prepare_features(self):
        """Extract and prepare features for training"""
        print("\n🔧 FEATURE ENGINEERING:")
        print("=" * 50)
        
        # 1. Text features from description
        print("📝 Extracting text features from descriptions...")
        descriptions = self.training_df['description'].fillna('').astype(str)
        
        # TF-IDF features
        self.feature_extractors['tfidf_desc'] = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        tfidf_features = self.feature_extractors['tfidf_desc'].fit_transform(descriptions)
        print(f"  ✅ TF-IDF features: {tfidf_features.shape}")
        
        # 2. Business tags features
        print("🏷️ Extracting business tags features...")
        business_tags = self.training_df['business_tags'].fillna('[]').astype(str)
        
        # Parse business tags (they're stored as string representations of lists)
        parsed_tags = []
        for tags_str in business_tags:
            try:
                if tags_str.startswith('[') and tags_str.endswith(']'):
                    tags = eval(tags_str)
                    parsed_tags.append(' '.join(tags) if isinstance(tags, list) else str(tags))
                else:
                    parsed_tags.append(tags_str)
            except:
                parsed_tags.append('')
                
        self.feature_extractors['tfidf_tags'] = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        tags_features = self.feature_extractors['tfidf_tags'].fit_transform(parsed_tags)
        print(f"  ✅ Business tags features: {tags_features.shape}")
        
        # 3. Categorical features
        print("📊 Processing categorical features...")
        categorical_features = []
        
        # One-hot encode sector, category, niche
        for col in ['sector', 'category', 'niche']:
            if col in self.training_df.columns:
                dummies = pd.get_dummies(self.training_df[col].fillna('Unknown'), prefix=col)
                categorical_features.append(dummies.values)
                self.feature_extractors[f'{col}_columns'] = dummies.columns.tolist()
                
        if categorical_features:
            categorical_matrix = np.hstack(categorical_features)
            print(f"  ✅ Categorical features: {categorical_matrix.shape}")
        else:
            categorical_matrix = np.array([]).reshape(len(self.training_df), 0)
            
        # 4. Sentence embeddings (if available)
        if self.sentence_model is not None:
            print("🧠 Computing sentence embeddings...")
            embeddings = self.sentence_model.encode(descriptions.tolist(), show_progress_bar=True)
            print(f"  ✅ Sentence embeddings: {embeddings.shape}")
        else:
            embeddings = np.array([]).reshape(len(self.training_df), 0)
            
        # 5. Combine all features
        print("🔗 Combining all features...")
        
        # Store feature matrices
        self.features = {
            'tfidf_desc': tfidf_features.toarray(),
            'tfidf_tags': tags_features.toarray(),
            'categorical': categorical_matrix,
            'embeddings': embeddings
        }
        
        # Combined feature matrix
        feature_matrices = [
            self.features['tfidf_desc'],
            self.features['tfidf_tags'], 
            self.features['categorical']
        ]
        
        if embeddings.shape[1] > 0:
            feature_matrices.append(embeddings)
            
        self.X = np.hstack(feature_matrices)
        print(f"✅ Combined feature matrix: {self.X.shape}")
        
        # Prepare labels
        self.y_primary = self.training_df['primary_label'].values
        
        # Multi-label preparation (selected_labels)
        selected_labels = []
        for labels_str in self.training_df['selected_labels'].fillna('[]'):
            try:
                if labels_str.startswith('[') and labels_str.endswith(']'):
                    labels = eval(labels_str)
                    selected_labels.append(labels if isinstance(labels, list) else [labels])
                else:
                    selected_labels.append([labels_str])
            except:
                selected_labels.append([])
                
        self.y_multilabel = self.multilabel_binarizer.fit_transform(selected_labels)
        print(f"✅ Multi-label matrix: {self.y_multilabel.shape}")
        
    def prepare_train_test_split(self, test_size: float = 0.15, random_state: int = 42):
        """Prepare train/test split with stratification"""
        print(f"\n📊 TRAIN/TEST SPLIT:")
        print("=" * 50)
        
        # Filter labels with multiple examples for stratified split
        label_counts = pd.Series(self.y_primary).value_counts()
        labels_with_multiple = label_counts[label_counts > 1].index
        
        # Create mask for stratifiable examples
        stratifiable_mask = pd.Series(self.y_primary).isin(labels_with_multiple)
        
        print(f"Total examples: {len(self.y_primary)}")
        print(f"Examples with stratifiable labels: {stratifiable_mask.sum()}")
        print(f"Single-example labels: {(~stratifiable_mask).sum()}")
        
        # Check if stratification is possible
        stratifiable_examples = stratifiable_mask.sum()
        unique_stratifiable_labels = len(labels_with_multiple)
        required_test_size = int(stratifiable_examples * test_size)
        
        print(f"Unique labels with multiple examples: {unique_stratifiable_labels}")
        print(f"Required test size: {required_test_size}")
        
        # Adjust test size if needed for stratification
        if required_test_size < unique_stratifiable_labels:
            # Need at least one example per label in test set
            adjusted_test_size = max(unique_stratifiable_labels / stratifiable_examples, 0.1)
            print(f"⚠️ Adjusting test size from {test_size:.2f} to {adjusted_test_size:.2f} for stratification")
            test_size = adjusted_test_size
        
        # Split stratifiable examples
        if stratifiable_mask.sum() > 0 and unique_stratifiable_labels > 1:
            try:
                X_strat = self.X[stratifiable_mask]
                y_strat = self.y_primary[stratifiable_mask]
                
                X_train_strat, X_test_strat, y_train_strat, y_test_strat = train_test_split(
                    X_strat, y_strat,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=y_strat
                )
                
                # Add single-example labels to training set
                X_single = self.X[~stratifiable_mask]
                y_single = self.y_primary[~stratifiable_mask]
                
                self.X_train = np.vstack([X_train_strat, X_single])
                self.y_train = np.hstack([y_train_strat, y_single])
                self.X_test = X_test_strat
                self.y_test = y_test_strat
                
                print("✅ Used stratified split for multi-example labels")
                
            except ValueError as e:
                print(f"⚠️ Stratification failed: {e}")
                print("   Falling back to random split")
                # Fallback: random split if stratification fails
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    self.X, self.y_primary,
                    test_size=0.15,  # Use smaller test size for random split
                    random_state=random_state
                )
        else:
            # Fallback: random split if no stratification possible
            print("⚠️ No stratification possible, using random split")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y_primary,
                test_size=0.15,  # Use smaller test size
                random_state=random_state
            )
            
        print(f"Training set: {self.X_train.shape[0]} examples")
        print(f"Test set: {self.X_test.shape[0]} examples")
        
        # Encode labels
        self.label_encoder.fit(self.y_primary)
        self.y_train_encoded = self.label_encoder.transform(self.y_train)
        self.y_test_encoded = self.label_encoder.transform(self.y_test)
        
        print(f"Unique labels in training: {len(set(self.y_train))}")
        print(f"Unique labels in test: {len(set(self.y_test))}")
        
    def train_models(self):
        """Train multiple classification models"""
        print(f"\n🤖 MODEL TRAINING:")
        print("=" * 50)
        
        # Define models to train
        models_to_train = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'Naive Bayes': GaussianNB()
        }
        
        # Train each model
        for name, model in models_to_train.items():
            print(f"\n🔄 Training {name}...")
            
            try:
                # Scale features for Logistic Regression
                if name in ['Logistic Regression']:
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(self.X_train)
                    X_test_scaled = scaler.transform(self.X_test)
                    
                    model.fit(X_train_scaled, self.y_train_encoded)
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)
                    
                    # Store scaler with model
                    self.models[name] = {
                        'model': model,
                        'scaler': scaler,
                        'requires_scaling': True
                    }
                else:
                    model.fit(self.X_train, self.y_train_encoded)
                    y_pred = model.predict(self.X_test)
                    y_pred_proba = model.predict_proba(self.X_test)
                    
                    self.models[name] = {
                        'model': model,
                        'requires_scaling': False
                    }
                
                # Calculate metrics
                accuracy = accuracy_score(self.y_test_encoded, y_pred)
                
                # Top-k accuracy (handle case where k > number of classes)
                n_classes = y_pred_proba.shape[1]
                k_3 = min(3, n_classes)
                k_5 = min(5, n_classes)
                
                # Custom top-k accuracy calculation to handle class mismatch
                def calculate_top_k_accuracy(y_true, y_prob, k):
                    top_k_preds = np.argsort(y_prob, axis=1)[:, -k:]
                    correct = 0
                    for i, true_label in enumerate(y_true):
                        if true_label in top_k_preds[i]:
                            correct += 1
                    return correct / len(y_true)
                
                top_3_acc = calculate_top_k_accuracy(self.y_test_encoded, y_pred_proba, k_3)
                top_5_acc = calculate_top_k_accuracy(self.y_test_encoded, y_pred_proba, k_5)
                
                # Precision, Recall, F1 (only for classes present in test set)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    self.y_test_encoded, y_pred, average='weighted', zero_division=0
                )
                
                # Store results
                self.results[name] = {
                    'accuracy': accuracy,
                    'top_3_accuracy': top_3_acc,
                    'top_5_accuracy': top_5_acc,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"  ✅ {name}:")
                print(f"     Accuracy: {accuracy:.3f}")
                print(f"     Top-3 Accuracy: {top_3_acc:.3f}")
                print(f"     Top-5 Accuracy: {top_5_acc:.3f}")
                print(f"     F1-Score: {f1:.3f}")
                
            except Exception as e:
                print(f"  ❌ Failed to train {name}: {e}")
                
    def evaluate_models(self):
        """Comprehensive model evaluation"""
        print(f"\n📊 MODEL EVALUATION:")
        print("=" * 50)
        
        # Create results DataFrame
        results_data = []
        for name, metrics in self.results.items():
            results_data.append({
                'Model': name,
                'Accuracy': metrics['accuracy'],
                'Top-3 Accuracy': metrics['top_3_accuracy'],
                'Top-5 Accuracy': metrics['top_5_accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score']
            })
            
        results_df = pd.DataFrame(results_data)
        results_df = results_df.sort_values('F1-Score', ascending=False)
        
        print("🏆 MODEL PERFORMANCE RANKING:")
        print(results_df.to_string(index=False, float_format='%.3f'))
        
        # Best model
        if len(results_df) > 0:
            best_model_name = results_df.iloc[0]['Model']
            print(f"\n🥇 Best performing model: {best_model_name}")
            
            # Detailed evaluation for best model
            self._detailed_evaluation(best_model_name)
        
        return results_df
        
    def _detailed_evaluation(self, model_name: str):
        """Detailed evaluation for a specific model"""
        print(f"\n🔍 DETAILED EVALUATION - {model_name}:")
        print("=" * 50)
        
        y_pred = self.results[model_name]['predictions']
        
        # Classification report
        print("📋 Classification Report (Top 10 labels):")
        unique_test_labels = np.unique(self.y_test_encoded)
        if len(unique_test_labels) > 10:
            # Show only top 10 most frequent labels in test set
            test_label_counts = pd.Series(self.y_test_encoded).value_counts()
            top_labels = test_label_counts.head(10).index.values
            
            mask = np.isin(self.y_test_encoded, top_labels)
            if mask.sum() > 0:
                # Get actual labels present in masked data
                actual_labels_in_mask = np.unique(self.y_test_encoded[mask])
                actual_pred_in_mask = np.unique(y_pred[mask])
                all_labels_in_mask = np.unique(np.concatenate([actual_labels_in_mask, actual_pred_in_mask]))
                
                report = classification_report(
                    self.y_test_encoded[mask], 
                    y_pred[mask],
                    labels=all_labels_in_mask,
                    target_names=self.label_encoder.classes_[all_labels_in_mask],
                    zero_division=0
                )
                print(report)
        else:
            report = classification_report(
                self.y_test_encoded, y_pred,
                target_names=self.label_encoder.classes_[unique_test_labels],
                zero_division=0
            )
            print(report)
                
    def save_models(self):
        """Save trained models and components"""
        print(f"\n💾 SAVING MODELS:")
        print("=" * 50)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save each model
        for name, model_data in self.models.items():
            model_filename = f"{name.lower().replace(' ', '_')}_model_{timestamp}.pkl"
            model_path = self.output_dir / model_filename
            
            # Package model with all necessary components
            model_package = {
                'model': model_data['model'],
                'scaler': model_data.get('scaler'),
                'requires_scaling': model_data['requires_scaling'],
                'label_encoder': self.label_encoder,
                'feature_extractors': self.feature_extractors,
                'multilabel_binarizer': self.multilabel_binarizer,
                'training_info': {
                    'timestamp': timestamp,
                    'training_examples': len(self.training_df),
                    'features_shape': self.X.shape,
                    'unique_labels': len(self.label_encoder.classes_),
                    'model_name': name
                }
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_package, f)
                
            print(f"✅ Saved {name} model to {model_path}")
            
        # Save training results
        results_path = self.output_dir / f"training_results_{timestamp}.json"
        results_to_save = {}
        for name, metrics in self.results.items():
            # Convert numpy arrays to lists for JSON serialization
            results_to_save[name] = {
                'accuracy': float(metrics['accuracy']),
                'top_3_accuracy': float(metrics['top_3_accuracy']),
                'top_5_accuracy': float(metrics['top_5_accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1_score': float(metrics['f1_score'])
            }
            
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
            
        print(f"✅ Saved training results to {results_path}")
        
    def run_full_pipeline(self):
        """Run the complete training pipeline"""
        print("🚀 STARTING COMPLETE TRAINING PIPELINE")
        print("=" * 60)
        
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Feature engineering
            self.prepare_features()
            
            # Step 3: Train/test split
            self.prepare_train_test_split()
            
            # Step 4: Train models
            self.train_models()
            
            # Step 5: Evaluate models
            results_df = self.evaluate_models()
            
            # Step 6: Save models
            self.save_models()
            
            print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            
            return results_df
            
        except Exception as e:
            print(f"\n❌ PIPELINE FAILED: {e}")
            raise
            

def main():
    """Main execution function"""
    print("🏢 Insurance Classification Model Training Pipeline")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = InsuranceClassificationTrainingPipeline()
    
    # Run full pipeline
    results = pipeline.run_full_pipeline()
    
    print(f"\n📊 FINAL RESULTS:")
    print(results.to_string(index=False, float_format='%.3f'))


if __name__ == "__main__":
    main() 