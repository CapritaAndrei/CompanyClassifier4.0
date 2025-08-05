"""
ML Training Pipeline for Insurance Company Classification
Tests multiple models on the 99% coverage dataset to evaluate if data cleaning is needed
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import ast
import json
from datetime import datetime
from collections import Counter

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

class InsuranceMLPipeline:
    """
    Complete ML pipeline for insurance company classification
    Handles multi-label classification from business tags to insurance labels
    """
    
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.vectorizer = None
        self.label_binarizer = None
        self.models = {}
        self.results = {}
        
        print("🤖 INSURANCE ML PIPELINE INITIALIZED")
        print("="*60)
        print("Goal: Train ML models on 99% coverage dataset")
        print("Task: Multi-label classification (business tags → insurance labels)")
        print("="*60)
    
    def load_data(self):
        """Load the 99% coverage dataset"""
        dataset_path = Path("src/data/output/cleaned_classification_results.csv")
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
            
        print(f"📂 Loading dataset: {dataset_path}")
        self.df = pd.read_csv(dataset_path)
        print(f"✅ Loaded {len(self.df):,} companies")
        
        return self.df
    
    def preprocess_data(self):
        """Preprocess data for ML training"""
        print(f"\n🔄 PREPROCESSING DATA FOR ML TRAINING")
        print("="*50)
        
        # Filter out companies without business tags or labels
        print("📋 Filtering companies with both business tags and labels...")
        
        # Companies with business tags
        has_tags = (
            self.df['business_tags'].notna() & 
            (self.df['business_tags'] != '') & 
            (self.df['business_tags'] != '[]')
        )
        
        # Companies with insurance labels
        has_labels = (
            self.df['insurance_labels'].notna() & 
            (self.df['insurance_labels'] != '') & 
            self.df['num_labels_assigned'] > 0
        )
        
        # Keep companies that have both
        valid_companies = has_tags & has_labels
        
        print(f"   Companies with business tags: {has_tags.sum():,}")
        print(f"   Companies with insurance labels: {has_labels.sum():,}")
        print(f"   Companies with both (usable for ML): {valid_companies.sum():,}")
        
        self.df = self.df[valid_companies].copy()
        
        if len(self.df) == 0:
            raise ValueError("No companies have both business tags and insurance labels!")
        
        print(f"✅ Filtered dataset: {len(self.df):,} companies ready for ML")
        
        # Prepare features (X) - Business tags
        print(f"\n🏷️ Preparing features from business tags...")
        
        business_tags_text = []
        for _, row in self.df.iterrows():
            try:
                tags = ast.literal_eval(row['business_tags'])
                if isinstance(tags, list):
                    # Join tags into a single text string
                    tags_text = ' '.join(tags)
                else:
                    tags_text = str(row['business_tags'])
            except:
                tags_text = str(row['business_tags'])
            
            business_tags_text.append(tags_text)
        
        print(f"✅ Prepared {len(business_tags_text):,} business tag texts")
        
        # Prepare targets (y) - Insurance labels (multi-label)
        print(f"\n🎯 Preparing targets from insurance labels...")
        
        insurance_labels_lists = []
        for _, row in self.df.iterrows():
            if pd.notna(row['insurance_labels']) and row['insurance_labels'] != '':
                labels = [label.strip() for label in row['insurance_labels'].split(';')]
                insurance_labels_lists.append(labels)
            else:
                insurance_labels_lists.append([])
        
        print(f"✅ Prepared {len(insurance_labels_lists):,} insurance label lists")
        
        # Store for later use
        self.business_tags_text = business_tags_text
        self.insurance_labels_lists = insurance_labels_lists
        
        # Label statistics
        all_labels = []
        for labels in insurance_labels_lists:
            all_labels.extend(labels)
        
        label_counts = Counter(all_labels)
        
        print(f"\n📊 Target label statistics:")
        print(f"   Total label assignments: {len(all_labels):,}")
        print(f"   Unique labels: {len(label_counts):,}")
        print(f"   Average labels per company: {len(all_labels)/len(insurance_labels_lists):.1f}")
        print(f"   Most common labels:")
        for label, count in label_counts.most_common(5):
            print(f"      {label}: {count:,} companies")
        
        return business_tags_text, insurance_labels_lists
    
    def create_features_and_targets(self):
        """Create ML-ready features and targets"""
        print(f"\n🔧 CREATING ML FEATURES AND TARGETS")
        print("="*50)
        
        # Create TF-IDF features from business tags
        print("📊 Creating TF-IDF features from business tags...")
        
        self.vectorizer = TfidfVectorizer(
            max_features=5000,  # Limit features to most important
            min_df=2,          # Ignore very rare terms
            max_df=0.8,        # Ignore very common terms
            ngram_range=(1, 2), # Include bigrams
            stop_words='english'
        )
        
        X = self.vectorizer.fit_transform(self.business_tags_text)
        print(f"✅ Created TF-IDF matrix: {X.shape[0]:,} samples × {X.shape[1]:,} features")
        
        # Create multi-label binary targets
        print("🎯 Creating multi-label binary targets...")
        
        self.label_binarizer = MultiLabelBinarizer()
        y = self.label_binarizer.fit_transform(self.insurance_labels_lists)
        
        print(f"✅ Created target matrix: {y.shape[0]:,} samples × {y.shape[1]:,} labels")
        print(f"   Label names: {len(self.label_binarizer.classes_):,} unique labels")
        
        # Show label distribution
        label_sums = y.sum(axis=0)
        print(f"   Label frequency range: {label_sums.min():,} - {label_sums.max():,} assignments")
        
        self.X = X
        self.y = y
        
        return X, y
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into train/test sets"""
        print(f"\n✂️ SPLITTING DATA")
        print("="*30)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=test_size, 
            random_state=random_state,
            shuffle=True
        )
        
        print(f"Training set: {self.X_train.shape[0]:,} samples")
        print(f"Test set: {self.X_test.shape[0]:,} samples")
        print(f"Features: {self.X_train.shape[1]:,}")
        print(f"Labels: {self.y_train.shape[1]:,}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """Train multiple ML models"""
        print(f"\n🎓 TRAINING ML MODELS")
        print("="*40)
        
        # Model configurations
        model_configs = {
            'logistic_regression': {
                'model': MultiOutputClassifier(LogisticRegression(
                    max_iter=1000, 
                    random_state=42,
                    n_jobs=-1
                )),
                'description': 'Logistic Regression (baseline)'
            },
            'random_forest': {
                'model': MultiOutputClassifier(RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )),
                'description': 'Random Forest (ensemble)'
            }
        }
        
        for model_name, config in model_configs.items():
            print(f"\n🤖 Training {config['description']}...")
            
            try:
                # Train model
                start_time = datetime.now()
                model = config['model']
                model.fit(self.X_train, self.y_train)
                training_time = (datetime.now() - start_time).total_seconds()
                
                print(f"✅ Training completed in {training_time:.1f} seconds")
                
                # Store model
                self.models[model_name] = {
                    'model': model,
                    'training_time': training_time,
                    'description': config['description']
                }
                
            except Exception as e:
                print(f"❌ Training failed: {e}")
                continue
        
        print(f"\n✅ Trained {len(self.models)} models successfully")
        return self.models
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        print(f"\n📊 EVALUATING MODELS")
        print("="*40)
        
        for model_name, model_info in self.models.items():
            print(f"\n🔍 Evaluating {model_info['description']}...")
            
            try:
                model = model_info['model']
                
                # Predictions
                y_pred = model.predict(self.X_test)
                
                # Calculate metrics
                # Overall metrics
                n_samples, n_labels = self.y_test.shape
                exact_match_ratio = np.mean(np.all(self.y_test == y_pred, axis=1))
                hamming_loss = np.mean(self.y_test != y_pred)
                
                # Per-label metrics
                precision_per_label = []
                recall_per_label = []
                f1_per_label = []
                
                for i in range(n_labels):
                    y_true_label = self.y_test[:, i]
                    y_pred_label = y_pred[:, i]
                    
                    # Calculate precision, recall, F1 for this label
                    true_positives = np.sum((y_true_label == 1) & (y_pred_label == 1))
                    false_positives = np.sum((y_true_label == 0) & (y_pred_label == 1))
                    false_negatives = np.sum((y_true_label == 1) & (y_pred_label == 0))
                    
                    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    precision_per_label.append(precision)
                    recall_per_label.append(recall)
                    f1_per_label.append(f1)
                
                # Average metrics
                avg_precision = np.mean(precision_per_label)
                avg_recall = np.mean(recall_per_label)
                avg_f1 = np.mean(f1_per_label)
                
                # Coverage metrics
                predicted_any = np.sum(y_pred, axis=1) > 0
                coverage = np.mean(predicted_any)
                
                # Store results
                results = {
                    'exact_match_ratio': exact_match_ratio,
                    'hamming_loss': hamming_loss,
                    'avg_precision': avg_precision,
                    'avg_recall': avg_recall,
                    'avg_f1': avg_f1,
                    'coverage': coverage,
                    'training_time': model_info['training_time']
                }
                
                self.results[model_name] = results
                
                print(f"📈 Results:")
                print(f"   Exact match ratio: {exact_match_ratio:.3f}")
                print(f"   Hamming loss: {hamming_loss:.3f}")
                print(f"   Average precision: {avg_precision:.3f}")
                print(f"   Average recall: {avg_recall:.3f}")
                print(f"   Average F1: {avg_f1:.3f}")
                print(f"   Coverage (% with ≥1 label): {coverage:.3f}")
                print(f"   Training time: {model_info['training_time']:.1f}s")
                
            except Exception as e:
                print(f"❌ Evaluation failed: {e}")
                continue
        
        return self.results
    
    def analyze_results(self):
        """Analyze results to determine if data cleaning is needed"""
        print(f"\n🔬 RESULTS ANALYSIS")
        print("="*40)
        
        if not self.results:
            print("❌ No results to analyze")
            return
        
        # Compare models
        print("🏆 Model comparison:")
        for model_name, results in self.results.items():
            model_desc = self.models[model_name]['description']
            print(f"\n   {model_desc}:")
            print(f"      F1 Score: {results['avg_f1']:.3f}")
            print(f"      Coverage: {results['coverage']:.3f}")
            print(f"      Precision: {results['avg_precision']:.3f}")
            print(f"      Recall: {results['avg_recall']:.3f}")
        
        # Best model
        best_model = max(self.results.keys(), key=lambda k: self.results[k]['avg_f1'])
        best_f1 = self.results[best_model]['avg_f1']
        best_coverage = self.results[best_model]['coverage']
        
        print(f"\n🥇 Best performing model: {self.models[best_model]['description']}")
        print(f"   F1 Score: {best_f1:.3f}")
        print(f"   Coverage: {best_coverage:.3f}")
        
        # Data quality assessment
        print(f"\n🧹 DATA CLEANING RECOMMENDATIONS:")
        print("="*50)
        
        if best_f1 > 0.7 and best_coverage > 0.9:
            print("✅ EXCELLENT: Models perform well on current data")
            print("   → Data cleaning may not be necessary")
            print("   → Consider deploying current models")
            
        elif best_f1 > 0.5 and best_coverage > 0.8:
            print("🟡 GOOD: Decent performance, some room for improvement")
            print("   → Light data cleaning might help")
            print("   → Focus on removing obvious noise labels")
            
        elif best_f1 > 0.3:
            print("🟠 MODERATE: Models struggling, data quality issues likely")
            print("   → Data cleaning strongly recommended")
            print("   → Review low-frequency and cross-sector labels")
            
        else:
            print("❌ POOR: Models performing badly")
            print("   → Major data quality issues")
            print("   → Comprehensive data cleaning required")
            print("   → Consider different feature engineering approach")
        
        return best_model, best_f1, best_coverage
    
    def save_models_and_results(self):
        """Save trained models and results"""
        print(f"\n💾 SAVING MODELS AND RESULTS")
        print("="*40)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save models
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        for model_name, model_info in self.models.items():
            model_path = models_dir / f"{model_name}_{timestamp}.pkl"
            joblib.dump(model_info['model'], model_path)
            print(f"📦 Saved {model_info['description']}: {model_path}")
        
        # Save vectorizer and label binarizer
        joblib.dump(self.vectorizer, models_dir / f"vectorizer_{timestamp}.pkl")
        joblib.dump(self.label_binarizer, models_dir / f"label_binarizer_{timestamp}.pkl")
        
        # Save results
        results_summary = {
            'timestamp': timestamp,
            'dataset_info': {
                'total_companies': len(self.df),
                'features': self.X.shape[1],
                'labels': self.y.shape[1],
                'train_samples': self.X_train.shape[0],
                'test_samples': self.X_test.shape[0]
            },
            'model_results': self.results
        }
        
        results_path = Path(f"ml_results_{timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"📊 Results summary saved: {results_path}")
        
        return models_dir, results_path
    
    def run_full_pipeline(self):
        """Run the complete ML pipeline"""
        try:
            # Load and preprocess data
            self.load_data()
            self.preprocess_data()
            self.create_features_and_targets()
            self.split_data()
            
            # Train and evaluate models
            self.train_models()
            self.evaluate_models()
            
            # Analyze results
            best_model, best_f1, best_coverage = self.analyze_results()
            
            # Save everything
            self.save_models_and_results()
            
            print(f"\n🎉 ML PIPELINE COMPLETE!")
            print(f"✅ Best model F1 score: {best_f1:.3f}")
            print(f"✅ Coverage: {best_coverage:.3f}")
            
            return self.results
            
        except Exception as e:
            print(f"❌ ML pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main execution function"""
    pipeline = InsuranceMLPipeline()
    results = pipeline.run_full_pipeline()
    
    if results:
        print(f"\n🎯 NEXT STEPS:")
        print(f"1. Review model performance metrics")
        print(f"2. Decide on data cleaning based on results")
        print(f"3. Iterate on feature engineering if needed")
        print(f"4. Deploy best performing model")

if __name__ == "__main__":
    main()