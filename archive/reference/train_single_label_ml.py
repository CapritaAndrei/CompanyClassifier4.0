"""
Single-Label ML Training Pipeline
Uses the 30% clean dataset (classification differences) for training
Much simpler problem: business tags → single insurance label
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

class SingleLabelMLPipeline:
    """
    Single-label ML pipeline for insurance classification
    Uses the clean 30% dataset with 1 label per company
    """
    
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.vectorizer = None
        self.label_encoder = None
        self.models = {}
        self.results = {}
        
        print("🎯 SINGLE-LABEL ML PIPELINE INITIALIZED")
        print("="*60)
        print("Goal: Train on clean 30% dataset (single label per company)")
        print("Task: Single-label classification (business tags → 1 insurance label)")
        print("="*60)
    
    def load_clean_dataset(self):
        """Load the clean 30% dataset with single labels"""
        dataset_path = Path("src/data/output/classification_differences_20250804_214821.csv")
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Clean dataset not found: {dataset_path}")
            
        print(f"📂 Loading clean dataset: {dataset_path}")
        self.df = pd.read_csv(dataset_path)
        print(f"✅ Loaded {len(self.df):,} companies with single labels")
        
        # Verify it's single-label
        if 'labels_added' in self.df.columns:
            single_label_mask = self.df['labels_added'] == 1
            print(f"   Companies with exactly 1 label: {single_label_mask.sum():,}")
            
            if single_label_mask.sum() != len(self.df):
                print(f"⚠️  Warning: Not all companies have exactly 1 label")
                # Filter to single-label only
                self.df = self.df[single_label_mask].copy()
                print(f"   Filtered to {len(self.df):,} single-label companies")
        
        return self.df
    
    def preprocess_single_label_data(self):
        """Preprocess for single-label classification"""
        print(f"\n🔄 PREPROCESSING SINGLE-LABEL DATA")
        print("="*50)
        
        # Check for required columns
        required_cols = ['business_tags', 'insurance_labels']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Required column missing: {col}")
        
        # Filter companies with both business tags and insurance labels
        has_tags = (
            self.df['business_tags'].notna() & 
            (self.df['business_tags'] != '') & 
            (self.df['business_tags'] != '[]')
        )
        
        has_labels = (
            self.df['insurance_labels'].notna() & 
            (self.df['insurance_labels'] != '')
        )
        
        valid_companies = has_tags & has_labels
        
        print(f"   Total companies: {len(self.df):,}")
        print(f"   Companies with business tags: {has_tags.sum():,}")
        print(f"   Companies with insurance labels: {has_labels.sum():,}")
        print(f"   Valid companies for ML: {valid_companies.sum():,}")
        
        self.df = self.df[valid_companies].copy()
        
        if len(self.df) == 0:
            raise ValueError("No valid companies for ML training!")
        
        # Prepare features (business tags)
        print(f"\n🏷️ Preparing business tags features...")
        
        business_tags_text = []
        for _, row in self.df.iterrows():
            try:
                # Handle different possible formats
                if pd.notna(row.get('synthetic_business_tags')) and row['synthetic_business_tags'] != '':
                    # Use synthetic tags if available
                    tags = ast.literal_eval(row['synthetic_business_tags'])
                    if isinstance(tags, list):
                        tags_text = ' '.join(tags)
                    else:
                        tags_text = str(row['synthetic_business_tags'])
                elif pd.notna(row.get('original_business_tags')) and row['original_business_tags'] != '':
                    # Use original tags
                    tags = ast.literal_eval(row['original_business_tags'])
                    if isinstance(tags, list):
                        tags_text = ' '.join(tags)
                    else:
                        tags_text = str(row['original_business_tags'])
                else:
                    # Fallback to business_tags column
                    tags = ast.literal_eval(row['business_tags'])
                    if isinstance(tags, list):
                        tags_text = ' '.join(tags)
                    else:
                        tags_text = str(row['business_tags'])
            except:
                # Last resort - use as string
                tags_text = str(row['business_tags'])
            
            business_tags_text.append(tags_text)
        
        print(f"✅ Prepared {len(business_tags_text):,} business tag texts")
        
        # Prepare single labels
        print(f"\n🎯 Preparing single insurance labels...")
        
        insurance_labels = []
        for _, row in self.df.iterrows():
            # These should be single labels, but let's handle safely
            label_str = str(row['insurance_labels']).strip()
            if ';' in label_str:
                # Take the first label if multiple exist
                labels = [l.strip() for l in label_str.split(';')]
                insurance_labels.append(labels[0])
            else:
                insurance_labels.append(label_str)
        
        print(f"✅ Prepared {len(insurance_labels):,} single insurance labels")
        
        # Label statistics  
        label_counts = Counter(insurance_labels)
        
        print(f"\n📊 Initial label statistics:")
        print(f"   Unique labels: {len(label_counts):,}")
        print(f"   Most common labels:")
        for label, count in label_counts.most_common(10):
            print(f"      {label}: {count:,} companies")
        
        print(f"   Least common labels:")
        for label, count in label_counts.most_common()[-5:]:
            print(f"      {label}: {count} companies")
        
        # Filter out rare labels (need at least 3 samples for stratified split)
        MIN_SAMPLES_PER_LABEL = 3
        print(f"\n🔧 Filtering labels with <{MIN_SAMPLES_PER_LABEL} samples...")
        
        valid_labels = {label for label, count in label_counts.items() if count >= MIN_SAMPLES_PER_LABEL}
        
        # Keep only companies with valid labels
        filtered_tags = []
        filtered_labels = []
        
        for i, label in enumerate(insurance_labels):
            if label in valid_labels:
                filtered_tags.append(business_tags_text[i])
                filtered_labels.append(label)
        
        print(f"   Removed {len(label_counts) - len(valid_labels)} rare labels")
        print(f"   Kept {len(filtered_labels):,} companies with {len(valid_labels):,} labels")
        
        # Update label statistics
        filtered_label_counts = Counter(filtered_labels)
        print(f"\n📊 Filtered label statistics:")
        print(f"   Unique labels: {len(filtered_label_counts):,}")
        print(f"   Min samples per label: {min(filtered_label_counts.values()):,}")
        print(f"   Max samples per label: {max(filtered_label_counts.values()):,}")
        print(f"   Average samples per label: {sum(filtered_label_counts.values())/len(filtered_label_counts):.1f}")
        
        # Store processed data
        self.business_tags_text = filtered_tags
        self.insurance_labels = filtered_labels
        
        return filtered_tags, filtered_labels
    
    def create_features_and_targets(self):
        """Create features and targets for single-label classification"""
        print(f"\n🔧 CREATING FEATURES AND TARGETS")
        print("="*50)
        
        # Create TF-IDF features
        print("📊 Creating TF-IDF features from business tags...")
        
        self.vectorizer = TfidfVectorizer(
            max_features=3000,      # Fewer features for smaller dataset
            min_df=2,              # Must appear in at least 2 documents
            max_df=0.9,            # Ignore very common terms
            ngram_range=(1, 2),    # Include bigrams
            stop_words='english'
        )
        
        X = self.vectorizer.fit_transform(self.business_tags_text)
        print(f"✅ Created TF-IDF matrix: {X.shape[0]:,} samples × {X.shape[1]:,} features")
        
        # Encode single labels
        print("🎯 Encoding single labels...")
        
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(self.insurance_labels)
        
        print(f"✅ Encoded labels: {len(y):,} samples, {len(self.label_encoder.classes_):,} unique classes")
        
        # Show class distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"   Class distribution:")
        print(f"      Min class size: {counts.min():,}")
        print(f"      Max class size: {counts.max():,}")
        print(f"      Average class size: {counts.mean():.1f}")
        
        self.X = X
        self.y = y
        
        return X, y
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split data for training and testing"""
        print(f"\n✂️ SPLITTING DATA")
        print("="*30)
        
        # Stratified split to maintain class distribution
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y,  # Maintain class distribution
            shuffle=True
        )
        
        print(f"Training set: {self.X_train.shape[0]:,} samples")
        print(f"Test set: {self.X_test.shape[0]:,} samples") 
        print(f"Features: {self.X_train.shape[1]:,}")
        print(f"Classes: {len(self.label_encoder.classes_):,}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """Train single-label classification models"""
        print(f"\n🎓 TRAINING SINGLE-LABEL MODELS")
        print("="*50)
        
        # Model configurations for single-label classification
        model_configs = {
            'logistic_regression': {
                'model': LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'  # Handle class imbalance
                ),
                'description': 'Logistic Regression (balanced)'
            },
            'random_forest': {
                'model': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'  # Handle class imbalance
                ),
                'description': 'Random Forest (balanced)'
            }
        }
        
        for model_name, config in model_configs.items():
            print(f"\n🤖 Training {config['description']}...")
            
            try:
                start_time = datetime.now()
                model = config['model']
                model.fit(self.X_train, self.y_train)
                training_time = (datetime.now() - start_time).total_seconds()
                
                print(f"✅ Training completed in {training_time:.1f} seconds")
                
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
        """Evaluate single-label classification models"""
        print(f"\n📊 EVALUATING SINGLE-LABEL MODELS")
        print("="*50)
        
        for model_name, model_info in self.models.items():
            print(f"\n🔍 Evaluating {model_info['description']}...")
            
            try:
                model = model_info['model']
                
                # Predictions
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                
                # Classification report
                report = classification_report(
                    self.y_test, y_pred, 
                    target_names=self.label_encoder.classes_,
                    output_dict=True,
                    zero_division=0
                )
                
                # Overall metrics
                precision = report['macro avg']['precision']
                recall = report['macro avg']['recall']
                f1 = report['macro avg']['f1-score']
                
                # Coverage (should be 100% for single-label)
                predicted_labels = len(np.unique(y_pred))
                total_labels = len(self.label_encoder.classes_)
                label_coverage = predicted_labels / total_labels
                
                # Top-1 and Top-3 accuracy (if probabilities available)
                top3_accuracy = 0
                if y_pred_proba is not None:
                    # Top-3 accuracy
                    top3_pred = np.argsort(y_pred_proba, axis=1)[:, -3:]
                    top3_accuracy = np.mean([self.y_test[i] in top3_pred[i] for i in range(len(self.y_test))])
                
                # Store results
                results = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'label_coverage': label_coverage,
                    'top3_accuracy': top3_accuracy,
                    'training_time': model_info['training_time'],
                    'predicted_labels': predicted_labels,
                    'total_labels': total_labels
                }
                
                self.results[model_name] = results
                
                print(f"📈 Results:")
                print(f"   Accuracy: {accuracy:.3f}")
                print(f"   Precision (macro): {precision:.3f}")
                print(f"   Recall (macro): {recall:.3f}")
                print(f"   F1 Score (macro): {f1:.3f}")
                print(f"   Label coverage: {label_coverage:.3f} ({predicted_labels}/{total_labels})")
                if top3_accuracy > 0:
                    print(f"   Top-3 accuracy: {top3_accuracy:.3f}")
                print(f"   Training time: {model_info['training_time']:.1f}s")
                
            except Exception as e:
                print(f"❌ Evaluation failed: {e}")
                continue
        
        return self.results
    
    def analyze_single_label_results(self):
        """Analyze results for single-label classification"""
        print(f"\n🔬 SINGLE-LABEL RESULTS ANALYSIS")
        print("="*50)
        
        if not self.results:
            print("❌ No results to analyze")
            return
        
        # Compare models
        print("🏆 Model comparison:")
        for model_name, results in self.results.items():
            model_desc = self.models[model_name]['description']
            print(f"\n   {model_desc}:")
            print(f"      Accuracy: {results['accuracy']:.3f}")
            print(f"      F1 Score: {results['f1_score']:.3f}")
            print(f"      Precision: {results['precision']:.3f}")
            print(f"      Recall: {results['recall']:.3f}")
        
        # Best model
        best_model = max(self.results.keys(), key=lambda k: self.results[k]['f1_score'])
        best_f1 = self.results[best_model]['f1_score']
        best_accuracy = self.results[best_model]['accuracy']
        
        print(f"\n🥇 Best performing model: {self.models[best_model]['description']}")
        print(f"   Accuracy: {best_accuracy:.3f}")
        print(f"   F1 Score: {best_f1:.3f}")
        
        # Performance assessment
        print(f"\n🎯 PERFORMANCE ASSESSMENT:")
        print("="*40)
        
        if best_f1 > 0.8 and best_accuracy > 0.8:
            print("🎉 EXCELLENT: Outstanding model performance!")
            print("   → Single-label approach works very well")
            print("   → Data quality is high")
            print("   → Ready for production deployment")
            
        elif best_f1 > 0.6 and best_accuracy > 0.7:
            print("✅ VERY GOOD: Strong model performance")
            print("   → Single-label approach is effective")
            print("   → Minor improvements possible with tuning")
            print("   → Consider ensemble methods")
            
        elif best_f1 > 0.4 and best_accuracy > 0.5:
            print("🟡 MODERATE: Decent performance, room for improvement")
            print("   → Feature engineering could help")
            print("   → Consider more advanced models")
            print("   → Review class imbalance issues")
            
        else:
            print("🟠 POOR: Performance needs improvement")
            print("   → Feature engineering required")
            print("   → Consider different approaches")
            print("   → Review data quality")
        
        return best_model, best_f1, best_accuracy
    
    def save_single_label_models(self):
        """Save single-label models and results"""
        print(f"\n💾 SAVING SINGLE-LABEL MODELS")
        print("="*40)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save models
        models_dir = Path("models/single_label")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, model_info in self.models.items():
            model_path = models_dir / f"{model_name}_{timestamp}.pkl"
            joblib.dump(model_info['model'], model_path)
            print(f"📦 Saved {model_info['description']}: {model_path}")
        
        # Save preprocessing components
        joblib.dump(self.vectorizer, models_dir / f"vectorizer_{timestamp}.pkl")
        joblib.dump(self.label_encoder, models_dir / f"label_encoder_{timestamp}.pkl")
        
        # Save results
        results_summary = {
            'timestamp': timestamp,
            'dataset_type': 'single_label_clean',
            'dataset_info': {
                'total_companies': len(self.df),
                'features': self.X.shape[1],
                'classes': len(self.label_encoder.classes_),
                'train_samples': self.X_train.shape[0],
                'test_samples': self.X_test.shape[0]
            },
            'model_results': self.results,
            'label_classes': list(self.label_encoder.classes_)
        }
        
        results_path = Path(f"single_label_ml_results_{timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"📊 Results summary saved: {results_path}")
        
        return models_dir, results_path
    
    def run_single_label_pipeline(self):
        """Run the complete single-label ML pipeline"""
        try:
            # Load and preprocess clean dataset
            self.load_clean_dataset()
            self.preprocess_single_label_data()
            self.create_features_and_targets()
            self.split_data()
            
            # Train and evaluate models
            self.train_models()
            self.evaluate_models()
            
            # Analyze results
            best_model, best_f1, best_accuracy = self.analyze_single_label_results()
            
            # Save everything
            self.save_single_label_models()
            
            print(f"\n🎉 SINGLE-LABEL ML PIPELINE COMPLETE!")
            print(f"✅ Best model accuracy: {best_accuracy:.3f}")
            print(f"✅ Best model F1 score: {best_f1:.3f}")
            
            return self.results
            
        except Exception as e:
            print(f"❌ Single-label ML pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main execution function"""
    pipeline = SingleLabelMLPipeline()
    results = pipeline.run_single_label_pipeline()
    
    if results:
        print(f"\n🎯 NEXT STEPS:")
        print(f"1. Compare with multi-label results")
        print(f"2. Consider hybrid approach (semantic + ML)")  
        print(f"3. Deploy best performing model")
        print(f"4. Test on new company data")

if __name__ == "__main__":
    main()