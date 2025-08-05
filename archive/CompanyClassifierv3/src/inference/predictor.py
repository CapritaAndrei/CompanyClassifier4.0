"""
Model Predictor for Insurance Classification
Loads trained models and provides prediction interface
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json

try:
    from sentence_transformers import SentenceTransformer
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False


class InsuranceModelPredictor:
    """
    Predictor class for trained insurance classification models
    """
    
    def __init__(self, model_path: str):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: Path to saved model pickle file
        """
        self.model_path = model_path
        self.model_package = None
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_extractors = None
        self.multilabel_binarizer = None
        self.requires_scaling = False
        self.training_info = None
        
        # Load sentence transformer if available and needed
        self.sentence_model = None
        
        # Load model
        self._load_model()
        
    def _load_model(self):
        """Load the trained model and all components"""
        print(f"📦 Loading model from {self.model_path}...")
        
        with open(self.model_path, 'rb') as f:
            self.model_package = pickle.load(f)
            
        # Extract components
        self.model = self.model_package['model']
        self.scaler = self.model_package.get('scaler')
        self.requires_scaling = self.model_package['requires_scaling']
        self.label_encoder = self.model_package['label_encoder']
        self.feature_extractors = self.model_package['feature_extractors']
        self.multilabel_binarizer = self.model_package['multilabel_binarizer']
        self.training_info = self.model_package['training_info']
        
        # Load sentence transformer if embeddings were used in training
        if 'embeddings' in self.training_info.get('features_used', []) and DEEP_LEARNING_AVAILABLE:
            print("🧠 Loading sentence transformer for embeddings...")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
        print(f"✅ Loaded {self.training_info['model_name']} model")
        print(f"   Training examples: {self.training_info['training_examples']}")
        print(f"   Unique labels: {self.training_info['unique_labels']}")
        print(f"   Features shape: {self.training_info['features_shape']}")
        
    def _extract_features(self, company_data: Dict) -> np.ndarray:
        """
        Extract features from company data using the same pipeline as training
        
        Args:
            company_data: Dictionary with company information
            
        Returns:
            Feature vector as numpy array
        """
        
        # 1. Description TF-IDF features
        description = str(company_data.get('description', ''))
        tfidf_desc_features = self.feature_extractors['tfidf_desc'].transform([description]).toarray()
        
        # 2. Business tags TF-IDF features
        business_tags = company_data.get('business_tags', [])
        if isinstance(business_tags, str):
            try:
                if business_tags.startswith('[') and business_tags.endswith(']'):
                    business_tags = eval(business_tags)
                else:
                    business_tags = [business_tags]
            except:
                business_tags = []
                
        tags_text = ' '.join(business_tags) if isinstance(business_tags, list) else str(business_tags)
        tfidf_tags_features = self.feature_extractors['tfidf_tags'].transform([tags_text]).toarray()
        
        # 3. Categorical features
        categorical_features = []
        for col in ['sector', 'category', 'niche']:
            if f'{col}_columns' in self.feature_extractors:
                col_value = company_data.get(col, 'Unknown')
                # Create one-hot encoding
                col_columns = self.feature_extractors[f'{col}_columns']
                col_encoded = np.zeros(len(col_columns))
                
                for i, column_name in enumerate(col_columns):
                    if column_name == f"{col}_{col_value}":
                        col_encoded[i] = 1
                        break
                        
                categorical_features.append(col_encoded)
                
        if categorical_features:
            categorical_matrix = np.hstack(categorical_features).reshape(1, -1)
        else:
            categorical_matrix = np.array([]).reshape(1, 0)
            
        # 4. Sentence embeddings (if available)
        if self.sentence_model is not None:
            embeddings = self.sentence_model.encode([description])
        else:
            embeddings = np.array([]).reshape(1, 0)
            
        # 5. Combine all features
        feature_matrices = [
            tfidf_desc_features,
            tfidf_tags_features,
            categorical_matrix
        ]
        
        if embeddings.shape[1] > 0:
            feature_matrices.append(embeddings)
            
        combined_features = np.hstack(feature_matrices)
        
        # 6. Handle feature dimension mismatch
        expected_features = self.training_info['features_shape'][1]
        current_features = combined_features.shape[1]
        
        if current_features < expected_features:
            # Pad with zeros if we have fewer features
            padding = np.zeros((1, expected_features - current_features))
            combined_features = np.hstack([combined_features, padding])
        elif current_features > expected_features:
            # Truncate if we have more features (shouldn't happen but just in case)
            combined_features = combined_features[:, :expected_features]
        
        return combined_features
        
    def predict(self, company_data: Dict, top_k: int = 5) -> Dict:
        """
        Predict insurance labels for a company
        
        Args:
            company_data: Dictionary with company information
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with predictions and probabilities
        """
        
        # Extract features
        features = self._extract_features(company_data)
        
        # Scale features if required
        if self.requires_scaling and self.scaler is not None:
            features = self.scaler.transform(features)
            
        # Make prediction
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        # Get top-k predictions
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        
        # Convert to label names
        predicted_label = self.label_encoder.inverse_transform([prediction])[0]
        
        top_predictions = []
        for idx in top_indices:
            label = self.label_encoder.inverse_transform([idx])[0]
            confidence = probabilities[idx]
            top_predictions.append({
                'label': label,
                'confidence': float(confidence)
            })
            
        return {
            'primary_prediction': predicted_label,
            'primary_confidence': float(probabilities[prediction]),
            'top_predictions': top_predictions,
            'model_info': {
                'model_name': self.training_info['model_name'],
                'training_timestamp': self.training_info['timestamp']
            }
        }
        
    def predict_batch(self, companies_data: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Predict insurance labels for multiple companies
        
        Args:
            companies_data: List of company data dictionaries
            top_k: Number of top predictions to return per company
            
        Returns:
            List of prediction dictionaries
        """
        
        results = []
        for i, company_data in enumerate(companies_data):
            print(f"🔄 Processing company {i+1}/{len(companies_data)}")
            prediction = self.predict(company_data, top_k=top_k)
            results.append(prediction)
            
        return results
        
    def evaluate_on_test_data(self, test_data_path: str) -> Dict:
        """
        Evaluate model performance on test data
        
        Args:
            test_data_path: Path to test data CSV
            
        Returns:
            Evaluation metrics
        """
        
        print(f"📊 Evaluating model on test data: {test_data_path}")
        
        # Load test data
        test_df = pd.read_csv(test_data_path)
        
        # Make predictions
        predictions = []
        true_labels = []
        
        for _, row in test_df.iterrows():
            company_data = {
                'description': row.get('description', ''),
                'business_tags': row.get('business_tags', []),
                'sector': row.get('sector', ''),
                'category': row.get('category', ''),
                'niche': row.get('niche', '')
            }
            
            prediction = self.predict(company_data, top_k=5)
            predictions.append(prediction)
            true_labels.append(row.get('primary_label', ''))
            
        # Calculate metrics
        correct_predictions = 0
        top_3_correct = 0
        top_5_correct = 0
        
        for pred, true_label in zip(predictions, true_labels):
            # Primary accuracy
            if pred['primary_prediction'] == true_label:
                correct_predictions += 1
                
            # Top-3 accuracy
            top_3_labels = [p['label'] for p in pred['top_predictions'][:3]]
            if true_label in top_3_labels:
                top_3_correct += 1
                
            # Top-5 accuracy
            top_5_labels = [p['label'] for p in pred['top_predictions'][:5]]
            if true_label in top_5_labels:
                top_5_correct += 1
                
        total = len(predictions)
        
        metrics = {
            'total_examples': total,
            'accuracy': correct_predictions / total,
            'top_3_accuracy': top_3_correct / total,
            'top_5_accuracy': top_5_correct / total,
            'model_info': {
                'model_name': self.training_info['model_name'],
                'training_timestamp': self.training_info['timestamp']
            }
        }
        
        print(f"📈 Evaluation Results:")
        print(f"   Accuracy: {metrics['accuracy']:.3f}")
        print(f"   Top-3 Accuracy: {metrics['top_3_accuracy']:.3f}")
        print(f"   Top-5 Accuracy: {metrics['top_5_accuracy']:.3f}")
        
        return metrics
        
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'model_name': self.training_info['model_name'],
            'training_timestamp': self.training_info['timestamp'],
            'training_examples': self.training_info['training_examples'],
            'unique_labels': self.training_info['unique_labels'],
            'features_shape': self.training_info['features_shape'],
            'requires_scaling': self.requires_scaling,
            'has_sentence_embeddings': self.sentence_model is not None
        }


def demo_prediction():
    """Demo function to show how to use the predictor"""
    
    # Example company data
    sample_company = {
        'description': 'A technology company that develops software solutions for healthcare providers, including electronic health records and patient management systems.',
        'business_tags': ['Software Development', 'Healthcare Technology', 'Electronic Health Records'],
        'sector': 'Services',
        'category': 'Software Development',
        'niche': 'Healthcare Software'
    }
    
    # List available models
    models_dir = Path('models/')
    if models_dir.exists():
        model_files = list(models_dir.glob('*.pkl'))
        if model_files:
            print("📁 Available models:")
            for i, model_file in enumerate(model_files):
                print(f"   {i+1}. {model_file.name}")
                
            # Use the first model for demo
            model_path = model_files[0]
            print(f"\n🔄 Using model: {model_path.name}")
            
            # Initialize predictor
            predictor = InsuranceModelPredictor(str(model_path))
            
            # Make prediction
            print(f"\n🏢 Sample Company:")
            print(f"   Description: {sample_company['description'][:100]}...")
            print(f"   Business Tags: {sample_company['business_tags']}")
            print(f"   Sector: {sample_company['sector']}")
            
            prediction = predictor.predict(sample_company, top_k=5)
            
            print(f"\n🎯 Predictions:")
            print(f"   Primary: {prediction['primary_prediction']} ({prediction['primary_confidence']:.3f})")
            print(f"   Top 5:")
            for i, pred in enumerate(prediction['top_predictions'], 1):
                print(f"      {i}. {pred['label']} ({pred['confidence']:.3f})")
                
        else:
            print("❌ No trained models found in models/ directory")
            print("   Run training_pipeline.py first to train models")
    else:
        print("❌ Models directory not found")
        print("   Run training_pipeline.py first to train models")


if __name__ == "__main__":
    demo_prediction() 