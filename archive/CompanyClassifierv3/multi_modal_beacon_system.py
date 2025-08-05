"""
Multi-Modal Hierarchical Insurance Classification System
Inspired by BEACON but adapted for multiple feature types
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
import json
import ast
from typing import Dict, List, Tuple, Any

class MultiModalInsuranceClassifier(BaseEstimator, ClassifierMixin):
    """
    Multi-Modal Hierarchical Insurance Classifier
    
    Architecture:
    1. Feature Extraction: Text + Categorical + Business Tags
    2. Stage 1: Predict Insurance Super-Categories (hierarchical grouping)
    3. Stage 2: Predict specific Insurance Labels within super-categories
    4. Ensemble: Combine multiple prediction strategies
    """
    
    def __init__(self, 
                 use_text=True,
                 use_categorical=True, 
                 use_business_tags=True,
                 knn_neighbors=5,
                 rf_n_estimators=100,
                 hierarchical_weight=0.6,
                 knn_weight=0.3,
                 rf_weight=0.1,
                 verbose=0):
        
        self.use_text = use_text
        self.use_categorical = use_categorical
        self.use_business_tags = use_business_tags
        self.knn_neighbors = knn_neighbors
        self.rf_n_estimators = rf_n_estimators
        self.hierarchical_weight = hierarchical_weight
        self.knn_weight = knn_weight
        self.rf_weight = rf_weight
        self.verbose = verbose
        
        # Initialize components
        self.text_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.tag_vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
        self.categorical_encoders = {}
        
        # Models for different stages
        self.super_category_model = RandomForestClassifier(n_estimators=rf_n_estimators)
        self.specific_models = {}  # One model per super-category
        self.knn_model = KNeighborsClassifier(n_neighbors=knn_neighbors)
        self.rf_fallback = RandomForestClassifier(n_estimators=rf_n_estimators)

    def _create_insurance_super_categories(self, insurance_labels: List[str]) -> Dict[str, List[str]]:
        """
        Create hierarchical groupings of insurance labels
        This would normally be done by domain experts, but we'll create logical groups
        """
        super_categories = {
            'Construction_Services': [
                'Pipeline Construction Services', 'Excavation Services', 'Tile Installation Services',
                'Septic System Services', 'Land Leveling Services', 'Water Treatment Services',
                'Boiler Installation Services', 'Tank Installation Services', 'Well Drilling Services',
                'Cable Installation Services'
            ],
            'Manufacturing': [
                'Chemical Manufacturing', 'Pharmaceutical Manufacturing', 'Food Processing Services',
                'Plastic Manufacturing', 'Rubber Manufacturing', 'Textile Manufacturing Services',
                'Printing Services', 'Canvas Manufacturing', 'Stationery Manufacturing'
            ],
            'Professional_Services': [
                'Consulting Services', 'Management Consulting', 'Strategic Planning Services',
                'Financial Services', 'Insurance Services', 'Real Estate Services', 'Legal Services'
            ],
            'Technical_Services': [
                'Software Manufacturing', 'Laboratory Services', 'Engineering Services',
                'Welding Services', 'Painting Services', 'Spray Painting Services'
            ],
            'Healthcare_Services': [
                'Veterinary Services', 'Veterinary Clinics', 'Medical Services',
                'Health and Wellness Services'
            ],
            'Agricultural_Services': [
                'Agricultural Equipment Services', 'Soil Nutrient Application Services',
                'Pesticide Application Services', 'Landscaping Services', 'Gardening Services',
                'Tree Services - Pruning / Removal'
            ],
            'Transportation_Services': [
                'Logistics Services', 'Delivery Services', 'Travel Services'
            ],
            'Creative_Services': [
                'Arts Services', 'Training Services', 'Publishing Services'
            ],
            'Food_Services': [
                'Catering Services', 'Restaurant Services', 'Food Services'
            ],
            'Other_Services': []  # Catch-all for unmatched labels
        }
        
        # Auto-assign unmatched labels to Other_Services
        assigned_labels = set()
        for category_labels in super_categories.values():
            assigned_labels.update(category_labels)
        
        for label in insurance_labels:
            if label not in assigned_labels:
                super_categories['Other_Services'].append(label)
                
        return super_categories

    def _extract_features(self, X: pd.DataFrame) -> np.ndarray:
        """Extract and combine features from all modalities"""
        features = []
        
        # Text features from descriptions
        if self.use_text and 'description' in X.columns:
            if hasattr(self, 'is_fitted_') and self.is_fitted_:
                text_features = self.text_vectorizer.transform(X['description']).toarray()
            else:
                text_features = self.text_vectorizer.fit_transform(X['description']).toarray()
            features.append(text_features)
            if self.verbose:
                print(f"📝 Text features: {text_features.shape}")
        
        # Categorical features (sector, category, niche)
        if self.use_categorical:
            categorical_cols = ['sector', 'category', 'niche']
            for col in categorical_cols:
                if col in X.columns:
                    if col not in self.categorical_encoders:
                        self.categorical_encoders[col] = LabelEncoder()
                        encoded = self.categorical_encoders[col].fit_transform(X[col].astype(str))
                    else:
                        # Handle unseen categories during prediction
                        encoded = []
                        for val in X[col].astype(str):
                            if val in self.categorical_encoders[col].classes_:
                                encoded.append(self.categorical_encoders[col].transform([val])[0])
                            else:
                                encoded.append(-1)  # Unknown category
                        encoded = np.array(encoded)
                    
                    # One-hot encode
                    n_classes = len(self.categorical_encoders[col].classes_)
                    one_hot = np.zeros((len(encoded), n_classes + 1))  # +1 for unknown
                    for i, val in enumerate(encoded):
                        if val >= 0:
                            one_hot[i, val] = 1
                        else:
                            one_hot[i, -1] = 1  # Unknown category
                    
                    features.append(one_hot)
                    if self.verbose:
                        print(f"🏷️ {col} features: {one_hot.shape}")
        
        # Business tags features
        if self.use_business_tags and 'business_tags' in X.columns:
            # Convert string representation of lists to actual lists
            tags_text = []
            for tags in X['business_tags']:
                if isinstance(tags, str):
                    try:
                        tags_list = ast.literal_eval(tags)
                        if isinstance(tags_list, list):
                            tags_text.append(' '.join(tags_list))
                        else:
                            tags_text.append(str(tags))
                    except:
                        tags_text.append(str(tags))
                else:
                    tags_text.append(str(tags))
            
            if hasattr(self, 'is_fitted_') and self.is_fitted_:
                tag_features = self.tag_vectorizer.transform(tags_text).toarray()
            else:
                tag_features = self.tag_vectorizer.fit_transform(tags_text).toarray()
            features.append(tag_features)
            if self.verbose:
                print(f"🏪 Business tag features: {tag_features.shape}")
        
        # Combine all features
        if features:
            combined_features = np.hstack(features)
            if self.verbose:
                print(f"🔗 Combined features: {combined_features.shape}")
            return combined_features
        else:
            raise ValueError("No features extracted. Check your feature configuration.")

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """
        Fit the multi-modal hierarchical classifier
        
        Parameters:
        X: DataFrame with columns: description, sector, category, niche, business_tags
        y: Array of insurance labels
        """
        if self.verbose:
            print("🚀 Training Multi-Modal Insurance Classifier")
            print("=" * 50)
        
        # Create super-categories mapping
        unique_labels = list(set(y))
        self.super_categories = self._create_insurance_super_categories(unique_labels)
        
        # Create reverse mapping: label -> super_category
        self.label_to_super = {}
        for super_cat, labels in self.super_categories.items():
            for label in labels:
                self.label_to_super[label] = super_cat
        
        # Extract features
        X_features = self._extract_features(X)
        
        # Create super-category labels
        y_super = [self.label_to_super.get(label, 'Other_Services') for label in y]
        
        # Stage 1: Train super-category classifier
        if self.verbose:
            print(f"📊 Stage 1: Training super-category classifier ({len(set(y_super))} categories)")
        self.super_category_model.fit(X_features, y_super)
        
        # Stage 2: Train specific classifiers for each super-category
        if self.verbose:
            print("📊 Stage 2: Training category-specific classifiers")
        
        for super_cat in set(y_super):
            # Get samples for this super-category
            mask = np.array(y_super) == super_cat
            if np.sum(mask) > 1:  # Need at least 2 samples
                X_cat = X_features[mask]
                y_cat = y[mask]
                
                # Train category-specific model
                model = RandomForestClassifier(n_estimators=self.rf_n_estimators)
                model.fit(X_cat, y_cat)
                self.specific_models[super_cat] = model
                
                if self.verbose:
                    print(f"  ✓ {super_cat}: {np.sum(mask)} samples, {len(set(y_cat))} labels")
        
        # Stage 3: Train fallback models (KNN and RF)
        if self.verbose:
            print("📊 Stage 3: Training fallback models")
        
        self.knn_model.fit(X_features, y)
        self.rf_fallback.fit(X_features, y)
        
        # Store for later use
        self.classes_ = unique_labels
        self.is_fitted_ = True
        
        if self.verbose:
            print("✅ Training completed!")
        
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities using hierarchical ensemble approach"""
        X_features = self._extract_features(X)
        n_samples = X_features.shape[0]
        n_classes = len(self.classes_)
        
        # Initialize probability matrix
        proba_matrix = np.zeros((n_samples, n_classes))
        class_to_idx = {cls: i for i, cls in enumerate(self.classes_)}
        
        # Stage 1: Predict super-categories
        super_proba = self.super_category_model.predict_proba(X_features)
        super_classes = self.super_category_model.classes_
        
        for i in range(n_samples):
            hierarchical_proba = np.zeros(n_classes)
            
            # Stage 2: For each super-category, get specific predictions
            for j, super_cat in enumerate(super_classes):
                super_cat_prob = super_proba[i, j]
                
                if super_cat in self.specific_models and super_cat_prob > 0:
                    # Get predictions from category-specific model
                    specific_model = self.specific_models[super_cat]
                    try:
                        sample_features = X_features[i:i+1]
                        specific_proba = specific_model.predict_proba(sample_features)[0]
                        specific_classes = specific_model.classes_
                        
                        # Combine super-category and specific probabilities
                        for k, specific_class in enumerate(specific_classes):
                            if specific_class in class_to_idx:
                                hierarchical_proba[class_to_idx[specific_class]] += (
                                    super_cat_prob * specific_proba[k]
                                )
                    except:
                        # If specific model fails, distribute probability equally
                        category_labels = self.super_categories.get(super_cat, [])
                        for label in category_labels:
                            if label in class_to_idx:
                                hierarchical_proba[class_to_idx[label]] += (
                                    super_cat_prob / len(category_labels)
                                )
            
            # Get KNN predictions
            knn_proba = np.zeros(n_classes)
            try:
                knn_pred = self.knn_model.predict_proba(X_features[i:i+1])[0]
                knn_classes = self.knn_model.classes_
                for j, cls in enumerate(knn_classes):
                    if cls in class_to_idx:
                        knn_proba[class_to_idx[cls]] = knn_pred[j]
            except:
                pass
            
            # Get Random Forest predictions
            rf_proba = np.zeros(n_classes)
            try:
                rf_pred = self.rf_fallback.predict_proba(X_features[i:i+1])[0]
                rf_classes = self.rf_fallback.classes_
                for j, cls in enumerate(rf_classes):
                    if cls in class_to_idx:
                        rf_proba[class_to_idx[cls]] = rf_pred[j]
            except:
                pass
            
            # Ensemble combination
            final_proba = (
                self.hierarchical_weight * hierarchical_proba +
                self.knn_weight * knn_proba +
                self.rf_weight * rf_proba
            )
            
            # Normalize
            if final_proba.sum() > 0:
                final_proba = final_proba / final_proba.sum()
            
            proba_matrix[i] = final_proba
        
        return proba_matrix

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict insurance labels"""
        proba = self.predict_proba(X)
        predictions = []
        
        for i in range(proba.shape[0]):
            best_idx = np.argmax(proba[i])
            predictions.append(self.classes_[best_idx])
        
        return np.array(predictions)

    def get_feature_importance(self) -> Dict[str, Any]:
        """Get feature importance from different components"""
        importance = {}
        
        if hasattr(self.super_category_model, 'feature_importances_'):
            importance['super_category_model'] = self.super_category_model.feature_importances_
        
        if hasattr(self.rf_fallback, 'feature_importances_'):
            importance['rf_fallback'] = self.rf_fallback.feature_importances_
        
        return importance

# Example usage and testing functions
def create_sample_training_data():
    """Create sample training data for testing"""
    data = {
        'description': [
            'Civil engineering construction company specializing in utility networks',
            'Software development and IT consulting services',
            'Veterinary clinic providing pet care services',
            'Manufacturing plastic components for automotive industry',
            'Restaurant catering services for events'
        ],
        'sector': ['Services', 'Services', 'Services', 'Manufacturing', 'Services'],
        'category': ['Civil Engineering', 'IT Services', 'Veterinary', 'Plastic Manufacturing', 'Food Services'],
        'niche': ['Construction', 'Software Development', 'Pet Care', 'Automotive Parts', 'Catering'],
        'business_tags': [
            "['Construction Services', 'Utility Network Connections']",
            "['Software Manufacturing', 'Consulting Services']", 
            "['Veterinary Services', 'Pet Care']",
            "['Plastic Manufacturing', 'Automotive']",
            "['Catering Services', 'Food Services']"
        ]
    }
    
    labels = [
        'Pipeline Construction Services',
        'Software Manufacturing', 
        'Veterinary Services',
        'Plastic Manufacturing',
        'Catering Services'
    ]
    
    return pd.DataFrame(data), np.array(labels)

def test_classifier():
    """Test the multi-modal classifier"""
    print("🧪 Testing Multi-Modal Insurance Classifier")
    print("=" * 50)
    
    # Create sample data
    X_train, y_train = create_sample_training_data()
    
    # Initialize and train classifier
    classifier = MultiModalInsuranceClassifier(verbose=1)
    classifier.fit(X_train, y_train)
    
    # Test predictions
    X_test = pd.DataFrame({
        'description': ['Construction company doing electrical work'],
        'sector': ['Services'],
        'category': ['Construction'],
        'niche': ['Electrical'],
        'business_tags': ["['Construction Services', 'Electrical Services']"]
    })
    
    predictions = classifier.predict(X_test)
    probabilities = classifier.predict_proba(X_test)
    
    print(f"\n🎯 Prediction: {predictions[0]}")
    print(f"📊 Top probabilities:")
    
    # Show top 3 predictions
    top_indices = np.argsort(probabilities[0])[-3:][::-1]
    for idx in top_indices:
        print(f"  {classifier.classes_[idx]}: {probabilities[0][idx]:.4f}")

if __name__ == "__main__":
    test_classifier() 