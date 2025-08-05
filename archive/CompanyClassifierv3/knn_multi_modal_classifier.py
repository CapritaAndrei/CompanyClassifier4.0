"""
KNN-Based Multi-Modal Insurance Classifier
Uses similarity matching across text, categorical, and business tag features
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import ast
from typing import Dict, List, Tuple, Any
from collections import Counter

class KNNMultiModalClassifier(BaseEstimator, ClassifierMixin):
    """
    KNN-based Multi-Modal Classifier using weighted similarity across modalities
    
    Architecture:
    1. Separate feature spaces for text, categorical, and business tags
    2. Calculate similarity in each feature space
    3. Weighted combination of similarities
    4. KNN voting with distance weighting
    """
    
    def __init__(self, 
                 k_neighbors=5,
                 text_weight=0.4,
                 categorical_weight=0.3,
                 tags_weight=0.3,
                 distance_weighting=True,
                 verbose=0):
        
        self.k_neighbors = k_neighbors
        self.text_weight = text_weight
        self.categorical_weight = categorical_weight
        self.tags_weight = tags_weight
        self.distance_weighting = distance_weighting
        self.verbose = verbose
        
        # Feature extractors
        self.text_vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
        self.tag_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.categorical_encoders = {}
        self.categorical_scaler = StandardScaler()
        
        # Storage for training data
        self.X_train_text_ = None
        self.X_train_categorical_ = None
        self.X_train_tags_ = None
        self.y_train_ = None
        self.classes_ = None

    def _extract_text_features(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Extract TF-IDF features from descriptions"""
        if 'description' not in X.columns:
            return np.empty((len(X), 0))
        
        if fit:
            features = self.text_vectorizer.fit_transform(X['description']).toarray()
        else:
            features = self.text_vectorizer.transform(X['description']).toarray()
        
        if self.verbose:
            print(f"📝 Text features: {features.shape}")
        return features

    def _extract_categorical_features(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Extract and encode categorical features"""
        categorical_cols = ['sector', 'category', 'niche']
        features = []
        
        for col in categorical_cols:
            if col in X.columns:
                if fit:
                    if col not in self.categorical_encoders:
                        self.categorical_encoders[col] = LabelEncoder()
                    encoded = self.categorical_encoders[col].fit_transform(X[col].astype(str))
                else:
                    if col in self.categorical_encoders:
                        # Handle unseen categories
                        encoded = []
                        for val in X[col].astype(str):
                            if val in self.categorical_encoders[col].classes_:
                                encoded.append(self.categorical_encoders[col].transform([val])[0])
                            else:
                                encoded.append(-1)  # Unknown category
                        encoded = np.array(encoded)
                    else:
                        encoded = np.array([-1] * len(X))
                
                features.append(encoded.reshape(-1, 1))
        
        if features:
            categorical_features = np.hstack(features)
            
            if fit:
                categorical_features = self.categorical_scaler.fit_transform(categorical_features)
            else:
                categorical_features = self.categorical_scaler.transform(categorical_features)
            
            if self.verbose:
                print(f"🏷️ Categorical features: {categorical_features.shape}")
            return categorical_features
        else:
            return np.empty((len(X), 0))

    def _extract_tag_features(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Extract TF-IDF features from business tags"""
        if 'business_tags' not in X.columns:
            return np.empty((len(X), 0))
        
        # Convert string representation of lists to text
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
        
        if fit:
            features = self.tag_vectorizer.fit_transform(tags_text).toarray()
        else:
            features = self.tag_vectorizer.transform(tags_text).toarray()
        
        if self.verbose:
            print(f"🏪 Tag features: {features.shape}")
        return features

    def _calculate_multi_modal_similarity(self, X_query: np.ndarray, X_train: np.ndarray, 
                                        query_text: np.ndarray, train_text: np.ndarray,
                                        query_categorical: np.ndarray, train_categorical: np.ndarray,
                                        query_tags: np.ndarray, train_tags: np.ndarray) -> np.ndarray:
        """Calculate weighted similarity across all modalities"""
        n_query = query_text.shape[0] if query_text.shape[0] > 0 else query_categorical.shape[0] if query_categorical.shape[0] > 0 else query_tags.shape[0]
        n_train = train_text.shape[0] if train_text.shape[0] > 0 else train_categorical.shape[0] if train_categorical.shape[0] > 0 else train_tags.shape[0]
        similarities = np.zeros((n_query, n_train))
        
        for i in range(n_query):
            # Initialize combined similarity array
            combined_sim = np.zeros(n_train)
            total_weight = 0
            
            # Text similarity
            if query_text.shape[1] > 0 and train_text.shape[1] > 0:
                text_sim = cosine_similarity(query_text[i:i+1], train_text).flatten()
                combined_sim += self.text_weight * text_sim
                total_weight += self.text_weight
            
            # Categorical similarity (using inverse distance)
            if query_categorical.shape[1] > 0 and train_categorical.shape[1] > 0:
                distances = euclidean_distances(query_categorical[i:i+1], train_categorical).flatten()
                # Convert distances to similarities (higher distance = lower similarity)
                max_dist = distances.max() if distances.max() > 0 else 1
                categorical_sim = 1 - (distances / max_dist)
                combined_sim += self.categorical_weight * categorical_sim
                total_weight += self.categorical_weight
            
            # Tag similarity
            if query_tags.shape[1] > 0 and train_tags.shape[1] > 0:
                tag_sim = cosine_similarity(query_tags[i:i+1], train_tags).flatten()
                combined_sim += self.tags_weight * tag_sim
                total_weight += self.tags_weight
            
            # Normalize by total weight
            if total_weight > 0:
                combined_sim = combined_sim / total_weight
            
            similarities[i] = combined_sim
        
        return similarities

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """Fit the KNN multi-modal classifier"""
        if self.verbose:
            print("🚀 Training KNN Multi-Modal Classifier")
            print("=" * 50)
        
        # Extract features for each modality
        self.X_train_text_ = self._extract_text_features(X, fit=True)
        self.X_train_categorical_ = self._extract_categorical_features(X, fit=True)
        self.X_train_tags_ = self._extract_tag_features(X, fit=True)
        
        # Store training labels
        self.y_train_ = np.array(y)
        self.classes_ = np.unique(y)
        
        if self.verbose:
            print(f"✅ Training completed with {len(self.y_train_)} samples")
            print(f"📊 Classes: {len(self.classes_)}")
        
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities using KNN voting"""
        # Extract features for query samples
        X_query_text = self._extract_text_features(X, fit=False)
        X_query_categorical = self._extract_categorical_features(X, fit=False)
        X_query_tags = self._extract_tag_features(X, fit=False)
        
        # Calculate similarities
        similarities = self._calculate_multi_modal_similarity(
            X, X,  # Dummy parameters (not used in current implementation)
            X_query_text, self.X_train_text_,
            X_query_categorical, self.X_train_categorical_,
            X_query_tags, self.X_train_tags_
        )
        
        n_samples = len(X)
        n_classes = len(self.classes_)
        probabilities = np.zeros((n_samples, n_classes))
        class_to_idx = {cls: i for i, cls in enumerate(self.classes_)}
        
        for i in range(n_samples):
            # Get k most similar neighbors
            neighbor_indices = np.argsort(similarities[i])[-self.k_neighbors:]
            neighbor_similarities = similarities[i][neighbor_indices]
            neighbor_labels = self.y_train_[neighbor_indices]
            
            # Calculate weighted votes
            if self.distance_weighting:
                # Use similarity as weights (higher similarity = higher weight)
                weights = neighbor_similarities
                # Avoid division by zero
                weights = np.maximum(weights, 1e-10)
            else:
                # Equal weights
                weights = np.ones(len(neighbor_labels))
            
            # Aggregate votes
            class_votes = np.zeros(n_classes)
            for label, weight in zip(neighbor_labels, weights):
                if label in class_to_idx:
                    class_votes[class_to_idx[label]] += weight
            
            # Normalize to probabilities
            if class_votes.sum() > 0:
                class_votes = class_votes / class_votes.sum()
            
            probabilities[i] = class_votes
        
        return probabilities

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict classes using KNN voting"""
        probabilities = self.predict_proba(X)
        predictions = []
        
        for i in range(len(probabilities)):
            best_class_idx = np.argmax(probabilities[i])
            predictions.append(self.classes_[best_class_idx])
        
        return np.array(predictions)

    def get_nearest_neighbors(self, X: pd.DataFrame, n_neighbors: int = None) -> List[List[Tuple[int, float]]]:
        """Get nearest neighbors for each query sample"""
        if n_neighbors is None:
            n_neighbors = self.k_neighbors
        
        # Extract features
        X_query_text = self._extract_text_features(X, fit=False)
        X_query_categorical = self._extract_categorical_features(X, fit=False)
        X_query_tags = self._extract_tag_features(X, fit=False)
        
        # Calculate similarities
        similarities = self._calculate_multi_modal_similarity(
            X, X,
            X_query_text, self.X_train_text_,
            X_query_categorical, self.X_train_categorical_,
            X_query_tags, self.X_train_tags_
        )
        
        neighbors = []
        for i in range(len(X)):
            # Get most similar neighbors
            neighbor_indices = np.argsort(similarities[i])[-n_neighbors:][::-1]
            neighbor_similarities = similarities[i][neighbor_indices]
            
            sample_neighbors = []
            for idx, sim in zip(neighbor_indices, neighbor_similarities):
                sample_neighbors.append((idx, sim))
            
            neighbors.append(sample_neighbors)
        
        return neighbors

# Test function
def test_knn_classifier():
    """Test the KNN multi-modal classifier"""
    print("🧪 Testing KNN Multi-Modal Classifier")
    print("=" * 50)
    
    # Create sample data
    data = {
        'description': [
            'Civil engineering construction company specializing in utility networks',
            'Software development and IT consulting services',
            'Veterinary clinic providing pet care services',
            'Manufacturing plastic components for automotive industry',
            'Restaurant catering services for events',
            'Construction company specializing in electrical installations'
        ],
        'sector': ['Services', 'Services', 'Services', 'Manufacturing', 'Services', 'Services'],
        'category': ['Civil Engineering', 'IT Services', 'Veterinary', 'Plastic Manufacturing', 'Food Services', 'Construction'],
        'niche': ['Construction', 'Software Development', 'Pet Care', 'Automotive Parts', 'Catering', 'Electrical'],
        'business_tags': [
            "['Construction Services', 'Utility Network Connections']",
            "['Software Manufacturing', 'Consulting Services']", 
            "['Veterinary Services', 'Pet Care']",
            "['Plastic Manufacturing', 'Automotive']",
            "['Catering Services', 'Food Services']",
            "['Construction Services', 'Electrical Services']"
        ]
    }
    
    labels = [
        'Pipeline Construction Services',
        'Software Manufacturing', 
        'Veterinary Services',
        'Plastic Manufacturing',
        'Catering Services',
        'Cable Installation Services'
    ]
    
    X_train = pd.DataFrame(data)
    y_train = np.array(labels)
    
    # Train classifier
    classifier = KNNMultiModalClassifier(k_neighbors=3, verbose=1)
    classifier.fit(X_train, y_train)
    
    # Test prediction
    X_test = pd.DataFrame({
        'description': ['Electrical contractor installing power lines'],
        'sector': ['Services'],
        'category': ['Construction'],
        'niche': ['Electrical'],
        'business_tags': ["['Construction Services', 'Electrical Installation']"]
    })
    
    predictions = classifier.predict(X_test)
    probabilities = classifier.predict_proba(X_test)
    neighbors = classifier.get_nearest_neighbors(X_test, n_neighbors=3)
    
    print(f"\n🎯 Prediction: {predictions[0]}")
    print(f"📊 Top probabilities:")
    
    # Show top 3 predictions
    top_indices = np.argsort(probabilities[0])[-3:][::-1]
    for idx in top_indices:
        print(f"  {classifier.classes_[idx]}: {probabilities[0][idx]:.4f}")
    
    print(f"\n🔍 Nearest neighbors:")
    for i, (neighbor_idx, similarity) in enumerate(neighbors[0]):
        print(f"  {i+1}. Training sample {neighbor_idx}: {similarity:.4f}")
        print(f"     Label: {y_train[neighbor_idx]}")
        print(f"     Description: {X_train.iloc[neighbor_idx]['description'][:60]}...")

if __name__ == "__main__":
    test_knn_classifier() 