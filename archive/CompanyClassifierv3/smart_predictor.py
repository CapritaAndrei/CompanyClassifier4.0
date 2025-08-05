#!/usr/bin/env python3
"""
Smart Insurance Predictor with Domain Awareness
Makes intelligent predictions using semantic understanding
"""

import numpy as np
import pickle
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


class SmartInsurancePredictor:
    """
    Intelligent predictor that:
    - Uses semantic similarity to boost related labels
    - Penalizes completely unrelated domains
    - Handles short descriptions better
    - Provides explainable predictions
    """
    
    def __init__(self, model_path: str = "models/smart_insurance_model.pkl"):
        """Load the smart model and components"""
        print("🧠 Loading smart model...")
        
        with open(model_path, 'rb') as f:
            self.model_package = pickle.load(f)
        
        self.model = self.model_package['model']
        self.label_encoder = self.model_package['label_encoder']
        self.feature_extractors = self.model_package['feature_extractors']
        self.label_embeddings = self.model_package['label_embeddings']
        self.domain_groups = self.model_package['domain_groups']
        
        # Get the classes the model was actually trained on
        if hasattr(self.model, 'classes_'):
            self.model_classes = self.model.classes_
        else:
            # For models wrapped in dict (like LR with scaler)
            if isinstance(self.model, dict) and 'model' in self.model:
                self.model_classes = self.model['model'].classes_
            else:
                # Assume all classes if not available
                self.model_classes = np.arange(len(self.label_encoder.classes_))
        
        # Load sentence transformer
        print("🧠 Loading semantic model...")
        self.sentence_model = SentenceTransformer(self.model_package['sentence_model_name'])
        
        # Create reverse domain mapping
        self.label_to_domain = {}
        for domain, labels in self.domain_groups.items():
            for label in labels:
                self.label_to_domain[label] = domain
        
        print("✅ Smart predictor ready!")
    
    def predict(self, company_data: Dict, top_k: int = 5) -> Dict:
        """Make smart predictions with domain awareness"""
        
        # Extract features
        features = self._extract_features(company_data)
        
        # Get base predictions from model
        if hasattr(self.model, 'predict_proba'):
            model_proba = self.model.predict_proba(features.reshape(1, -1))[0]
        else:
            # For LR with scaler
            scaled_features = self.model['scaler'].transform(features.reshape(1, -1))
            model_proba = self.model['model'].predict_proba(scaled_features)[0]
        
        # Map model predictions to full label set
        n_encoder_classes = len(self.label_encoder.classes_)
        base_proba = np.zeros(n_encoder_classes)
        
        # Fill in probabilities for classes the model knows about
        for i, class_idx in enumerate(self.model_classes):
            if class_idx < n_encoder_classes:
                base_proba[class_idx] = model_proba[i]
        
        # Apply smart adjustments
        adjusted_proba = self._apply_domain_intelligence(company_data, base_proba)
        
        # Get top predictions
        top_indices = np.argsort(adjusted_proba)[-top_k:][::-1]
        
        # Decode labels
        all_labels = self.label_encoder.classes_
        
        predictions = []
        for idx in top_indices:
            label = all_labels[idx]
            score = adjusted_proba[idx]
            domain = self.label_to_domain.get(label, 'general')
            
            predictions.append({
                'label': label,
                'confidence': float(score),
                'domain': domain
            })
        
        # Primary prediction with explanation
        primary_label = predictions[0]['label']
        primary_confidence = predictions[0]['confidence']
        
        # Generate explanation
        explanation = self._generate_explanation(company_data, predictions[0])
        
        return {
            'primary_prediction': primary_label,
            'primary_confidence': primary_confidence,
            'top_predictions': predictions,
            'explanation': explanation,
            'detected_domain': self._detect_company_domain(company_data)
        }
    
    def _extract_features(self, company_data: Dict) -> np.ndarray:
        """Extract features matching training pipeline"""
        
        description = company_data.get('description', '')
        
        # Handle empty/short descriptions
        if not description or len(description) < 20:
            # Augment with business tags
            tags = company_data.get('business_tags', [])
            if tags:
                description = f"{description} {' '.join(tags)}"
        
        # 1. TF-IDF features for description
        tfidf_desc = self.feature_extractors['tfidf_desc'].transform([description])
        
        # 2. Business tags features
        tags = company_data.get('business_tags', [])
        if isinstance(tags, list):
            tags_text = ' '.join(tags)
        else:
            tags_text = str(tags)
        
        tfidf_tags = self.feature_extractors['tfidf_tags'].transform([tags_text])
        
        # 3. Semantic embeddings
        embeddings = self.sentence_model.encode([description])
        
        # 4. Domain features
        domain_features = self._create_domain_features([description])
        
        # Combine all features
        features = np.hstack([
            tfidf_desc.toarray(),
            tfidf_tags.toarray(),
            embeddings,
            domain_features
        ])
        
        return features[0]
    
    def _create_domain_features(self, descriptions):
        """Create domain-specific signal features"""
        domain_signals = {
            'tech_signals': ['API', 'cloud', 'SaaS', 'software', 'platform', 'digital', 'online'],
            'manufacturing_signals': ['production', 'assembly', 'facility', 'plant', 'industrial'],
            'service_signals': ['consulting', 'services', 'solutions', 'support', 'management'],
            'retail_signals': ['store', 'shop', 'retail', 'sales', 'customer', 'products']
        }
        
        domain_features = []
        for desc in descriptions:
            desc_lower = desc.lower()
            features = []
            for signal_type, keywords in domain_signals.items():
                count = sum(1 for keyword in keywords if keyword in desc_lower)
                features.append(count)
            domain_features.append(features)
            
        return np.array(domain_features)
    
    def _detect_company_domain(self, company_data: Dict) -> str:
        """Detect the primary domain of the company"""
        
        description = company_data.get('description', '').lower()
        tags = ' '.join(company_data.get('business_tags', [])).lower()
        combined_text = f"{description} {tags}"
        
        # Domain detection rules
        domain_rules = {
            'software_tech': ['software', 'app', 'web', 'digital', 'technology', 'IT', 'platform'],
            'food_restaurant': ['restaurant', 'food', 'dining', 'cuisine', 'cafe', 'catering'],
            'manufacturing': ['manufacturing', 'production', 'factory', 'assembly', 'industrial'],
            'financial': ['financial', 'banking', 'investment', 'insurance', 'credit'],
            'healthcare': ['health', 'medical', 'hospital', 'clinic', 'care', 'therapy'],
            'retail': ['retail', 'store', 'shop', 'merchant', 'sales'],
            'automotive': ['automotive', 'car', 'vehicle', 'auto parts', 'motor']
        }
        
        domain_scores = {}
        for domain, keywords in domain_rules.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        return 'general'
    
    def _apply_domain_intelligence(self, company_data: Dict, base_proba: np.ndarray) -> np.ndarray:
        """Apply smart adjustments based on domain understanding"""
        
        adjusted_proba = base_proba.copy()
        
        # Detect company's domain
        company_domain = self._detect_company_domain(company_data)
        
        # Get company embedding
        description = company_data.get('description', '')
        if not description:
            description = ' '.join(company_data.get('business_tags', []))
        
        company_embedding = self.sentence_model.encode([description])[0]
        
        # Adjust probabilities based on domain matching
        all_labels = self.label_encoder.classes_
        
        for i, label in enumerate(all_labels):
            if label in self.label_to_domain:
                label_domain = self.label_to_domain[label]
                
                # Boost if domains match
                if label_domain == company_domain:
                    adjusted_proba[i] *= 1.5  # 50% boost
                    
                # Penalize if domains are completely different
                elif self._are_domains_incompatible(company_domain, label_domain):
                    adjusted_proba[i] *= 0.3  # 70% penalty
            
            # Additional semantic similarity boost
            if label in self.label_embeddings:
                label_embedding = self.label_embeddings[label]
                similarity = np.dot(company_embedding, label_embedding) / (
                    np.linalg.norm(company_embedding) * np.linalg.norm(label_embedding)
                )
                
                # Boost based on semantic similarity
                if similarity > 0.7:
                    adjusted_proba[i] *= (1 + similarity)
        
        # Handle NaN labels - heavily penalize
        for i, label in enumerate(all_labels):
            if pd.isna(label) or label == 'nan':
                adjusted_proba[i] *= 0.01  # 99% penalty
        
        # Normalize probabilities
        adjusted_proba = adjusted_proba / adjusted_proba.sum()
        
        return adjusted_proba
    
    def _are_domains_incompatible(self, domain1: str, domain2: str) -> bool:
        """Check if two domains are incompatible"""
        
        incompatible_pairs = [
            ('software_tech', 'food_restaurant'),
            ('software_tech', 'manufacturing'),
            ('food_restaurant', 'financial'),
            ('food_restaurant', 'automotive'),
            ('healthcare', 'automotive'),
            ('retail', 'manufacturing')
        ]
        
        for d1, d2 in incompatible_pairs:
            if (domain1 == d1 and domain2 == d2) or (domain1 == d2 and domain2 == d1):
                return True
        return False
    
    def _generate_explanation(self, company_data: Dict, prediction: Dict) -> str:
        """Generate human-readable explanation for the prediction"""
        
        label = prediction['label']
        domain = prediction['domain']
        confidence = prediction['confidence']
        
        company_domain = self._detect_company_domain(company_data)
        
        explanation_parts = []
        
        # Domain match explanation
        if domain == company_domain:
            explanation_parts.append(f"Strong domain match: both are {domain}")
        elif self._are_domains_incompatible(company_domain, domain):
            explanation_parts.append(f"Warning: domain mismatch ({company_domain} vs {domain})")
        
        # Confidence explanation
        if confidence > 0.7:
            explanation_parts.append("High confidence prediction")
        elif confidence > 0.3:
            explanation_parts.append("Moderate confidence - consider top 3 predictions")
        else:
            explanation_parts.append("Low confidence - review manually")
        
        # Keywords that influenced prediction
        description = company_data.get('description', '').lower()
        label_keywords = label.lower().split()
        matching_keywords = [kw for kw in label_keywords if kw in description]
        
        if matching_keywords:
            explanation_parts.append(f"Matching keywords: {', '.join(matching_keywords)}")
        
        return " | ".join(explanation_parts)
    
    def batch_predict(self, companies: List[Dict], show_progress: bool = True) -> List[Dict]:
        """Predict for multiple companies"""
        results = []
        
        for i, company in enumerate(companies):
            if show_progress and i % 10 == 0:
                print(f"Processing company {i+1}/{len(companies)}...")
            
            prediction = self.predict(company)
            results.append({
                'company_index': i,
                'prediction': prediction['primary_prediction'],
                'confidence': prediction['primary_confidence'],
                'domain': prediction['detected_domain'],
                'top_3': [p['label'] for p in prediction['top_predictions'][:3]]
            })
        
        return results


def test_smart_predictor():
    """Test the smart predictor with examples"""
    
    # Check if smart model exists
    from pathlib import Path
    if not Path('models/smart_insurance_model.pkl').exists():
        print("⚠️ Smart model not found. Train it first with: python3 smart_training_pipeline.py")
        return
    
    predictor = SmartInsurancePredictor()
    
    test_cases = [
        {
            'name': 'Software Company',
            'description': 'Software development company specializing in web applications',
            'business_tags': ['Software Development', 'Web Development'],
            'sector': 'Services',
            'category': 'Technology'
        },
        {
            'name': 'Restaurant',
            'description': 'Italian restaurant serving authentic cuisine in downtown area',
            'business_tags': ['Restaurant', 'Food Service', 'Italian'],
            'sector': 'Services',
            'category': 'Food Service'
        },
        {
            'name': 'Auto Parts Manufacturer',
            'description': 'Manufacturing facility producing brake systems and engine components',
            'business_tags': ['Manufacturing', 'Automotive Parts'],
            'sector': 'Manufacturing',
            'category': 'Automotive'
        }
    ]
    
    print("🧪 Testing Smart Predictor")
    print("=" * 60)
    
    for company in test_cases:
        print(f"\n📊 {company['name']}:")
        result = predictor.predict(company)
        
        print(f"  Primary: {result['primary_prediction']}")
        print(f"  Confidence: {result['primary_confidence']:.3f}")
        print(f"  Domain: {result['detected_domain']}")
        print(f"  Explanation: {result['explanation']}")
        
        print("  Top 3:")
        for i, pred in enumerate(result['top_predictions'][:3], 1):
            print(f"    {i}. {pred['label']} ({pred['confidence']:.3f}) - {pred['domain']}")


if __name__ == "__main__":
    test_smart_predictor() 