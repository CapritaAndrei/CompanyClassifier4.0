"""
Weighted Similarity Calculator for Insurance Classification
Implements your approach: heavy weight on tags, TF-IDF on descriptions
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer


class SimilarityCalculator:
    """
    Multi-modal similarity calculator with weighted approach
    Your strategy: Tags matter much more than descriptions
    """
    
    def __init__(self, model: SentenceTransformer, labels: List[str]):
        """
        Initialize similarity calculator
        
        Args:
            model: Sentence transformer model
            labels: List of insurance labels
        """
        self.model = model
        self.labels = labels
        
        # Weights for different components (your approach)
        self.weights = {
            'tag_embedding': 0.4,      # Heavy weight on tag semantic similarity
            'tag_tfidf': 0.3,          # Heavy weight on tag word matches  
            'desc_embedding': 0.2,     # Moderate weight on description semantics
            'desc_tfidf': 0.1          # Light weight on description word matches
        }
        
        # Pre-compute label components
        self._precompute_label_features()
        
    def _precompute_label_features(self):
        """Pre-compute embeddings and TF-IDF features for all labels"""
        print("🔧 Pre-computing label features for weighted similarity...")
        
        # Embeddings for semantic similarity
        self.label_embeddings = self.model.encode(self.labels, convert_to_tensor=True)
        
        # TF-IDF vectorizers for word-level similarity
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams for better matching
            max_features=1000,
            lowercase=True
        )
        
        # Fit TF-IDF on all labels
        self.label_tfidf_vectors = self.tfidf_vectorizer.fit_transform(self.labels)
        
        print(f"✅ Ready! Features computed for {len(self.labels)} labels")
        
    def calculate_weighted_similarity(self, company_data: Dict) -> List[Tuple[str, float, Dict]]:
        """
        Calculate weighted similarity using your multi-modal approach
        
        Args:
            company_data: Company information dictionary
            
        Returns:
            List of (label, total_score, score_breakdown) tuples
        """
        
        # Extract and prepare company features
        description = company_data.get('description', '')
        business_tags = self._extract_tags(company_data.get('business_tags', ''))
        tags_text = ' '.join(business_tags)
        
        # 1. TAG EMBEDDING SIMILARITY (highest weight)
        tag_embedding_scores = self._compute_embedding_similarity(tags_text)
        
        # 2. TAG TF-IDF SIMILARITY (high weight - word matches in tags)
        tag_tfidf_scores = self._compute_tfidf_similarity(tags_text)
        
        # 3. DESCRIPTION EMBEDDING SIMILARITY (moderate weight)
        desc_embedding_scores = self._compute_embedding_similarity(description)
        
        # 4. DESCRIPTION TF-IDF SIMILARITY (lowest weight)
        desc_tfidf_scores = self._compute_tfidf_similarity(description)
        
        # Combine all scores with weights
        final_scores = []
        
        for i, label in enumerate(self.labels):
            # Weighted combination
            total_score = (
                self.weights['tag_embedding'] * tag_embedding_scores[i] +
                self.weights['tag_tfidf'] * tag_tfidf_scores[i] +
                self.weights['desc_embedding'] * desc_embedding_scores[i] +
                self.weights['desc_tfidf'] * desc_tfidf_scores[i]
            )
            
            # Score breakdown for transparency
            breakdown = {
                'tag_embedding': tag_embedding_scores[i],
                'tag_tfidf': tag_tfidf_scores[i], 
                'desc_embedding': desc_embedding_scores[i],
                'desc_tfidf': desc_tfidf_scores[i],
                'weights_used': self.weights
            }
            
            final_scores.append((label, total_score, breakdown))
        
        # Sort by total score
        final_scores.sort(key=lambda x: x[1], reverse=True)
        
        return final_scores
    
    def _extract_tags(self, tags_raw) -> List[str]:
        """Extract and clean business tags"""
        if not tags_raw:
            return []
            
        try:
            if isinstance(tags_raw, str):
                # Try to parse string representation of list
                if tags_raw.startswith('['):
                    tags = eval(tags_raw)
                else:
                    tags = [tags_raw]
            else:
                tags = tags_raw
                
            # Clean and filter tags
            if isinstance(tags, list):
                return [tag.strip() for tag in tags if tag.strip()]
            else:
                return [str(tags).strip()]
                
        except Exception:
            return [str(tags_raw).strip()] if tags_raw else []
    
    def _compute_embedding_similarity(self, text: str) -> np.ndarray:
        """Compute semantic similarity using embeddings"""
        if not text.strip():
            return np.zeros(len(self.labels))
            
        # Get text embedding
        text_embedding = self.model.encode(text, convert_to_tensor=True)
        
        # Calculate similarities with all labels
        similarities = cosine_similarity(
            text_embedding.cpu().numpy().reshape(1, -1),
            self.label_embeddings.cpu().numpy()
        )[0]
        
        return similarities
    
    def _compute_tfidf_similarity(self, text: str) -> np.ndarray:
        """Compute word-level similarity using TF-IDF"""
        if not text.strip():
            return np.zeros(len(self.labels))
            
        try:
            # Transform text to TF-IDF vector
            text_tfidf = self.tfidf_vectorizer.transform([text])
            
            # Calculate similarities with all label vectors
            similarities = cosine_similarity(text_tfidf, self.label_tfidf_vectors)[0]
            
            return similarities
            
        except Exception:
            return np.zeros(len(self.labels))
    
    def get_top_suggestions(self, company_data: Dict, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get top-k suggestions using weighted similarity
        
        Args:
            company_data: Company information
            top_k: Number of suggestions to return
            
        Returns:
            List of (label, score) tuples
        """
        scored_labels = self.calculate_weighted_similarity(company_data)
        
        # Return top-k with just label and score
        return [(label, score) for label, score, _ in scored_labels[:top_k]]
    
    def explain_similarity(self, company_data: Dict, top_k: int = 3) -> Dict:
        """
        Explain how similarity scores were calculated
        
        Args:
            company_data: Company information
            top_k: Number of top results to explain
            
        Returns:
            Detailed explanation of similarity calculation
        """
        scored_labels = self.calculate_weighted_similarity(company_data)
        
        # Extract company features for explanation
        description = company_data.get('description', '')
        business_tags = self._extract_tags(company_data.get('business_tags', ''))
        
        explanation = {
            'company_features': {
                'description': description[:200] + '...' if len(description) > 200 else description,
                'business_tags': business_tags,
                'tags_text': ' '.join(business_tags)
            },
            'methodology': {
                'approach': 'Weighted multi-modal similarity',
                'weights': self.weights,
                'components': [
                    'Tag Embedding (40%): Semantic similarity of business tags',
                    'Tag TF-IDF (30%): Word matches in business tags',  
                    'Description Embedding (20%): Semantic similarity of description',
                    'Description TF-IDF (10%): Word matches in description'
                ]
            },
            'top_matches': []
        }
        
        for label, total_score, breakdown in scored_labels[:top_k]:
            match_info = {
                'label': label,
                'total_score': total_score,
                'breakdown': breakdown,
                'interpretation': self._interpret_scores(breakdown)
            }
            explanation['top_matches'].append(match_info)
        
        return explanation
    
    def _interpret_scores(self, breakdown: Dict) -> str:
        """Generate human-readable interpretation of score breakdown"""
        interpretations = []
        
        # Check which component contributed most
        scores = {k: v for k, v in breakdown.items() if k != 'weights_used'}
        max_component = max(scores.keys(), key=lambda k: scores[k])
        
        if max_component == 'tag_embedding' and scores[max_component] > 0.5:
            interpretations.append("Strong semantic match with business tags")
        elif max_component == 'tag_tfidf' and scores[max_component] > 0.3:
            interpretations.append("Direct word matches in business tags")
        elif max_component == 'desc_embedding' and scores[max_component] > 0.4:
            interpretations.append("Good semantic match with description")
        elif max_component == 'desc_tfidf' and scores[max_component] > 0.2:
            interpretations.append("Word matches found in description")
        
        # Check for balanced scoring
        high_scores = [k for k, v in scores.items() if v > 0.3]
        if len(high_scores) > 1:
            interpretations.append("Multiple strong signals")
        
        return '; '.join(interpretations) if interpretations else "Low overall similarity" 