"""
Weighted Insurance Classifier with Few-Shot Learning
Main classifier implementing your weighted approach
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from sentence_transformers import SentenceTransformer

from .similarity import SimilarityCalculator


class WeightedInsuranceClassifier:
    """
    Main insurance classifier with weighted similarity and few-shot learning
    Implements your approach: heavy emphasis on business tags
    """
    
    def __init__(self, taxonomy_path: str, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize weighted classifier
        
        Args:
            taxonomy_path: Path to insurance taxonomy CSV
            model_name: Sentence transformer model to use
        """
        self.taxonomy_path = taxonomy_path
        self.model_name = model_name
        
        # Load taxonomy
        print("📋 Loading insurance taxonomy...")
        self.taxonomy_df = pd.read_csv(taxonomy_path)
        self.labels = self.taxonomy_df['label'].tolist()
        print(f"✅ Loaded {len(self.labels)} insurance labels")
        
        # Initialize AI model
        print("🤖 Loading AI model...")
        self.model = SentenceTransformer(model_name)
        
        # Initialize weighted similarity calculator
        self.similarity_calc = SimilarityCalculator(self.model, self.labels)
        
        # Few-shot learning storage
        self.validations_path = Path('data/manual_validations.json')
        self.load_validations()
        
    def load_validations(self):
        """Load existing manual validations for few-shot learning"""
        if self.validations_path.exists():
            with open(self.validations_path, 'r') as f:
                self.validations = json.load(f)
            print(f"🧠 Loaded {self.validations['metadata']['total_validations']} previous validations")
        else:
            self.validations = {
                'positive_examples': {},  # label -> list of company data
                'negative_examples': {},  # label -> list of company data  
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'total_validations': 0,
                    'approach': 'weighted_similarity_with_tag_emphasis'
                }
            }
            print("🆕 Starting fresh validation storage")
            
    def save_validations(self):
        """Save manual validations to disk"""
        self.validations_path.parent.mkdir(exist_ok=True)
        with open(self.validations_path, 'w') as f:
            json.dump(self.validations, f, indent=2)
            
    def get_suggestions(self, company_data: Dict, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get top-k insurance label suggestions using weighted approach
        
        Args:
            company_data: Company information dictionary
            top_k: Number of suggestions to return
            
        Returns:
            List of (label, score) tuples
        """
        # Get weighted similarity scores
        suggestions = self.similarity_calc.get_top_suggestions(company_data, top_k)
        
        # Apply few-shot learning boost if we have training data
        if self.validations['positive_examples']:
            suggestions = self._apply_few_shot_boost(company_data, suggestions)
            
        return suggestions
    
    def _apply_few_shot_boost(self, company_data: Dict, suggestions: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Apply few-shot learning boost based on manual validations
        
        Args:
            company_data: Current company data
            suggestions: Initial weighted suggestions
            
        Returns:
            Boosted suggestions
        """
        # Extract current company features for comparison
        current_tags = self.similarity_calc._extract_tags(company_data.get('business_tags', ''))
        current_desc = company_data.get('description', '')
        current_text = current_desc + ' ' + ' '.join(current_tags)
        
        boosted_suggestions = []
        
        for label, score in suggestions:
            boost = 0.0
            
            # Check positive examples (boost similar companies)
            if label in self.validations['positive_examples']:
                positive_examples = self.validations['positive_examples'][label]
                if positive_examples:
                    # Calculate similarity with positive examples
                    similarities = []
                    for example in positive_examples:
                        example_text = example.get('description', '') + ' ' + ' '.join(
                            self.similarity_calc._extract_tags(example.get('business_tags', ''))
                        )
                        similarity = self._text_similarity(current_text, example_text)
                        similarities.append(similarity)
                    
                    # Boost based on average similarity to positive examples
                    avg_similarity = np.mean(similarities) if similarities else 0
                    boost += avg_similarity * 0.15  # 15% boost for positive examples
            
            # Check negative examples (penalize similar rejections)
            if label in self.validations['negative_examples']:
                negative_examples = self.validations['negative_examples'][label]
                if negative_examples:
                    similarities = []
                    for example in negative_examples:
                        example_text = example.get('description', '') + ' ' + ' '.join(
                            self.similarity_calc._extract_tags(example.get('business_tags', ''))
                        )
                        similarity = self._text_similarity(current_text, example_text)
                        similarities.append(similarity)
                    
                    # Penalty based on similarity to negative examples
                    avg_similarity = np.mean(similarities) if similarities else 0
                    boost -= avg_similarity * 0.1  # 10% penalty for negative examples
            
            # Apply boost (cap at 1.0)
            final_score = min(1.0, max(0.0, score + boost))
            boosted_suggestions.append((label, final_score))
        
        # Re-sort by boosted scores
        boosted_suggestions.sort(key=lambda x: x[1], reverse=True)
        return boosted_suggestions
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using embeddings"""
        if not text1.strip() or not text2.strip():
            return 0.0
            
        embeddings = self.model.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)
    
    def record_validation(self, company_data: Dict, label: str, is_correct: bool):
        """
        Record manual validation for few-shot learning
        
        Args:
            company_data: Company information
            label: The insurance label
            is_correct: Whether the label is correct for this company
        """
        # Store the full company data (not just text) for better learning
        validation_data = {
            'description': company_data.get('description', ''),
            'business_tags': company_data.get('business_tags', ''),
            'sector': company_data.get('sector', ''),
            'category': company_data.get('category', ''),
            'niche': company_data.get('niche', ''),
            'timestamp': datetime.now().isoformat()
        }
        
        if is_correct:
            if label not in self.validations['positive_examples']:
                self.validations['positive_examples'][label] = []
            self.validations['positive_examples'][label].append(validation_data)
        else:
            if label not in self.validations['negative_examples']:
                self.validations['negative_examples'][label] = []
            self.validations['negative_examples'][label].append(validation_data)
            
        self.validations['metadata']['total_validations'] += 1
        self.save_validations()
        
    def explain_suggestion(self, company_data: Dict, top_k: int = 3) -> Dict:
        """
        Explain how suggestions were calculated using weighted approach
        
        Args:
            company_data: Company information
            top_k: Number of top suggestions to explain
            
        Returns:
            Detailed explanation of the scoring process
        """
        return self.similarity_calc.explain_similarity(company_data, top_k)
    
    def get_validation_stats(self) -> Dict:
        """Get statistics about manual validations"""
        stats = {
            'total_validations': self.validations['metadata']['total_validations'],
            'labels_with_positive_examples': len(self.validations['positive_examples']),
            'labels_with_negative_examples': len(self.validations['negative_examples']),
            'approach': self.validations['metadata'].get('approach', 'weighted_similarity'),
            'top_validated_labels': []
        }
        
        # Get most validated labels
        if self.validations['positive_examples']:
            sorted_labels = sorted(
                self.validations['positive_examples'].items(),
                key=lambda x: len(x[1]),
                reverse=True
            )[:10]
            
            stats['top_validated_labels'] = [
                {'label': label, 'examples': len(examples)}
                for label, examples in sorted_labels
            ]
        
        return stats
    
    def display_company_info(self, company: pd.Series, idx: int, total: int):
        """
        Display complete company information for manual labeling
        
        Args:
            company: Company data row
            idx: Current index  
            total: Total companies
        """
        print(f"\n{'='*80}")
        print(f"📊 Company {idx + 1}/{total}")
        print(f"{'='*80}")
        
        # Description
        print(f"\n📝 DESCRIPTION:")
        print(f"   {company['description']}")
        
        # Business tags with emphasis (your approach focuses on these)
        print(f"\n🏷️  BUSINESS TAGS (PRIMARY FEATURES):")
        try:
            tags = self.similarity_calc._extract_tags(company['business_tags'])
            if tags:
                for i, tag in enumerate(tags, 1):
                    print(f"   {i}. {tag}")
            else:
                print("   No tags available")
        except:
            print(f"   {company['business_tags']}")
        
        # Classification hierarchy
        print(f"\n📋 CLASSIFICATION HIERARCHY:")
        print(f"   Sector: {company['sector']}")
        print(f"   Category: {company['category']}")
        print(f"   Niche: {company['niche']}")
        
        print(f"\n{'='*80}")
        
    def display_weighted_suggestions(self, suggestions: List[Tuple[str, float]], show_details: bool = False):
        """
        Display AI suggestions with weighted scoring details
        
        Args:
            suggestions: List of (label, score) tuples
            show_details: Whether to show score breakdown
        """
        print(f"\n🤖 WEIGHTED AI SUGGESTIONS:")
        print(f"   Method: 40% Tag Semantics + 30% Tag Words + 20% Desc Semantics + 10% Desc Words")
        
        for i, (label, score) in enumerate(suggestions):
            # Visual confidence bar
            confidence_bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            
            print(f"\n  {i+1:2d}. {label}")
            print(f"      Total Score: {score:.3f} [{confidence_bar}]")
            
            if show_details:
                # Would need access to breakdown - simplified for now
                if score > 0.6:
                    print(f"      🔥 Strong match (likely driven by business tags)")
                elif score > 0.4:
                    print(f"      ⚡ Good match (multiple factors contributing)")
                else:
                    print(f"      💡 Moderate match (check for keyword overlaps)")
    
    def adjust_weights(self, new_weights: Dict[str, float]):
        """
        Allow dynamic adjustment of similarity weights
        
        Args:
            new_weights: Dictionary with new weight values
        """
        # Validate weights sum to 1.0
        total_weight = sum(new_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
            
        self.similarity_calc.weights.update(new_weights)
        print(f"✅ Updated similarity weights: {self.similarity_calc.weights}")
        
    def get_current_weights(self) -> Dict[str, float]:
        """Get current similarity weights"""
        return self.similarity_calc.weights.copy() 