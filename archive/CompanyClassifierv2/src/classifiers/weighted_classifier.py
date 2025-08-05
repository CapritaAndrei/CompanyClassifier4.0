"""
Weighted classifier prioritizing business tags
"""
import numpy as np
import pandas as pd
from .base_classifier import BaseSICClassifier
from ..preprocessing.text_utils import preprocess_text, parse_tag_string
from ..utils.formatting import get_division_name


class WeightedSICClassifier(BaseSICClassifier):
    """Weighted approach classifier - higher weight for business tags"""
    
    def __init__(self, sic_hierarchy, model_name='all-MiniLM-L6-v2', 
                 weights=None):
        """
        Initialize weighted classifier
        
        Args:
            sic_hierarchy: SICHierarchy instance
            model_name: Sentence transformer model name
            weights: Dictionary of component weights
        """
        super().__init__(sic_hierarchy, model_name)
        self.weights = weights or {
            'tags': 0.4, 
            'description': 0.3, 
            'category': 0.2, 
            'niche': 0.1
        }
    
    def classify_company(self, company_data):
        """
        Classify company using weighted approach
        
        Args:
            company_data: Dictionary with company information
        
        Returns:
            dict: Classification result with confidence scores
        """
        # Create separate embeddings for different components
        components = {}
        
        # Business tags (highest weight)
        tags = parse_tag_string(company_data.get('business_tags', ''))
        if tags:
            tags_text = ' '.join([preprocess_text(tag) for tag in tags])
            components['tags'] = self.model.encode([tags_text])[0]
        
        # Description
        if 'description' in company_data and pd.notna(company_data['description']):
            desc = preprocess_text(company_data['description'])
            if desc:
                components['description'] = self.model.encode([desc])[0]
        
        # Category  
        if 'category' in company_data and pd.notna(company_data['category']):
            category = preprocess_text(company_data['category'])
            if category:
                components['category'] = self.model.encode([category])[0]
        
        # Niche
        if 'niche' in company_data and pd.notna(company_data['niche']):
            niche = preprocess_text(company_data['niche'])
            if niche:
                components['niche'] = self.model.encode([niche])[0]
        
        if not components:
            # Fallback to comprehensive representation
            from ..preprocessing.company_features import create_company_representation
            company_text = create_company_representation(company_data)
            company_emb = self.model.encode([company_text])
        else:
            # Create weighted embedding
            weighted_embedding = np.zeros_like(list(components.values())[0])
            total_weight = 0
            
            for component, embedding in components.items():
                if component in self.weights:
                    weight = self.weights[component]
                    weighted_embedding += weight * embedding
                    total_weight += weight
            
            # Normalize
            if total_weight > 0:
                weighted_embedding = weighted_embedding / total_weight
            
            company_emb = weighted_embedding.reshape(1, -1)
        
        # Step 1: Best Division
        best_div_code, div_confidence = self._classify_division(company_emb)
        
        # Step 2: Best Major Group within this division
        best_mg_code, mg_confidence = self._classify_major_group(company_emb, best_div_code)
        
        # Step 3: Best Industry Group within this major group
        best_ig_code, ig_confidence = self._classify_industry_group(company_emb, best_mg_code)
        
        # Step 4: Best SIC Code within this industry group
        best_sic_code, sic_confidence = self._classify_sic_code(
            company_emb, best_div_code, best_mg_code, best_ig_code
        )
        
        return {
            'division_code': best_div_code,
            'division_name': get_division_name(best_div_code),
            'division_confidence': div_confidence,
            'major_group_code': best_mg_code,
            'major_group_confidence': mg_confidence,
            'industry_group_code': best_ig_code,
            'industry_group_confidence': ig_confidence,
            'sic_code': best_sic_code,
            'sic_confidence': sic_confidence
        } 