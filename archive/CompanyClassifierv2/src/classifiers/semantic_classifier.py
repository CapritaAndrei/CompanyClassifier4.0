"""
Semantic classifier using comprehensive company representation
"""
from .base_classifier import BaseSICClassifier
from ..preprocessing.company_features import create_company_representation
from ..utils.formatting import get_division_name


class SemanticSICClassifier(BaseSICClassifier):
    """Comprehensive semantic embeddings classifier"""
    
    def classify_company(self, company_data):
        """
        Classify company using comprehensive semantic embeddings
        
        Args:
            company_data: Dictionary with company information
        
        Returns:
            dict: Classification result with confidence scores
        """
        # Create comprehensive company representation
        company_text = create_company_representation(company_data)
        
        # Embed company description
        company_emb = self.model.encode([company_text])
        
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