"""
Base classifier interface for SIC classification
"""
from abc import ABC, abstractmethod
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from ..data.sic_hierarchy import SICHierarchy


class BaseSICClassifier(ABC):
    """Abstract base class for SIC classifiers"""
    
    def __init__(self, sic_hierarchy: SICHierarchy, model_name='all-MiniLM-L6-v2'):
        """
        Initialize base classifier
        
        Args:
            sic_hierarchy: SICHierarchy instance
            model_name: Sentence transformer model name
        """
        self.hierarchy = sic_hierarchy
        self.model = SentenceTransformer(model_name)
        self.embeddings = {}
        
    def create_embeddings(self):
        """Create embeddings for all hierarchy levels"""
        print("🚀 Creating embeddings...")
        
        # Division embeddings
        div_texts = []
        self.div_codes = []
        for div_code, div_data in self.hierarchy.hierarchy['divisions'].items():
            text = f"{div_data['name']}. Examples: {div_data['description']}"
            div_texts.append(text)
            self.div_codes.append(div_code)
        
        self.embeddings['divisions'] = self.model.encode(div_texts)
        
        # Major group embeddings
        mg_texts = []
        self.mg_codes = []
        for mg_code, mg_data in self.hierarchy.hierarchy['major_groups'].items():
            text = f"Major Group {mg_code}: {mg_data['description']}"
            mg_texts.append(text)
            self.mg_codes.append(mg_code)
        
        self.embeddings['major_groups'] = self.model.encode(mg_texts)
        
        # Industry group embeddings
        ig_texts = []
        self.ig_codes = []
        for ig_code, ig_data in self.hierarchy.hierarchy['industry_groups'].items():
            text = f"Industry Group {ig_code}: {ig_data['description']}"
            ig_texts.append(text)
            self.ig_codes.append(ig_code)
        
        self.embeddings['industry_groups'] = self.model.encode(ig_texts)
        
        print(f"✅ Created embeddings for all levels")
    
    @abstractmethod
    def classify_company(self, company_data):
        """
        Classify a company - must be implemented by subclasses
        
        Args:
            company_data: Company data (format depends on implementation)
        
        Returns:
            dict: Classification result
        """
        pass
    
    def _classify_division(self, company_embedding):
        """Find best division for company embedding"""
        div_similarities = cosine_similarity(company_embedding, self.embeddings['divisions'])[0]
        best_div_idx = np.argmax(div_similarities)
        best_div_code = self.div_codes[best_div_idx]
        div_confidence = float(div_similarities[best_div_idx])
        
        return best_div_code, div_confidence
    
    def _classify_major_group(self, company_embedding, division_code):
        """Find best major group within division"""
        target_mgs = [(i, mg) for i, mg in enumerate(self.mg_codes) 
                     if self.hierarchy.hierarchy['major_groups'][mg]['division'] == division_code]
        
        if not target_mgs:
            return None, 0.0
        
        mg_indices = [i for i, _ in target_mgs]
        mg_embeddings = self.embeddings['major_groups'][mg_indices]
        mg_similarities = cosine_similarity(company_embedding, mg_embeddings)[0]
        
        best_mg_local_idx = np.argmax(mg_similarities)
        best_mg_code = target_mgs[best_mg_local_idx][1]
        mg_confidence = float(mg_similarities[best_mg_local_idx])
        
        return best_mg_code, mg_confidence
    
    def _classify_industry_group(self, company_embedding, major_group_code):
        """Find best industry group within major group"""
        target_igs = [(i, ig) for i, ig in enumerate(self.ig_codes) 
                     if self.hierarchy.hierarchy['industry_groups'][ig]['major_group'] == major_group_code]
        
        if not target_igs:
            return None, 0.0
        
        ig_indices = [i for i, _ in target_igs]
        ig_embeddings = self.embeddings['industry_groups'][ig_indices]
        ig_similarities = cosine_similarity(company_embedding, ig_embeddings)[0]
        
        best_ig_local_idx = np.argmax(ig_similarities)
        best_ig_code = target_igs[best_ig_local_idx][1]
        ig_confidence = float(ig_similarities[best_ig_local_idx])
        
        return best_ig_code, ig_confidence
    
    def _classify_sic_code(self, company_embedding, division_code, major_group_code, industry_group_code):
        """Find best SIC code within industry group"""
        target_sics = self.hierarchy.get_sic_codes_in_industry_group(
            division_code, major_group_code, industry_group_code
        )
        
        if len(target_sics) == 0:
            return None, 0.0
        
        sic_descriptions = target_sics['Description'].tolist()
        sic_codes = target_sics['SIC'].tolist()
        
        if len(sic_descriptions) == 1:
            # Only one SIC code, select it directly
            return sic_codes[0], 0.8  # Default high confidence for single option
        
        # Multiple SIC codes, find best match
        sic_embeddings = self.model.encode(sic_descriptions)
        sic_similarities = cosine_similarity(company_embedding, sic_embeddings)[0]
        
        best_sic_idx = np.argmax(sic_similarities)
        best_sic_code = sic_codes[best_sic_idx]
        sic_confidence = float(sic_similarities[best_sic_idx])
        
        return best_sic_code, sic_confidence 