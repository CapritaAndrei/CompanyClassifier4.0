#!/usr/bin/env python3
"""
Final SIC Hierarchical Classifier
Returns single best classification for each company through the hierarchy:
Division → Major Group → Industry Group
"""

import pandas as pd
import numpy as np
import json
import re
import ast
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Preprocessing utility functions
def preprocess_text(text, to_lower=True, punct_remove=True, stopword_remove=False, lemmatize=False):
    """Basic text preprocessing"""
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text).strip()
    
    if to_lower:
        text = text.lower()
    
    if punct_remove:
        # Remove punctuation but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def format_sic_code(division, major_group, industry_group, sic=None):
    """Format SIC codes with consistent leading zeros"""
    if major_group is None:
        return f"{division}"
    elif industry_group is None:
        return f"{division}-{str(major_group).zfill(2)}"
    elif sic is None:
        return f"{division}-{str(major_group).zfill(2)}-{str(industry_group).zfill(3)}"
    else:
        return f"{division}-{str(major_group).zfill(2)}-{str(industry_group).zfill(3)}-{str(sic).zfill(4)}"

def parse_tag_string(tag_string):
    """Parse business tags from string format"""
    if pd.isna(tag_string) or tag_string is None:
        return []
    
    tag_string = str(tag_string).strip()
    
    # Try to parse as list literal first
    try:
        if tag_string.startswith('[') and tag_string.endswith(']'):
            return ast.literal_eval(tag_string)
    except:
        pass
    
    # Split by comma as fallback
    tags = [tag.strip().strip("'\"") for tag in tag_string.split(',')]
    return [tag for tag in tags if tag]

def parse_keywords_string(keywords_string):
    """Parse keywords from string format"""
    return parse_tag_string(keywords_string)

def create_company_representation(company_row):
    """Create comprehensive company text representation"""
    parts = []
    
    # Description
    if 'description' in company_row and pd.notna(company_row['description']):
        desc = preprocess_text(company_row['description'])
        if desc:
            parts.append(f"Description: {desc}")
    
    # Business tags
    if 'business_tags' in company_row and pd.notna(company_row['business_tags']):
        tags = parse_tag_string(company_row['business_tags'])
        if tags:
            tags_text = ' '.join([preprocess_text(tag) for tag in tags])
            parts.append(f"Business Tags: {tags_text}")
    
    # Sector
    if 'sector' in company_row and pd.notna(company_row['sector']):
        sector = preprocess_text(company_row['sector'])
        if sector:
            parts.append(f"Sector: {sector}")
    
    # Category
    if 'category' in company_row and pd.notna(company_row['category']):
        category = preprocess_text(company_row['category'])
        if category:
            parts.append(f"Category: {category}")
    
    # Niche
    if 'niche' in company_row and pd.notna(company_row['niche']):
        niche = preprocess_text(company_row['niche'])
        if niche:
            parts.append(f"Niche: {niche}")
    
    return '. '.join(parts)

class FinalSICClassifier:
    """Final SIC classifier - single best result per company"""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.sic_data = None
        self.hierarchy = None
        self.embeddings = {}
        
    def load_and_prepare_data(self):
        """Load SIC data and create hierarchy using actual SIC descriptions"""
        print("📥 Loading SIC data and creating hierarchy...")
        
        # Load SIC data
        self.sic_data = pd.read_csv('data/input/sic_codes.csv')
        print(f"✅ Loaded {len(self.sic_data)} SIC codes")
        
        # Create hierarchy using actual SIC descriptions
        self.hierarchy = {
            'divisions': {},
            'major_groups': {},
            'industry_groups': {}
        }
        
        # Division level - collect all descriptions for each division
        for div in self.sic_data['Division'].unique():
            div_data = self.sic_data[self.sic_data['Division'] == div]
            # Use actual SIC descriptions, not generated text
            descriptions = div_data['Description'].unique()
            combined_desc = '; '.join(descriptions[:10])  # Use top 10 to avoid too long text
            
            self.hierarchy['divisions'][div] = {
                'name': self._get_division_name(div),
                'description': combined_desc,
                'count': len(div_data)
            }
        
        # Major Group level - use actual SIC descriptions  
        for mg in self.sic_data['Major Group'].unique():
            mg_data = self.sic_data[self.sic_data['Major Group'] == mg]
            descriptions = mg_data['Description'].unique()
            combined_desc = '; '.join(descriptions[:5])  # Use top 5
            
            self.hierarchy['major_groups'][mg] = {
                'division': mg_data['Division'].iloc[0],
                'description': combined_desc,
                'count': len(mg_data)
            }
        
        # Industry Group level - use actual SIC descriptions
        for ig in self.sic_data['Industry Group'].unique():
            ig_data = self.sic_data[self.sic_data['Industry Group'] == ig]
            descriptions = ig_data['Description'].unique()
            combined_desc = '; '.join(descriptions)  # Use all descriptions
            
            self.hierarchy['industry_groups'][ig] = {
                'major_group': ig_data['Major Group'].iloc[0],
                'division': ig_data['Division'].iloc[0],
                'description': combined_desc,
                'count': len(ig_data)
            }
        
        print(f"✅ Hierarchy: {len(self.hierarchy['divisions'])} divisions, {len(self.hierarchy['major_groups'])} major groups, {len(self.hierarchy['industry_groups'])} industry groups")
        
    def create_embeddings(self):
        """Create embeddings for all hierarchy levels"""
        print("🚀 Creating embeddings...")
        
        # Division embeddings
        div_texts = []
        self.div_codes = []
        for div_code, div_data in self.hierarchy['divisions'].items():
            text = f"{div_data['name']}. Examples: {div_data['description']}"
            div_texts.append(text)
            self.div_codes.append(div_code)
        
        self.embeddings['divisions'] = self.model.encode(div_texts)
        
        # Major group embeddings
        mg_texts = []
        self.mg_codes = []
        for mg_code, mg_data in self.hierarchy['major_groups'].items():
            text = f"Major Group {mg_code}: {mg_data['description']}"
            mg_texts.append(text)
            self.mg_codes.append(mg_code)
        
        self.embeddings['major_groups'] = self.model.encode(mg_texts)
        
        # Industry group embeddings
        ig_texts = []
        self.ig_codes = []
        for ig_code, ig_data in self.hierarchy['industry_groups'].items():
            text = f"Industry Group {ig_code}: {ig_data['description']}"
            ig_texts.append(text)
            self.ig_codes.append(ig_code)
        
        self.embeddings['industry_groups'] = self.model.encode(ig_texts)
        
        print(f"✅ Created embeddings for all levels")
    
    def classify_company(self, description):
        """Classify single company through hierarchy - returns single best result"""
        
        # Embed company description
        company_emb = self.model.encode([description])
        
        # Step 1: Best Division
        div_similarities = cosine_similarity(company_emb, self.embeddings['divisions'])[0]
        best_div_idx = np.argmax(div_similarities)
        best_div_code = self.div_codes[best_div_idx]
        div_confidence = float(div_similarities[best_div_idx])
        
        # Step 2: Best Major Group within this division
        target_mgs = [(i, mg) for i, mg in enumerate(self.mg_codes) 
                     if self.hierarchy['major_groups'][mg]['division'] == best_div_code]
        
        best_mg_code = None
        mg_confidence = 0.0
        
        if target_mgs:
            mg_indices = [i for i, _ in target_mgs]
            mg_embeddings = self.embeddings['major_groups'][mg_indices]
            mg_similarities = cosine_similarity(company_emb, mg_embeddings)[0]
            
            best_mg_local_idx = np.argmax(mg_similarities)
            best_mg_code = target_mgs[best_mg_local_idx][1]
            mg_confidence = float(mg_similarities[best_mg_local_idx])
        
        # Step 3: Best Industry Group within this major group
        best_ig_code = None
        ig_confidence = 0.0
        
        if best_mg_code:
            target_igs = [(i, ig) for i, ig in enumerate(self.ig_codes) 
                         if self.hierarchy['industry_groups'][ig]['major_group'] == best_mg_code]
            
            if target_igs:
                ig_indices = [i for i, _ in target_igs]
                ig_embeddings = self.embeddings['industry_groups'][ig_indices]
                ig_similarities = cosine_similarity(company_emb, ig_embeddings)[0]
                
                best_ig_local_idx = np.argmax(ig_similarities)
                best_ig_code = target_igs[best_ig_local_idx][1]
                ig_confidence = float(ig_similarities[best_ig_local_idx])
        
        # Step 4: Best SIC Code within this industry group
        best_sic_code = None
        sic_confidence = 0.0
        
        if best_ig_code:
            # Get all SIC codes in this industry group
            target_sics = self.sic_data[
                (self.sic_data['Division'] == best_div_code) &
                (self.sic_data['Major Group'] == best_mg_code) &
                (self.sic_data['Industry Group'] == best_ig_code)
            ]
            
            if len(target_sics) > 0:
                sic_descriptions = target_sics['Description'].tolist()
                sic_codes = target_sics['SIC'].tolist()
                
                if len(sic_descriptions) == 1:
                    # Only one SIC code, select it directly
                    best_sic_code = sic_codes[0]
                    sic_confidence = ig_confidence  # Use same confidence as IG
                else:
                    # Multiple SIC codes, find best match
                    sic_embeddings = self.model.encode(sic_descriptions)
                    sic_similarities = cosine_similarity(company_emb, sic_embeddings)[0]
                    
                    best_sic_idx = np.argmax(sic_similarities)
                    best_sic_code = sic_codes[best_sic_idx]
                    sic_confidence = float(sic_similarities[best_sic_idx])
        
        return {
            'division_code': best_div_code,
            'division_name': self.hierarchy['divisions'][best_div_code]['name'],
            'division_confidence': div_confidence,
            'major_group_code': best_mg_code,
            'major_group_confidence': mg_confidence,
            'industry_group_code': best_ig_code,
            'industry_group_confidence': ig_confidence,
            'sic_code': best_sic_code,
            'sic_confidence': sic_confidence
        }
    
    def classify_company_tfidf(self, description):
        """Classify company using TF-IDF instead of semantic embeddings"""
        
        # Create TF-IDF vectors for divisions
        div_texts = []
        for div_code in self.div_codes:
            div_data = self.hierarchy['divisions'][div_code]
            text = f"{div_data['name']}. {div_data['description']}"
            div_texts.append(text)
        
        # Add company description
        all_texts = div_texts + [description]
        
        # Create TF-IDF matrix
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
        tfidf_matrix = tfidf.fit_transform(all_texts)
        
        # Calculate similarities
        company_tfidf = tfidf_matrix[-1]  # Last one is company
        div_tfidf = tfidf_matrix[:-1]     # All others are divisions
        
        similarities = linear_kernel(company_tfidf, div_tfidf)[0]
        
        # Step 1: Best Division
        best_div_idx = np.argmax(similarities)
        best_div_code = self.div_codes[best_div_idx]
        div_confidence = float(similarities[best_div_idx])
        
        # Step 2: Best Major Group within this division (using TF-IDF)
        target_mgs = [(i, mg) for i, mg in enumerate(self.mg_codes) 
                     if self.hierarchy['major_groups'][mg]['division'] == best_div_code]
        
        best_mg_code = None
        mg_confidence = 0.0
        
        if target_mgs:
            mg_texts = []
            mg_codes_filtered = []
            for _, mg_code in target_mgs:
                mg_data = self.hierarchy['major_groups'][mg_code]
                text = f"Major Group {mg_code}: {mg_data['description']}"
                mg_texts.append(text)
                mg_codes_filtered.append(mg_code)
            
            # TF-IDF for major groups
            mg_all_texts = mg_texts + [description]
            mg_tfidf_matrix = tfidf.fit_transform(mg_all_texts)
            mg_company_tfidf = mg_tfidf_matrix[-1]
            mg_candidates_tfidf = mg_tfidf_matrix[:-1]
            
            mg_similarities = linear_kernel(mg_company_tfidf, mg_candidates_tfidf)[0]
            best_mg_local_idx = np.argmax(mg_similarities)
            best_mg_code = mg_codes_filtered[best_mg_local_idx]
            mg_confidence = float(mg_similarities[best_mg_local_idx])
        
        # Step 3: Best Industry Group (using TF-IDF)
        best_ig_code = None
        ig_confidence = 0.0
        
        if best_mg_code:
            target_igs = [(i, ig) for i, ig in enumerate(self.ig_codes) 
                         if self.hierarchy['industry_groups'][ig]['major_group'] == best_mg_code]
            
            if target_igs:
                ig_texts = []
                ig_codes_filtered = []
                for _, ig_code in target_igs:
                    ig_data = self.hierarchy['industry_groups'][ig_code]
                    text = f"Industry Group {ig_code}: {ig_data['description']}"
                    ig_texts.append(text)
                    ig_codes_filtered.append(ig_code)
                
                # TF-IDF for industry groups
                ig_all_texts = ig_texts + [description]
                ig_tfidf_matrix = tfidf.fit_transform(ig_all_texts)
                ig_company_tfidf = ig_tfidf_matrix[-1]
                ig_candidates_tfidf = ig_tfidf_matrix[:-1]
                
                ig_similarities = linear_kernel(ig_company_tfidf, ig_candidates_tfidf)[0]
                best_ig_local_idx = np.argmax(ig_similarities)
                best_ig_code = ig_codes_filtered[best_ig_local_idx]
                ig_confidence = float(ig_similarities[best_ig_local_idx])
        
        # Step 4: Best SIC Code within this industry group (using TF-IDF)
        best_sic_code = None
        sic_confidence = 0.0
        
        if best_ig_code:
            # Get all SIC codes in this industry group
            target_sics = self.sic_data[
                (self.sic_data['Division'] == best_div_code) &
                (self.sic_data['Major Group'] == best_mg_code) &
                (self.sic_data['Industry Group'] == best_ig_code)
            ]
            
            if len(target_sics) > 0:
                sic_descriptions = target_sics['Description'].tolist()
                sic_codes = target_sics['SIC'].tolist()
                
                if len(sic_descriptions) == 1:
                    # Only one SIC code, select it directly
                    best_sic_code = sic_codes[0]
                    sic_confidence = ig_confidence  # Use same confidence as IG
                else:
                    # Multiple SIC codes, find best match using TF-IDF
                    sic_all_texts = sic_descriptions + [description]
                    sic_tfidf_matrix = tfidf.fit_transform(sic_all_texts)
                    sic_company_tfidf = sic_tfidf_matrix[-1]
                    sic_candidates_tfidf = sic_tfidf_matrix[:-1]
                    
                    sic_similarities = linear_kernel(sic_company_tfidf, sic_candidates_tfidf)[0]
                    best_sic_idx = np.argmax(sic_similarities)
                    best_sic_code = sic_codes[best_sic_idx]
                    sic_confidence = float(sic_similarities[best_sic_idx])
        
        return {
            'division_code': best_div_code,
            'division_name': self.hierarchy['divisions'][best_div_code]['name'],
            'division_confidence': div_confidence,
            'major_group_code': best_mg_code,
            'major_group_confidence': mg_confidence,
            'industry_group_code': best_ig_code,
            'industry_group_confidence': ig_confidence,
            'sic_code': best_sic_code,
            'sic_confidence': sic_confidence
        }
    
    def classify_company_tag_focused(self, company_row):
        """Classify company using tag-focused matching"""
        
        # Extract business tags
        tags = parse_tag_string(company_row.get('business_tags', ''))
        if not tags:
            # Fallback to description if no tags
            description = create_company_representation(company_row)
            return self.classify_company(description)
        
        # Create tag-based text representation
        tags_text = ' '.join([preprocess_text(tag) for tag in tags])
        
        # Add category and niche for additional context
        additional_context = []
        if 'category' in company_row and pd.notna(company_row['category']):
            additional_context.append(preprocess_text(company_row['category']))
        if 'niche' in company_row and pd.notna(company_row['niche']):
            additional_context.append(preprocess_text(company_row['niche']))
        
        if additional_context:
            tags_text += ' ' + ' '.join(additional_context)
        
        print(f"🏷️  Tag-focused text: {tags_text}")
        
        # Use semantic similarity on tag-focused text
        company_emb = self.model.encode([tags_text])
        
        # Step 1: Best Division
        div_similarities = cosine_similarity(company_emb, self.embeddings['divisions'])[0]
        best_div_idx = np.argmax(div_similarities)
        best_div_code = self.div_codes[best_div_idx]
        div_confidence = float(div_similarities[best_div_idx])
        
        # Step 2: Best Major Group within this division
        target_mgs = [(i, mg) for i, mg in enumerate(self.mg_codes) 
                     if self.hierarchy['major_groups'][mg]['division'] == best_div_code]
        
        best_mg_code = None
        mg_confidence = 0.0
        
        if target_mgs:
            mg_indices = [i for i, _ in target_mgs]
            mg_embeddings = self.embeddings['major_groups'][mg_indices]
            mg_similarities = cosine_similarity(company_emb, mg_embeddings)[0]
            
            best_mg_local_idx = np.argmax(mg_similarities)
            best_mg_code = target_mgs[best_mg_local_idx][1]
            mg_confidence = float(mg_similarities[best_mg_local_idx])
        
        # Step 3: Best Industry Group within this major group
        best_ig_code = None
        ig_confidence = 0.0
        
        if best_mg_code:
            target_igs = [(i, ig) for i, ig in enumerate(self.ig_codes) 
                         if self.hierarchy['industry_groups'][ig]['major_group'] == best_mg_code]
            
            if target_igs:
                ig_indices = [i for i, _ in target_igs]
                ig_embeddings = self.embeddings['industry_groups'][ig_indices]
                ig_similarities = cosine_similarity(company_emb, ig_embeddings)[0]
                
                best_ig_local_idx = np.argmax(ig_similarities)
                best_ig_code = target_igs[best_ig_local_idx][1]
                ig_confidence = float(ig_similarities[best_ig_local_idx])
        
        # Step 4: Best SIC Code within this industry group
        best_sic_code = None
        sic_confidence = 0.0
        
        if best_ig_code:
            # Get all SIC codes in this industry group
            target_sics = self.sic_data[
                (self.sic_data['Division'] == best_div_code) &
                (self.sic_data['Major Group'] == best_mg_code) &
                (self.sic_data['Industry Group'] == best_ig_code)
            ]
            
            if len(target_sics) > 0:
                sic_descriptions = target_sics['Description'].tolist()
                sic_codes = target_sics['SIC'].tolist()
                
                if len(sic_descriptions) == 1:
                    # Only one SIC code, select it directly
                    best_sic_code = sic_codes[0]
                    sic_confidence = ig_confidence  # Use same confidence as IG
                else:
                    # Multiple SIC codes, find best match
                    sic_embeddings = self.model.encode(sic_descriptions)
                    sic_similarities = cosine_similarity(company_emb, sic_embeddings)[0]
                    
                    best_sic_idx = np.argmax(sic_similarities)
                    best_sic_code = sic_codes[best_sic_idx]
                    sic_confidence = float(sic_similarities[best_sic_idx])
        
        return {
            'division_code': best_div_code,
            'division_name': self.hierarchy['divisions'][best_div_code]['name'],
            'division_confidence': div_confidence,
            'major_group_code': best_mg_code,
            'major_group_confidence': mg_confidence,
            'industry_group_code': best_ig_code,
            'industry_group_confidence': ig_confidence,
            'sic_code': best_sic_code,
            'sic_confidence': sic_confidence
        }
    
    def classify_company_weighted(self, company_row, weights={'tags': 0.4, 'description': 0.3, 'category': 0.2, 'niche': 0.2}):
        """Classify company using weighted approach - higher weight for business tags"""
        
        # Create separate embeddings for different components
        components = {}
        
        # Business tags (highest weight)
        tags = parse_tag_string(company_row.get('business_tags', ''))
        if tags:
            tags_text = ' '.join([preprocess_text(tag) for tag in tags])
            components['tags'] = self.model.encode([tags_text])[0]
        
        # Description
        if 'description' in company_row and pd.notna(company_row['description']):
            desc = preprocess_text(company_row['description'])
            if desc:
                components['description'] = self.model.encode([desc])[0]
        
        # Category  
        if 'category' in company_row and pd.notna(company_row['category']):
            category = preprocess_text(company_row['category'])
            if category:
                components['category'] = self.model.encode([category])[0]
        
        # Niche
        if 'niche' in company_row and pd.notna(company_row['niche']):
            niche = preprocess_text(company_row['niche'])
            if niche:
                components['niche'] = self.model.encode([niche])[0]
        
        if not components:
            # Fallback to comprehensive representation
            description = create_company_representation(company_row)
            return self.classify_company(description)
        
        print(f"⚖️  Weighted components: {list(components.keys())} with weights {weights}")
        
        # Create weighted embedding
        weighted_embedding = np.zeros_like(list(components.values())[0])
        total_weight = 0
        
        for component, embedding in components.items():
            if component in weights:
                weight = weights[component]
                weighted_embedding += weight * embedding
                total_weight += weight
        
        # Normalize
        if total_weight > 0:
            weighted_embedding = weighted_embedding / total_weight
        
        weighted_embedding = weighted_embedding.reshape(1, -1)
        
        # Step 1: Best Division
        div_similarities = cosine_similarity(weighted_embedding, self.embeddings['divisions'])[0]
        best_div_idx = np.argmax(div_similarities)
        best_div_code = self.div_codes[best_div_idx]
        div_confidence = float(div_similarities[best_div_idx])
        
        # Step 2: Best Major Group within this division
        target_mgs = [(i, mg) for i, mg in enumerate(self.mg_codes) 
                     if self.hierarchy['major_groups'][mg]['division'] == best_div_code]
        
        best_mg_code = None
        mg_confidence = 0.0
        
        if target_mgs:
            mg_indices = [i for i, _ in target_mgs]
            mg_embeddings = self.embeddings['major_groups'][mg_indices]
            mg_similarities = cosine_similarity(weighted_embedding, mg_embeddings)[0]
            
            best_mg_local_idx = np.argmax(mg_similarities)
            best_mg_code = target_mgs[best_mg_local_idx][1]
            mg_confidence = float(mg_similarities[best_mg_local_idx])
        
        # Step 3: Best Industry Group within this major group
        best_ig_code = None
        ig_confidence = 0.0
        
        if best_mg_code:
            target_igs = [(i, ig) for i, ig in enumerate(self.ig_codes) 
                         if self.hierarchy['industry_groups'][ig]['major_group'] == best_mg_code]
            
            if target_igs:
                ig_indices = [i for i, _ in target_igs]
                ig_embeddings = self.embeddings['industry_groups'][ig_indices]
                ig_similarities = cosine_similarity(weighted_embedding, ig_embeddings)[0]
                
                best_ig_local_idx = np.argmax(ig_similarities)
                best_ig_code = target_igs[best_ig_local_idx][1]
                ig_confidence = float(ig_similarities[best_ig_local_idx])
        
        # Step 4: Best SIC Code within this industry group
        best_sic_code = None
        sic_confidence = 0.0
        
        if best_ig_code:
            # Get all SIC codes in this industry group
            target_sics = self.sic_data[
                (self.sic_data['Division'] == best_div_code) &
                (self.sic_data['Major Group'] == best_mg_code) &
                (self.sic_data['Industry Group'] == best_ig_code)
            ]
            
            if len(target_sics) > 0:
                sic_descriptions = target_sics['Description'].tolist()
                sic_codes = target_sics['SIC'].tolist()
                
                if len(sic_descriptions) == 1:
                    # Only one SIC code, select it directly
                    best_sic_code = sic_codes[0]
                    sic_confidence = ig_confidence  # Use same confidence as IG
                else:
                    # Multiple SIC codes, find best match
                    sic_embeddings = self.model.encode(sic_descriptions)
                    sic_similarities = cosine_similarity(weighted_embedding, sic_embeddings)[0]
                    
                    best_sic_idx = np.argmax(sic_similarities)
                    best_sic_code = sic_codes[best_sic_idx]
                    sic_confidence = float(sic_similarities[best_sic_idx])
        
        return {
            'division_code': best_div_code,
            'division_name': self.hierarchy['divisions'][best_div_code]['name'],
            'division_confidence': div_confidence,
            'major_group_code': best_mg_code,
            'major_group_confidence': mg_confidence,
            'industry_group_code': best_ig_code,
            'industry_group_confidence': ig_confidence,
            'sic_code': best_sic_code,
            'sic_confidence': sic_confidence
        }
    
    def test_first_company(self):
        """Test on first company from dataset using ALL classification approaches"""
        print("🧪 Testing ALL classification approaches on first company...")
        
        # Use csv module to read the file correctly
        import csv
        
        with open('data/input/ml_insurance_challenge.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            first_company = next(reader)
        
        print("📋 Raw company data:")
        print(f"  Description: {first_company['description'][:100]}...")
        print(f"  Business Tags: {first_company['business_tags']}")
        print(f"  Sector: {first_company['sector']}")
        print(f"  Category: {first_company['category']}")
        print(f"  Niche: {first_company['niche']}")
        
        # Create comprehensive company representation
        company_full_text = create_company_representation(first_company)
        
        print(f"\n📝 Comprehensive Company Text:")
        print(f"  {company_full_text[:200]}...")
        print(f"  Total length: {len(company_full_text)} characters")
        
        print("\n" + "="*60)
        print("🥊 CLASSIFICATION COMPARISON")
        print("="*60)
        
        # Method 1: Comprehensive Semantic Embeddings (Original improved)
        print("\n1️⃣ COMPREHENSIVE SEMANTIC EMBEDDINGS:")
        result1 = self.classify_company(company_full_text)
        formatted_1 = format_sic_code(result1['division_code'], result1.get('major_group_code'), result1.get('industry_group_code'), result1.get('sic_code'))
        sic_desc1 = ''
        if result1.get('sic_code'):
            sic_desc1 = self.sic_data[self.sic_data['SIC'] == result1['sic_code']]['Description'].iloc[0][:30]
        print(f"   📂 Classification: {formatted_1}")
        print(f"   📂 Division: {result1['division_code']} - {result1['division_name']} ({result1['division_confidence']:.3f})")
        if result1['major_group_code']:
            print(f"   📂 Major Group: {str(result1['major_group_code']).zfill(2)} ({result1['major_group_confidence']:.3f})")
        if result1['industry_group_code']:
            print(f"   📂 Industry Group: {str(result1['industry_group_code']).zfill(3)} ({result1['industry_group_confidence']:.3f})")
        if result1['sic_code']:
            print(f"   📂 SIC Code: {str(result1['sic_code']).zfill(4)} - {sic_desc1} ({result1['sic_confidence']:.3f})")
        
        # Method 2: TF-IDF Approach
        print("\n2️⃣ TF-IDF APPROACH:")
        result2 = self.classify_company_tfidf(company_full_text)
        formatted_2 = format_sic_code(result2['division_code'], result2.get('major_group_code'), result2.get('industry_group_code'), result2.get('sic_code'))
        sic_desc2 = ''
        if result2.get('sic_code'):
            sic_desc2 = self.sic_data[self.sic_data['SIC'] == result2['sic_code']]['Description'].iloc[0][:30]
        print(f"   📂 Classification: {formatted_2}")
        print(f"   📂 Division: {result2['division_code']} - {result2['division_name']} ({result2['division_confidence']:.3f})")
        if result2['major_group_code']:
            print(f"   📂 Major Group: {str(result2['major_group_code']).zfill(2)} ({result2['major_group_confidence']:.3f})")
        if result2['industry_group_code']:
            print(f"   📂 Industry Group: {str(result2['industry_group_code']).zfill(3)} ({result2['industry_group_confidence']:.3f})")
        if result2['sic_code']:
            print(f"   📂 SIC Code: {str(result2['sic_code']).zfill(4)} - {sic_desc2} ({result2['sic_confidence']:.3f})")
        
        # Method 3: Tag-Focused Matching
        print("\n3️⃣ TAG-FOCUSED MATCHING:")
        result3 = self.classify_company_tag_focused(first_company)
        formatted_3 = format_sic_code(result3['division_code'], result3.get('major_group_code'), result3.get('industry_group_code'), result3.get('sic_code'))
        sic_desc3 = ''
        if result3.get('sic_code'):
            sic_desc3 = self.sic_data[self.sic_data['SIC'] == result3['sic_code']]['Description'].iloc[0][:30]
        print(f"   📂 Classification: {formatted_3}")
        print(f"   📂 Division: {result3['division_code']} - {result3['division_name']} ({result3['division_confidence']:.3f})")
        if result3['major_group_code']:
            print(f"   📂 Major Group: {str(result3['major_group_code']).zfill(2)} ({result3['major_group_confidence']:.3f})")
        if result3['industry_group_code']:
            print(f"   📂 Industry Group: {str(result3['industry_group_code']).zfill(3)} ({result3['industry_group_confidence']:.3f})")
        if result3['sic_code']:
            print(f"   📂 SIC Code: {str(result3['sic_code']).zfill(4)} - {sic_desc3} ({result3['sic_confidence']:.3f})")
        
        # Method 4: Weighted Approach (Business tags get highest weight)
        print("\n4️⃣ WEIGHTED APPROACH (Tags=40%, Desc=30%, Cat=20%, Niche=10%):")
        result4 = self.classify_company_weighted(first_company)
        formatted_4 = format_sic_code(result4['division_code'], result4.get('major_group_code'), result4.get('industry_group_code'), result4.get('sic_code'))
        sic_desc4 = ''
        if result4.get('sic_code'):
            sic_desc4 = self.sic_data[self.sic_data['SIC'] == result4['sic_code']]['Description'].iloc[0][:30]
        print(f"   📂 Classification: {formatted_4}")
        print(f"   📂 Division: {result4['division_code']} - {result4['division_name']} ({result4['division_confidence']:.3f})")
        if result4['major_group_code']:
            print(f"   📂 Major Group: {str(result4['major_group_code']).zfill(2)} ({result4['major_group_confidence']:.3f})")
        if result4['industry_group_code']:
            print(f"   📂 Industry Group: {str(result4['industry_group_code']).zfill(3)} ({result4['industry_group_confidence']:.3f})")
        if result4['sic_code']:
            print(f"   📂 SIC Code: {str(result4['sic_code']).zfill(4)} - {sic_desc4} ({result4['sic_confidence']:.3f})")
        
        # Summary comparison
        print("\n" + "="*60)
        print("📊 CONFIDENCE SCORE COMPARISON:")
        print("="*60)
        
        results = [
            ("Comprehensive Semantic", result1),
            ("TF-IDF", result2), 
            ("Tag-Focused", result3),
            ("Weighted", result4)
        ]
        
        for name, result in results:
            div_conf = result['division_confidence']
            mg_conf = result.get('major_group_confidence', 0)
            ig_conf = result.get('industry_group_confidence', 0)
            avg_conf = (div_conf + mg_conf + ig_conf) / 3
            print(f"{name:20} | Div: {div_conf:.3f} | MG: {mg_conf:.3f} | IG: {ig_conf:.3f} | Avg: {avg_conf:.3f}")
        
        print("\n✅ All classification approaches tested!")
        return results
    
    def test_sample_companies(self, sample_size=10):
        """Test all classification approaches on a sample of companies"""
        print(f"🧪 Testing ALL classification approaches on {sample_size} companies...")
        
        # Use csv module to read the file correctly
        import csv
        
        companies = []
        with open('data/input/ml_insurance_challenge.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, company in enumerate(reader):
                if i >= sample_size:
                    break
                companies.append(company)
        
        print(f"✅ Loaded {len(companies)} companies for testing")
        
        # Store results for each method
        all_results = {
            'comprehensive': [],
            'tfidf': [],
            'tag_focused': [],
            'weighted': []
        }
        
        for i, company in enumerate(companies):
            print(f"\n{'='*60}")
            print(f"🏢 COMPANY {i+1}: {company['description'][:50]}...")
            print(f"   Tags: {company['business_tags'][:100]}...")
            print(f"   Category: {company['category']}")
            print('='*60)
            
            # Create comprehensive representation
            company_full_text = create_company_representation(company)
            
            try:
                # Method 1: Comprehensive Semantic
                result1 = self.classify_company(company_full_text)
                all_results['comprehensive'].append(result1)
                div_name = self.hierarchy['divisions'][result1['division_code']]['name']
                mg_desc = self.hierarchy['major_groups'].get(result1.get('major_group_code'), {}).get('description', 'N/A')[:50]
                ig_desc = self.hierarchy['industry_groups'].get(result1.get('industry_group_code'), {}).get('description', 'N/A')[:50]
                sic_desc1 = ''
                if result1.get('sic_code'):
                    sic_desc1 = self.sic_data[self.sic_data['SIC'] == result1['sic_code']]['Description'].iloc[0][:30]
                formatted_code1 = format_sic_code(result1['division_code'], result1.get('major_group_code'), result1.get('industry_group_code'), result1.get('sic_code'))
                print(f"1️⃣ Comprehensive: {formatted_code1}")
                print(f"   └─ {div_name} → {mg_desc}... → {ig_desc}... → {sic_desc1}... | SIC: {result1.get('sic_confidence', 0):.3f}")
                
                # Method 2: TF-IDF
                result2 = self.classify_company_tfidf(company_full_text)
                all_results['tfidf'].append(result2)
                div_name2 = self.hierarchy['divisions'][result2['division_code']]['name']
                mg_desc2 = self.hierarchy['major_groups'].get(result2.get('major_group_code'), {}).get('description', 'N/A')[:50]
                ig_desc2 = self.hierarchy['industry_groups'].get(result2.get('industry_group_code'), {}).get('description', 'N/A')[:50]
                sic_desc2 = ''
                if result2.get('sic_code'):
                    sic_desc2 = self.sic_data[self.sic_data['SIC'] == result2['sic_code']]['Description'].iloc[0][:30]
                formatted_code2 = format_sic_code(result2['division_code'], result2.get('major_group_code'), result2.get('industry_group_code'), result2.get('sic_code'))
                print(f"2️⃣ TF-IDF:        {formatted_code2}")
                print(f"   └─ {div_name2} → {mg_desc2}... → {ig_desc2}... → {sic_desc2}... | SIC: {result2.get('sic_confidence', 0):.3f}")
                
                # Method 3: Tag-Focused
                result3 = self.classify_company_tag_focused(company)
                all_results['tag_focused'].append(result3)
                div_name3 = self.hierarchy['divisions'][result3['division_code']]['name']
                mg_desc3 = self.hierarchy['major_groups'].get(result3.get('major_group_code'), {}).get('description', 'N/A')[:50]
                ig_desc3 = self.hierarchy['industry_groups'].get(result3.get('industry_group_code'), {}).get('description', 'N/A')[:50]
                sic_desc3 = ''
                if result3.get('sic_code'):
                    sic_desc3 = self.sic_data[self.sic_data['SIC'] == result3['sic_code']]['Description'].iloc[0][:30]
                formatted_code3 = format_sic_code(result3['division_code'], result3.get('major_group_code'), result3.get('industry_group_code'), result3.get('sic_code'))
                print(f"3️⃣ Tag-Focused:   {formatted_code3}")
                print(f"   └─ {div_name3} → {mg_desc3}... → {ig_desc3}... → {sic_desc3}... | SIC: {result3.get('sic_confidence', 0):.3f}")
                
                # Method 4: Weighted
                result4 = self.classify_company_weighted(company)
                all_results['weighted'].append(result4)
                div_name4 = self.hierarchy['divisions'][result4['division_code']]['name']
                mg_desc4 = self.hierarchy['major_groups'].get(result4.get('major_group_code'), {}).get('description', 'N/A')[:50]
                ig_desc4 = self.hierarchy['industry_groups'].get(result4.get('industry_group_code'), {}).get('description', 'N/A')[:50]
                sic_desc4 = ''
                if result4.get('sic_code'):
                    sic_desc4 = self.sic_data[self.sic_data['SIC'] == result4['sic_code']]['Description'].iloc[0][:30]
                formatted_code4 = format_sic_code(result4['division_code'], result4.get('major_group_code'), result4.get('industry_group_code'), result4.get('sic_code'))
                print(f"4️⃣ Weighted:      {formatted_code4}")
                print(f"   └─ {div_name4} → {mg_desc4}... → {ig_desc4}... → {sic_desc4}... | SIC: {result4.get('sic_confidence', 0):.3f}")
                
            except Exception as e:
                print(f"❌ Error processing company {i+1}: {e}")
                continue
        
        # Calculate aggregate statistics
        print(f"\n{'='*60}")
        print("📊 AGGREGATE STATISTICS")
        print('='*60)
        
        method_names = ['Comprehensive', 'TF-IDF', 'Tag-Focused', 'Weighted']
        method_keys = ['comprehensive', 'tfidf', 'tag_focused', 'weighted']
        
        print(f"{'Method':<15} | {'Avg Div':<8} | {'Avg MG':<8} | {'Avg IG':<8} | {'Overall':<8}")
        print("-" * 60)
        
        for name, key in zip(method_names, method_keys):
            results = all_results[key]
            if results:
                div_scores = [r['division_confidence'] for r in results]
                mg_scores = [r.get('major_group_confidence', 0) for r in results]
                ig_scores = [r.get('industry_group_confidence', 0) for r in results]
                
                avg_div = np.mean(div_scores)
                avg_mg = np.mean(mg_scores)
                avg_ig = np.mean(ig_scores)
                overall = (avg_div + avg_mg + avg_ig) / 3
                
                print(f"{name:<15} | {avg_div:<8.3f} | {avg_mg:<8.3f} | {avg_ig:<8.3f} | {overall:<8.3f}")
        
        print(f"\n✅ Sample testing complete! Best method highlighted above.")
        return all_results
    
    def _get_division_name(self, code):
        names = {
            'A': 'Agriculture, Forestry, and Fishing',
            'B': 'Mining', 'C': 'Construction', 'D': 'Manufacturing',
            'E': 'Transportation, Communications, Electric, Gas, and Sanitary Services',
            'F': 'Wholesale Trade', 'G': 'Retail Trade',
            'H': 'Finance, Insurance, and Real Estate',
            'I': 'Services', 'J': 'Public Administration'
        }
        return names.get(code, f'Division {code}')

def main():
    """Test the final classifier on 10 companies"""
    
    print("🎯 Final SIC Hierarchical Classifier")
    print("=" * 40)
    
    classifier = FinalSICClassifier()
    classifier.load_and_prepare_data()
    classifier.create_embeddings()
    
    # Test on sample of 10 companies
    all_results = classifier.test_sample_companies(sample_size=10)
    
    print(f"\n✅ Sample classification testing complete!")
    print(f"🚀 Ready to classify full dataset with best performing method!")

if __name__ == "__main__":
    main() 