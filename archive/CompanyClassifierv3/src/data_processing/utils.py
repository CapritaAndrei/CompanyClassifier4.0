"""
Data Processing Utilities for Insurance Classification
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import ast


class DataProcessor:
    """
    Utility class for processing company data and business tags
    """
    
    @staticmethod
    def extract_business_tags(tags_raw: Any) -> List[str]:
        """
        Extract and clean business tags from various formats
        
        Args:
            tags_raw: Raw tags data (string, list, or other)
            
        Returns:
            List of cleaned tag strings
        """
        if not tags_raw or pd.isna(tags_raw):
            return []
            
        try:
            if isinstance(tags_raw, str):
                # Handle string representation of list
                if tags_raw.startswith('[') and tags_raw.endswith(']'):
                    tags = ast.literal_eval(tags_raw)
                else:
                    # Single tag or comma-separated
                    tags = [tag.strip() for tag in tags_raw.split(',')]
            elif isinstance(tags_raw, list):
                tags = tags_raw
            else:
                tags = [str(tags_raw)]
                
            # Clean and filter
            cleaned_tags = []
            for tag in tags:
                if isinstance(tag, str):
                    cleaned = tag.strip()
                    if cleaned and len(cleaned) > 1:  # Filter very short tags
                        cleaned_tags.append(cleaned)
                        
            return cleaned_tags
            
        except Exception:
            return [str(tags_raw).strip()] if tags_raw else []
    
    @staticmethod
    def prepare_company_text(company_data: Dict) -> Dict[str, str]:
        """
        Prepare different text representations of company data
        
        Args:
            company_data: Dictionary with company information
            
        Returns:
            Dictionary with processed text fields
        """
        # Extract components
        description = str(company_data.get('description', '')).strip()
        tags = DataProcessor.extract_business_tags(company_data.get('business_tags', ''))
        sector = str(company_data.get('sector', '')).strip()
        category = str(company_data.get('category', '')).strip()
        niche = str(company_data.get('niche', '')).strip()
        
        # Create different text combinations
        tags_text = ' '.join(tags) if tags else ''
        hierarchy_text = ' '.join([sector, category, niche]) if any([sector, category, niche]) else ''
        
        return {
            'description': description,
            'tags_text': tags_text,
            'hierarchy_text': hierarchy_text,
            'combined_text': ' '.join([description, tags_text, hierarchy_text]).strip(),
            'tags_only': tags_text,
            'description_only': description
        }
    
    @staticmethod
    def validate_company_data(company_data: Dict) -> Dict[str, Any]:
        """
        Validate and clean company data
        
        Args:
            company_data: Raw company data
            
        Returns:
            Validation results with cleaned data
        """
        issues = []
        cleaned_data = company_data.copy()
        
        # Check for missing description
        if not company_data.get('description'):
            issues.append("Missing description")
            cleaned_data['description'] = ''
            
        # Check for missing or empty tags
        tags = DataProcessor.extract_business_tags(company_data.get('business_tags'))
        if not tags:
            issues.append("No business tags")
        cleaned_data['processed_tags'] = tags
        
        # Check for missing hierarchy info
        hierarchy_fields = ['sector', 'category', 'niche']
        missing_hierarchy = [field for field in hierarchy_fields 
                           if not company_data.get(field)]
        if missing_hierarchy:
            issues.append(f"Missing hierarchy: {', '.join(missing_hierarchy)}")
        
        return {
            'cleaned_data': cleaned_data,
            'issues': issues,
            'is_valid': len(issues) == 0,
            'completeness_score': (3 - len(missing_hierarchy)) / 3.0
        }
    
    @staticmethod
    def analyze_dataset(companies_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the company dataset for quality and completeness
        
        Args:
            companies_df: DataFrame with company data
            
        Returns:
            Dataset analysis results
        """
        total_companies = len(companies_df)
        
        # Analyze fields
        field_analysis = {}
        for field in ['description', 'business_tags', 'sector', 'category', 'niche']:
            non_null = companies_df[field].notna().sum()
            non_empty = (companies_df[field].astype(str).str.strip() != '').sum()
            
            field_analysis[field] = {
                'non_null': non_null,
                'non_empty': non_empty,
                'completeness': non_empty / total_companies
            }
        
        # Analyze business tags specifically
        all_tags = []
        for _, company in companies_df.iterrows():
            tags = DataProcessor.extract_business_tags(company.get('business_tags'))
            all_tags.extend(tags)
            
        tag_analysis = {
            'total_unique_tags': len(set(all_tags)),
            'total_tag_instances': len(all_tags),
            'avg_tags_per_company': len(all_tags) / total_companies if total_companies > 0 else 0,
            'companies_with_tags': sum(1 for _, company in companies_df.iterrows() 
                                     if DataProcessor.extract_business_tags(company.get('business_tags')))
        }
        
        return {
            'total_companies': total_companies,
            'field_analysis': field_analysis,
            'tag_analysis': tag_analysis,
            'summary': {
                'high_quality_companies': sum(1 for _, company in companies_df.iterrows()
                                             if DataProcessor.validate_company_data(company.to_dict())['is_valid']),
                'avg_completeness': np.mean([DataProcessor.validate_company_data(company.to_dict())['completeness_score']
                                           for _, company in companies_df.iterrows()])
            }
        }
    
    @staticmethod
    def filter_quality_companies(companies_df: pd.DataFrame, 
                                min_completeness: float = 0.6) -> pd.DataFrame:
        """
        Filter companies based on data quality
        
        Args:
            companies_df: Original DataFrame
            min_completeness: Minimum completeness score (0.0-1.0)
            
        Returns:
            Filtered DataFrame with high-quality companies
        """
        quality_mask = []
        
        for _, company in companies_df.iterrows():
            validation = DataProcessor.validate_company_data(company.to_dict())
            meets_criteria = (
                validation['completeness_score'] >= min_completeness and
                company.get('description', '').strip() != '' and
                len(DataProcessor.extract_business_tags(company.get('business_tags'))) > 0
            )
            quality_mask.append(meets_criteria)
        
        filtered_df = companies_df[quality_mask].copy()
        
        print(f"📊 Data Quality Filter Results:")
        print(f"   Original companies: {len(companies_df)}")
        print(f"   High-quality companies: {len(filtered_df)}")
        print(f"   Filtered out: {len(companies_df) - len(filtered_df)}")
        print(f"   Quality retention: {len(filtered_df)/len(companies_df):.1%}")
        
        return filtered_df 