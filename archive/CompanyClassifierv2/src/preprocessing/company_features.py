"""
Company feature extraction for SIC classification
"""
import pandas as pd
from .text_utils import preprocess_text, parse_tag_string


def create_company_representation(company_row):
    """
    Create comprehensive company text representation
    
    Args:
        company_row: Dictionary/Series with company data
    
    Returns:
        str: Comprehensive text representation of the company
    """
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