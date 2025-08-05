"""
Text preprocessing utilities for SIC classification
"""
import pandas as pd
import re
import ast


def preprocess_text(text, to_lower=True, punct_remove=True, stopword_remove=False, lemmatize=False):
    """
    Basic text preprocessing
    
    Args:
        text: Input text to preprocess
        to_lower: Convert to lowercase
        punct_remove: Remove punctuation
        stopword_remove: Remove stopwords (not implemented)
        lemmatize: Lemmatize text (not implemented)
    
    Returns:
        str: Preprocessed text
    """
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


def parse_tag_string(tag_string):
    """
    Parse business tags from string format
    
    Args:
        tag_string: String representation of tags list
    
    Returns:
        list: List of parsed tags
    """
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
    """
    Parse keywords from string format
    
    Args:
        keywords_string: String representation of keywords
    
    Returns:
        list: List of parsed keywords
    """
    return parse_tag_string(keywords_string) 