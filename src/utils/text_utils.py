"""
Text processing utilities for cleaning and preprocessing business tags
"""

import re
import ast
from typing import List, Union
import logging

logger = logging.getLogger(__name__)

class TextProcessor:
    """Handles text cleaning and preprocessing for business tags"""
    
    @staticmethod
    def parse_business_tags(tags_string: str) -> List[str]:
        """
        Parse business tags from string representation of list
        
        Args:
            tags_string: String like "['tag1', 'tag2', 'tag3']"
            
        Returns:
            List of cleaned tag strings
        """
        try:
            # Try to parse as literal list
            if isinstance(tags_string, str):
                # Handle the case where it's already a string representation of a list
                tags_list = ast.literal_eval(tags_string)
            else:
                tags_list = tags_string
                
            # Ensure it's a list
            if not isinstance(tags_list, list):
                return []
                
            # Clean each tag
            cleaned_tags = []
            for tag in tags_list:
                if isinstance(tag, str):
                    cleaned_tag = TextProcessor.clean_text(tag)
                    if cleaned_tag:  # Only add non-empty tags
                        cleaned_tags.append(cleaned_tag)
                        
            return cleaned_tags
            
        except (ValueError, SyntaxError) as e:
            logger.warning(f"Failed to parse business tags: {tags_string}. Error: {e}")
            return []
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean individual text strings
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
            
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep alphanumeric, spaces, and common punctuation
        text = re.sub(r'[^\w\s\-\&\.\,]', '', text)
        
        # Remove multiple consecutive spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def normalize_for_matching(text: str) -> str:
        """
        Normalize text for keyword matching
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text (lowercase, minimal punctuation)
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and extra spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def extract_keywords(text: str) -> List[str]:
        """
        Extract meaningful keywords from text
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of keywords
        """
        if not isinstance(text, str):
            return []
            
        # Normalize text
        normalized = TextProcessor.normalize_for_matching(text)
        
        # Split into words
        words = normalized.split()
        
        # Filter out very short words and common stop words
        stop_words = {'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return keywords
    
    @staticmethod
    def calculate_keyword_overlap(text1: str, text2: str) -> float:
        """
        Calculate keyword overlap between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Overlap score (0-1)
        """
        keywords1 = set(TextProcessor.extract_keywords(text1))
        keywords2 = set(TextProcessor.extract_keywords(text2))
        
        if not keywords1 or not keywords2:
            return 0.0
            
        intersection = keywords1.intersection(keywords2)
        union = keywords1.union(keywords2)
        
        return len(intersection) / len(union) if union else 0.0 