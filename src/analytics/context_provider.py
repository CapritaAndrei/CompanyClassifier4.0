"""
Context Provider for showing company details and assignment reasoning
"""

import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class ContextProvider:
    """Provides detailed context about companies and their label assignments"""
    
    def __init__(self, company_df: pd.DataFrame, company_tags: Dict[int, List[str]], 
                 classification_data: Dict[int, List[Tuple[str, float]]]):
        """
        Initialize context provider
        
        Args:
            company_df: DataFrame with company information
            company_tags: Dict mapping company_id to business tags
            classification_data: Dict mapping company_id to classification results
        """
        self.company_df = company_df
        self.company_tags = company_tags
        self.classification_data = classification_data
        
        # Create lookup for faster access
        self.company_lookup = {}
        for _, row in company_df.iterrows():
            self.company_lookup[row['company_id']] = row.to_dict()
            
    def get_company_details(self, company_id: int) -> Optional[Dict]:
        """
        Get detailed information about a company
        
        Args:
            company_id: Company ID
            
        Returns:
            Dictionary with company details or None if not found
        """
        if company_id not in self.company_lookup:
            logger.warning(f"Company {company_id} not found")
            return None
            
        company_info = self.company_lookup[company_id].copy()
        
        # Add business tags
        company_info['business_tags'] = self.company_tags.get(company_id, [])
        
        # Add classification results
        company_info['assigned_labels'] = self.classification_data.get(company_id, [])
        
        return company_info
        
    def show_company_context(self, company_id: int, focus_label: str = None) -> Dict:
        """
        Show comprehensive context for a company
        
        Args:
            company_id: Company ID
            focus_label: Optional label to focus on
            
        Returns:
            Dictionary with comprehensive company context
        """
        company_details = self.get_company_details(company_id)
        
        if not company_details:
            return {}
            
        context = {
            'company_id': company_id,
            'basic_info': {
                'sector': company_details.get('sector', 'Unknown'),
                'category': company_details.get('category', 'Unknown'),
                'niche': company_details.get('niche', 'Unknown')
            },
            'description': company_details.get('description', ''),
            'business_tags': company_details.get('business_tags', []),
            'assigned_labels': company_details.get('assigned_labels', [])
        }
        
        # Add focus label analysis if specified
        if focus_label:
            context['focus_label_analysis'] = self.analyze_label_assignment(
                company_id, focus_label
            )
            
        return context
        
    def analyze_label_assignment(self, company_id: int, label: str) -> Dict:
        """
        Analyze why a specific label was assigned to a company
        
        Args:
            company_id: Company ID
            label: Label to analyze
            
        Returns:
            Dictionary with assignment analysis
        """
        company_details = self.get_company_details(company_id)
        
        if not company_details:
            return {'error': 'Company not found'}
            
        assigned_labels = company_details.get('assigned_labels', [])
        
        # Find the label and its confidence
        label_info = None
        for assigned_label, confidence in assigned_labels:
            if assigned_label == label:
                label_info = (assigned_label, confidence)
                break
                
        if not label_info:
            return {'error': f'Label "{label}" not assigned to this company'}
            
        _, confidence = label_info
        business_tags = company_details.get('business_tags', [])
        
        # Analyze potential reasons for assignment
        analysis = {
            'label': label,
            'confidence': confidence,
            'assignment_factors': self._analyze_assignment_factors(
                business_tags, label, company_details
            ),
            'business_tags': business_tags,
            'company_sector': company_details.get('sector', 'Unknown'),
            'company_category': company_details.get('category', 'Unknown'),
            'company_niche': company_details.get('niche', 'Unknown')
        }
        
        return analysis
        
    def _analyze_assignment_factors(self, business_tags: List[str], 
                                  label: str, company_details: Dict) -> Dict:
        """
        Analyze factors that might have contributed to label assignment
        
        Args:
            business_tags: List of business tags
            label: Assigned label
            company_details: Company details
            
        Returns:
            Dictionary with assignment factors
        """
        factors = {
            'potential_keyword_matches': [],
            'semantic_indicators': [],
            'sector_alignment': self._check_sector_alignment(
                company_details.get('sector', 'Unknown'), label
            )
        }
        
        # Check for potential keyword matches
        label_words = set(label.lower().split())
        
        for tag in business_tags:
            tag_words = set(tag.lower().split())
            
            # Check for word overlap
            overlap = label_words.intersection(tag_words)
            if overlap:
                factors['potential_keyword_matches'].append({
                    'business_tag': tag,
                    'matching_words': list(overlap)
                })
                
        # Check for semantic indicators
        semantic_indicators = self._find_semantic_indicators(business_tags, label)
        factors['semantic_indicators'] = semantic_indicators
        
        return factors
        
    def _check_sector_alignment(self, sector: str, label: str) -> Dict:
        """
        Check alignment between company sector and assigned label
        
        Args:
            sector: Company sector
            label: Assigned label
            
        Returns:
            Dictionary with alignment analysis
        """
        sector_lower = sector.lower()
        label_lower = label.lower()
        
        # Define some basic sector-label alignment rules
        alignment_rules = {
            'manufacturing': ['manufacturing', 'production', 'factory', 'processing'],
            'services': ['services', 'consulting', 'repair', 'maintenance', 'installation'],
            'wholesale': ['wholesale', 'distribution', 'supply', 'trading'],
            'retail': ['retail', 'store', 'shop', 'sales']
        }
        
        alignment_score = 'unknown'
        matching_keywords = []
        
        if sector_lower in alignment_rules:
            sector_keywords = alignment_rules[sector_lower]
            matching_keywords = [kw for kw in sector_keywords if kw in label_lower]
            
            if matching_keywords:
                alignment_score = 'good'
            else:
                # Check if label contains keywords from other sectors
                other_sector_matches = []
                for other_sector, keywords in alignment_rules.items():
                    if other_sector != sector_lower:
                        matches = [kw for kw in keywords if kw in label_lower]
                        if matches:
                            other_sector_matches.extend(matches)
                            
                if other_sector_matches:
                    alignment_score = 'poor'
                else:
                    alignment_score = 'neutral'
                    
        return {
            'score': alignment_score,
            'sector': sector,
            'label': label,
            'matching_keywords': matching_keywords,
            'explanation': self._get_alignment_explanation(alignment_score, sector, label)
        }
        
    def _get_alignment_explanation(self, score: str, sector: str, label: str) -> str:
        """Get explanation for alignment score"""
        
        explanations = {
            'good': f'Label "{label}" contains keywords typical for {sector} sector',
            'poor': f'Label "{label}" contains keywords more typical for other sectors',
            'neutral': f'Label "{label}" is sector-neutral or unclear alignment with {sector}',
            'unknown': f'Cannot determine alignment between {sector} and "{label}"'
        }
        
        return explanations.get(score, 'Unknown alignment')
        
    def _find_semantic_indicators(self, business_tags: List[str], label: str) -> List[Dict]:
        """
        Find semantic indicators that might explain label assignment
        
        Args:
            business_tags: List of business tags
            label: Assigned label
            
        Returns:
            List of semantic indicators
        """
        indicators = []
        
        # Look for conceptual relationships
        conceptual_groups = {
            'food_related': ['food', 'restaurant', 'bakery', 'catering', 'dining', 'culinary'],
            'construction': ['construction', 'building', 'concrete', 'formwork', 'installation'],
            'services': ['services', 'consulting', 'repair', 'maintenance', 'support'],
            'manufacturing': ['manufacturing', 'production', 'factory', 'processing', 'assembly'],
            'technology': ['technology', 'software', 'digital', 'tech', 'IT', 'system']
        }
        
        label_lower = label.lower()
        
        for concept_group, keywords in conceptual_groups.items():
            # Check if label belongs to this concept group
            if any(keyword in label_lower for keyword in keywords):
                # Check if business tags also relate to this concept
                matching_tags = []
                for tag in business_tags:
                    tag_lower = tag.lower()
                    if any(keyword in tag_lower for keyword in keywords):
                        matching_tags.append(tag)
                        
                if matching_tags:
                    indicators.append({
                        'concept_group': concept_group,
                        'label_keywords': [kw for kw in keywords if kw in label_lower],
                        'matching_business_tags': matching_tags,
                        'explanation': f'Both label and business tags relate to {concept_group}'
                    })
                    
        return indicators
        
    def get_batch_context(self, company_ids: List[int], focus_label: str = None) -> List[Dict]:
        """
        Get context for multiple companies
        
        Args:
            company_ids: List of company IDs
            focus_label: Optional label to focus on
            
        Returns:
            List of context dictionaries
        """
        contexts = []
        
        for company_id in company_ids:
            context = self.show_company_context(company_id, focus_label)
            if context:
                contexts.append(context)
                
        return contexts
        
    def format_company_display(self, company_id: int, focus_label: str = None) -> str:
        """
        Format company information for display
        
        Args:
            company_id: Company ID
            focus_label: Optional label to focus on
            
        Returns:
            Formatted string for display
        """
        context = self.show_company_context(company_id, focus_label)
        
        if not context:
            return f"Company {company_id}: Not found"
            
        lines = []
        lines.append(f"Company {company_id}:")
        lines.append(f"  Sector: {context['basic_info']['sector']}")
        lines.append(f"  Category: {context['basic_info']['category']}")
        lines.append(f"  Niche: {context['basic_info']['niche']}")
        lines.append(f"  Business Tags: {', '.join(context['business_tags'])}")
        
        if context['description']:
            # Show full description (no truncation)
            description = context['description']
            lines.append(f"  Description: {description}")
            
        if focus_label and 'focus_label_analysis' in context:
            analysis = context['focus_label_analysis']
            lines.append(f"  Focus Label '{focus_label}' Analysis:")
            lines.append(f"    Confidence: {analysis['confidence']:.3f}")
            lines.append(f"    Sector Alignment: {analysis['assignment_factors']['sector_alignment']['score']}")
            
        return "\n".join(lines) 