"""
Smart Sampler for selecting diverse company samples with different categories and niches
"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import random
import logging

logger = logging.getLogger(__name__)

class SmartCompanySampler:
    """Selects diverse company samples prioritizing different categories and niches"""
    
    def __init__(self, company_metadata: Dict[int, Dict[str, str]]):
        """
        Initialize sampler with company metadata
        
        Args:
            company_metadata: Dict mapping company_id to metadata (sector, category, niche)
        """
        self.company_metadata = company_metadata
        
    def get_diverse_samples(self, company_ids: List[int], max_samples: int = 10) -> List[int]:
        """
        Get diverse company samples prioritizing different categories and niches
        
        Args:
            company_ids: List of company IDs to sample from
            max_samples: Maximum number of samples to return
            
        Returns:
            List of selected company IDs with maximum diversity
        """
        if not company_ids:
            return []
            
        if len(company_ids) <= max_samples:
            return company_ids
            
        # Group companies by category and niche
        category_groups = defaultdict(list)
        niche_groups = defaultdict(list)
        
        for company_id in company_ids:
            if company_id not in self.company_metadata:
                continue
                
            metadata = self.company_metadata[company_id]
            category = metadata.get('category', 'Unknown')
            niche = metadata.get('niche', 'Unknown')
            
            category_groups[category].append(company_id)
            niche_groups[niche].append(company_id)
            
        # Use diversity-first sampling strategy
        selected = self._diversity_first_sampling(
            company_ids, category_groups, niche_groups, max_samples
        )
        
        logger.info(f"Selected {len(selected)} diverse samples from {len(company_ids)} companies")
        return selected
        
    def _diversity_first_sampling(self, company_ids: List[int], 
                                 category_groups: Dict[str, List[int]], 
                                 niche_groups: Dict[str, List[int]], 
                                 max_samples: int) -> List[int]:
        """
        Diversity-first sampling strategy
        
        Strategy:
        1. First pass: Select 1 company from each category
        2. Second pass: Select 1 company from each niche (if not already selected)
        3. Third pass: Fill remaining slots randomly from unselected companies
        
        Args:
            company_ids: All available company IDs
            category_groups: Companies grouped by category
            niche_groups: Companies grouped by niche
            max_samples: Maximum samples to return
            
        Returns:
            List of selected company IDs
        """
        selected = []
        used_companies = set()
        
        # Phase 1: Select one company from each category
        for category, companies in category_groups.items():
            if len(selected) >= max_samples:
                break
                
            # Select random company from this category
            available = [c for c in companies if c not in used_companies]
            if available:
                chosen = random.choice(available)
                selected.append(chosen)
                used_companies.add(chosen)
                
        # Phase 2: Select one company from each niche (if not already selected)
        for niche, companies in niche_groups.items():
            if len(selected) >= max_samples:
                break
                
            # Select random company from this niche that hasn't been selected
            available = [c for c in companies if c not in used_companies]
            if available:
                chosen = random.choice(available)
                selected.append(chosen)
                used_companies.add(chosen)
                
        # Phase 3: Fill remaining slots randomly
        remaining_companies = [c for c in company_ids if c not in used_companies]
        remaining_slots = max_samples - len(selected)
        
        if remaining_slots > 0 and remaining_companies:
            additional = random.sample(
                remaining_companies, 
                min(remaining_slots, len(remaining_companies))
            )
            selected.extend(additional)
            
        return selected
        
    def get_diversity_stats(self, company_ids: List[int]) -> Dict:
        """
        Get diversity statistics for a list of companies
        
        Args:
            company_ids: List of company IDs
            
        Returns:
            Dictionary with diversity statistics
        """
        if not company_ids:
            return {}
            
        categories = set()
        niches = set()
        sectors = set()
        
        for company_id in company_ids:
            if company_id not in self.company_metadata:
                continue
                
            metadata = self.company_metadata[company_id]
            sectors.add(metadata.get('sector', 'Unknown'))
            categories.add(metadata.get('category', 'Unknown'))
            niches.add(metadata.get('niche', 'Unknown'))
            
        return {
            'total_companies': len(company_ids),
            'unique_sectors': len(sectors),
            'unique_categories': len(categories),
            'unique_niches': len(niches),
            'sectors': list(sectors),
            'categories': list(categories),
            'niches': list(niches)
        }
        
    def get_category_niche_breakdown(self, company_ids: List[int]) -> Dict:
        """
        Get detailed breakdown of companies by category and niche
        
        Args:
            company_ids: List of company IDs
            
        Returns:
            Dictionary with category-niche breakdown
        """
        breakdown = defaultdict(lambda: defaultdict(list))
        
        for company_id in company_ids:
            if company_id not in self.company_metadata:
                continue
                
            metadata = self.company_metadata[company_id]
            category = metadata.get('category', 'Unknown')
            niche = metadata.get('niche', 'Unknown')
            
            breakdown[category][niche].append(company_id)
            
        # Convert to regular dict for JSON serialization
        return {
            category: dict(niches) 
            for category, niches in breakdown.items()
        }
        
    def explain_selection(self, selected_companies: List[int], 
                         original_companies: List[int]) -> Dict:
        """
        Explain why these companies were selected for diversity
        
        Args:
            selected_companies: List of selected company IDs
            original_companies: Original list of company IDs
            
        Returns:
            Dictionary explaining the selection rationale
        """
        selected_stats = self.get_diversity_stats(selected_companies)
        original_stats = self.get_diversity_stats(original_companies)
        
        return {
            'selection_summary': {
                'selected_count': len(selected_companies),
                'original_count': len(original_companies),
                'diversity_preserved': {
                    'categories': f"{selected_stats['unique_categories']}/{original_stats['unique_categories']}",
                    'niches': f"{selected_stats['unique_niches']}/{original_stats['unique_niches']}"
                }
            },
            'selected_diversity': selected_stats,
            'original_diversity': original_stats,
            'selection_strategy': "Diversity-first: 1 per category, then 1 per niche, then random fill"
        } 