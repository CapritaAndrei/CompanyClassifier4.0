"""
Heatmap Analyzer for analyzing label frequency and distribution across sectors
"""

import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)

class HeatmapAnalyzer:
    """Analyzes label frequency and distribution across sectors, categories, and niches"""
    
    def __init__(self):
        self.classification_data = None
        self.company_metadata = None
        self.sector_label_counts = defaultdict(lambda: defaultdict(int))
        self.label_analytics = {}
        
    def load_classification_data(self, classifications: Dict[int, List[Tuple[str, float]]], 
                               company_metadata: Dict[int, Dict[str, str]]) -> None:
        """
        Load classification results and company metadata
        
        Args:
            classifications: Dict mapping company_id to list of (label, confidence) tuples
            company_metadata: Dict mapping company_id to metadata (sector, category, niche)
        """
        self.classification_data = classifications
        self.company_metadata = company_metadata
        self._build_sector_label_counts()
        self._calculate_label_analytics()
        
        logger.info(f"Loaded classification data for {len(classifications)} companies")
        
    def _build_sector_label_counts(self) -> None:
        """Build frequency counts of labels by sector"""
        
        for company_id, labels in self.classification_data.items():
            if company_id not in self.company_metadata:
                continue
                
            sector = self.company_metadata[company_id].get('sector', 'Unknown')
            
            for label, confidence in labels:
                self.sector_label_counts[sector][label] += 1
                
        logger.info(f"Built label counts for {len(self.sector_label_counts)} sectors")
        
    def _calculate_label_analytics(self) -> None:
        """Calculate detailed analytics for each label in each sector"""
        
        for company_id, labels in self.classification_data.items():
            if company_id not in self.company_metadata:
                continue
                
            metadata = self.company_metadata[company_id]
            sector = metadata.get('sector', 'Unknown')
            category = metadata.get('category', 'Unknown')
            niche = metadata.get('niche', 'Unknown')
            
            for label, confidence in labels:
                analytics_key = (sector, label)
                
                if analytics_key not in self.label_analytics:
                    self.label_analytics[analytics_key] = {
                        'total_count': 0,
                        'categories': defaultdict(int),
                        'niches': defaultdict(int),
                        'confidences': [],
                        'company_ids': []
                    }
                
                analytics = self.label_analytics[analytics_key]
                analytics['total_count'] += 1
                analytics['categories'][category] += 1
                analytics['niches'][niche] += 1
                analytics['confidences'].append(confidence)
                analytics['company_ids'].append(company_id)
                
        logger.info(f"Calculated analytics for {len(self.label_analytics)} sector-label combinations")
        
    def get_sectors(self) -> List[str]:
        """Get list of all sectors"""
        return list(self.sector_label_counts.keys())
        
    def get_sector_labels_by_frequency(self, sector: str, ascending: bool = True) -> List[Tuple[str, int]]:
        """
        Get labels for a sector sorted by frequency
        
        Args:
            sector: Sector name
            ascending: If True, sort from least to most frequent (rarest first)
            
        Returns:
            List of (label, count) tuples sorted by frequency
        """
        if sector not in self.sector_label_counts:
            return []
            
        labels = self.sector_label_counts[sector]
        sorted_labels = sorted(labels.items(), key=lambda x: x[1], reverse=not ascending)
        
        return sorted_labels
        
    def get_label_analytics(self, sector: str, label: str) -> Optional[Dict]:
        """
        Get detailed analytics for a specific label in a sector
        
        Args:
            sector: Sector name
            label: Label name
            
        Returns:
            Dictionary with analytics or None if not found
        """
        analytics_key = (sector, label)
        
        if analytics_key not in self.label_analytics:
            return None
            
        analytics = self.label_analytics[analytics_key].copy()
        
        # Calculate summary statistics
        confidences = analytics['confidences']
        if confidences:
            analytics['confidence_stats'] = {
                'min': min(confidences),
                'max': max(confidences),
                'avg': sum(confidences) / len(confidences),
                'median': sorted(confidences)[len(confidences) // 2]
            }
        
        # Convert defaultdicts to regular dicts for JSON serialization
        analytics['categories'] = dict(analytics['categories'])
        analytics['niches'] = dict(analytics['niches'])
        
        return analytics
        
    def get_companies_with_label(self, sector: str, label: str) -> List[int]:
        """
        Get list of company IDs that have this label in this sector
        
        Args:
            sector: Sector name
            label: Label name
            
        Returns:
            List of company IDs
        """
        analytics = self.get_label_analytics(sector, label)
        if analytics:
            return analytics['company_ids']
        return []
        
    def get_sector_summary(self, sector: str) -> Dict:
        """
        Get summary statistics for a sector
        
        Args:
            sector: Sector name
            
        Returns:
            Dictionary with sector summary
        """
        if sector not in self.sector_label_counts:
            return {}
            
        labels = self.sector_label_counts[sector]
        
        # Count companies in this sector
        companies_in_sector = set()
        for company_id, metadata in self.company_metadata.items():
            if metadata.get('sector') == sector:
                companies_in_sector.add(company_id)
                
        # Get categories and niches in this sector
        categories = set()
        niches = set()
        for company_id in companies_in_sector:
            metadata = self.company_metadata[company_id]
            categories.add(metadata.get('category', 'Unknown'))
            niches.add(metadata.get('niche', 'Unknown'))
            
        return {
            'total_companies': len(companies_in_sector),
            'total_labels': len(labels),
            'total_label_assignments': sum(labels.values()),
            'categories_count': len(categories),
            'niches_count': len(niches),
            'avg_labels_per_company': sum(labels.values()) / len(companies_in_sector) if companies_in_sector else 0
        }
        
    def export_heatmap_data(self, output_path: str) -> None:
        """
        Export heatmap data to CSV for analysis
        
        Args:
            output_path: Path to save CSV file
        """
        rows = []
        
        for sector, labels in self.sector_label_counts.items():
            for label, count in labels.items():
                analytics = self.get_label_analytics(sector, label)
                
                if analytics:
                    row = {
                        'sector': sector,
                        'label': label,
                        'count': count,
                        'categories_spread': len(analytics['categories']),
                        'niches_spread': len(analytics['niches']),
                        'avg_confidence': analytics['confidence_stats']['avg'],
                        'min_confidence': analytics['confidence_stats']['min'],
                        'max_confidence': analytics['confidence_stats']['max']
                    }
                    rows.append(row)
                    
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Exported heatmap data to {output_path}")
        
    def get_overview_statistics(self) -> Dict:
        """Get overview statistics across all sectors"""
        
        total_companies = len(self.company_metadata)
        total_classifications = sum(len(labels) for labels in self.classification_data.values())
        companies_with_labels = sum(1 for labels in self.classification_data.values() if labels)
        
        # Get unique labels across all sectors
        all_labels = set()
        for labels in self.sector_label_counts.values():
            all_labels.update(labels.keys())
            
        return {
            'total_companies': total_companies,
            'companies_with_labels': companies_with_labels,
            'companies_without_labels': total_companies - companies_with_labels,
            'total_classifications': total_classifications,
            'unique_labels': len(all_labels),
            'sectors_count': len(self.sector_label_counts),
            'avg_labels_per_company': total_classifications / total_companies if total_companies > 0 else 0
        } 