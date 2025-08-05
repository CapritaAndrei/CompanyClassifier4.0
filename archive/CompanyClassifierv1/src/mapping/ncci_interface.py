"""
NCCI Interface Module

Interfaces with NCCI's Class Look-Up tool to map NAICS codes to 
workers compensation insurance classifications.
"""

import pandas as pd
import requests
import time
from typing import Dict, List, Optional, Tuple
import json
from dataclasses import dataclass


@dataclass
class NCCIMapping:
    """Data class for NCCI classification mapping."""
    naics_code: str
    class_code: str
    class_description: str
    state_specific: bool = False
    confidence_score: float = 1.0


class NCCIInterface:
    """Interface with NCCI's Class Look-Up system."""
    
    def __init__(self):
        self.base_url = "https://www.ncci.com"
        self.class_lookup_endpoint = "/classlookup"  # May need adjustment based on actual API
        self.session = requests.Session()
        self.rate_limit_delay = 1.0  # Seconds between requests
        
        # Cache for NAICS -> Insurance class mappings
        self.naics_insurance_cache = {}
        self.manual_mappings = self._load_manual_mappings()
        
    def _load_manual_mappings(self) -> Dict[str, List[NCCIMapping]]:
        """Load manually researched NAICS -> Insurance mappings."""
        # Based on research findings, start with known mappings
        return {
            "237990": [  # Other Heavy and Civil Engineering Construction
                NCCIMapping("237990", "5537", "Concrete Construction", confidence_score=0.9)
            ],
            "311411": [  # Frozen Fruit, Juice, and Vegetable Manufacturing  
                NCCIMapping("311411", "2111", "Food Manufacturing", confidence_score=0.9)
            ],
            "111998": [  # All Other Miscellaneous Crop Farming
                NCCIMapping("111998", "0005", "Farm - Diversified Crops", confidence_score=0.8)
            ],
            "541620": [  # Environmental Consulting Services
                NCCIMapping("541620", "8601", "Architect or Engineer - Consulting", confidence_score=0.9)
            ],
            "811121": [  # Automotive Body, Paint, and Interior Repair
                NCCIMapping("811121", "8380", "Automobile Service or Repair", confidence_score=0.95)
            ]
        }
    
    def lookup_naics_classification(self, naics_code: str) -> List[NCCIMapping]:
        """Look up insurance classification for a NAICS code."""
        # Check manual mappings first
        if naics_code in self.manual_mappings:
            return self.manual_mappings[naics_code]
        
        # Check cache
        if naics_code in self.naics_insurance_cache:
            return self.naics_insurance_cache[naics_code]
        
        # For now, return pattern-based mapping
        # TODO: Implement actual NCCI API integration
        mappings = self._pattern_based_mapping(naics_code)
        
        # Cache the result
        self.naics_insurance_cache[naics_code] = mappings
        
        return mappings
    
    def _pattern_based_mapping(self, naics_code: str) -> List[NCCIMapping]:
        """Pattern-based mapping when direct lookup isn't available."""
        naics_2digit = naics_code[:2]
        naics_3digit = naics_code[:3]
        
        # Agriculture (11)
        if naics_2digit == "11":
            if naics_3digit == "111":  # Crop Production
                return [NCCIMapping(naics_code, "0005", "Farm - Diversified Crops", confidence_score=0.7)]
            elif naics_3digit == "112":  # Animal Production
                return [NCCIMapping(naics_code, "0006", "Farm - Livestock", confidence_score=0.7)]
        
        # Manufacturing (31-33)
        elif naics_2digit in ["31", "32", "33"]:
            if naics_3digit == "311":  # Food Manufacturing
                return [NCCIMapping(naics_code, "2111", "Food Manufacturing", confidence_score=0.7)]
            else:
                return [NCCIMapping(naics_code, "3632", "Manufacturing - General", confidence_score=0.6)]
        
        # Construction (23)
        elif naics_2digit == "23":
            if naics_3digit == "236":  # Construction of Buildings
                return [NCCIMapping(naics_code, "5645", "Carpentry - Residential", confidence_score=0.7)]
            elif naics_3digit == "237":  # Heavy Construction
                return [NCCIMapping(naics_code, "5537", "Concrete Construction", confidence_score=0.7)]
            else:
                return [NCCIMapping(naics_code, "5606", "Contractor - General", confidence_score=0.6)]
        
        # Professional Services (54)
        elif naics_2digit == "54":
            if naics_3digit == "541":
                return [NCCIMapping(naics_code, "8601", "Architect or Engineer - Consulting", confidence_score=0.6)]
        
        # Transportation (48-49)
        elif naics_2digit in ["48", "49"]:
            return [NCCIMapping(naics_code, "7219", "Trucking - General Freight", confidence_score=0.6)]
        
        # Retail (44-45)
        elif naics_2digit in ["44", "45"]:
            return [NCCIMapping(naics_code, "8017", "Store - Retail", confidence_score=0.6)]
        
        # Default fallback
        return [NCCIMapping(naics_code, "8810", "Clerical Office Employees", confidence_score=0.3)]
    
    def map_naics_to_insurance_labels(self, naics_list: List[str], 
                                     insurance_taxonomy_df: pd.DataFrame) -> pd.DataFrame:
        """Map NAICS codes to insurance taxonomy labels."""
        results = []
        
        for naics_code in naics_list:
            # Get NCCI classifications for this NAICS code
            ncci_mappings = self.lookup_naics_classification(naics_code)
            
            for ncci_mapping in ncci_mappings:
                # Try to match NCCI class description to insurance taxonomy
                matches = self._match_to_insurance_taxonomy(
                    ncci_mapping, insurance_taxonomy_df
                )
                
                for insurance_label, similarity_score in matches:
                    results.append({
                        'naics_code': naics_code,
                        'ncci_class_code': ncci_mapping.class_code,
                        'ncci_description': ncci_mapping.class_description,
                        'insurance_label': insurance_label,
                        'confidence_score': ncci_mapping.confidence_score,
                        'similarity_score': similarity_score,
                        'combined_score': ncci_mapping.confidence_score * similarity_score
                    })
        
        return pd.DataFrame(results)
    
    def _match_to_insurance_taxonomy(self, ncci_mapping: NCCIMapping, 
                                   taxonomy_df: pd.DataFrame) -> List[Tuple[str, float]]:
        """Match NCCI classification to insurance taxonomy labels."""
        from difflib import SequenceMatcher
        
        matches = []
        ncci_desc_lower = ncci_mapping.class_description.lower()
        
        for _, row in taxonomy_df.iterrows():
            label_name = row['label']
            
            # Direct text similarity
            label_lower = label_name.lower()
            similarity = SequenceMatcher(None, ncci_desc_lower, label_lower).ratio()
            
            # Boost similarity for keyword matches
            if 'Keywords' in row and pd.notna(row['Keywords']):
                keywords = str(row['Keywords']).lower().split()
                for keyword in keywords:
                    if keyword in ncci_desc_lower:
                        similarity += 0.1
            
            # Include matches above threshold
            if similarity > 0.3:
                matches.append((label_name, similarity))
        
        # Sort by similarity and return top matches
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:5]  # Top 5 matches
    
    def create_ground_truth_dataset(self, companies_df: pd.DataFrame, 
                                  taxonomy_df: pd.DataFrame,
                                  min_confidence: float = 0.5) -> pd.DataFrame:
        """Create ground truth dataset for supervised learning."""
        # Get unique NAICS codes from companies
        unique_naics = companies_df['naics_code'].dropna().unique()
        
        # Map NAICS to insurance labels
        mapping_df = self.map_naics_to_insurance_labels(unique_naics, taxonomy_df)
        
        # Filter by confidence threshold
        high_confidence_mappings = mapping_df[
            mapping_df['combined_score'] >= min_confidence
        ].copy()
        
        # Join back to companies
        ground_truth = companies_df.merge(
            high_confidence_mappings[['naics_code', 'insurance_label', 'combined_score']],
            on='naics_code',
            how='inner'
        )
        
        print(f"Ground Truth Dataset Created:")
        print(f"  Companies with ground truth: {len(ground_truth)}")
        print(f"  Unique insurance labels: {ground_truth['insurance_label'].nunique()}")
        print(f"  Average confidence: {ground_truth['combined_score'].mean():.3f}")
        
        return ground_truth
    
    def save_ground_truth(self, ground_truth_df: pd.DataFrame, output_path: str):
        """Save ground truth dataset for ML training."""
        ground_truth_df.to_csv(output_path, index=False)
        print(f"Ground truth dataset saved to: {output_path}")
        
        # Save summary statistics
        summary_path = output_path.replace('.csv', '_summary.csv')
        summary = ground_truth_df.groupby('insurance_label').agg({
            'naics_code': 'nunique',
            'combined_score': ['mean', 'count']
        }).round(3)
        summary.to_csv(summary_path)
        print(f"Summary statistics saved to: {summary_path}")
    
    def expand_manual_mappings(self, naics_mappings: Dict[str, List[NCCIMapping]]):
        """Add new manual NAICS mappings."""
        self.manual_mappings.update(naics_mappings) 