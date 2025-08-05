"""
NAICS Lookup System

Automatically maps company niche classifications to NAICS codes using
official NAICS terminology and fuzzy matching.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from difflib import SequenceMatcher
import re


class NAICSLookup:
    """Comprehensive NAICS code lookup system."""
    
    def __init__(self):
        self.official_naics = self._load_official_naics_codes()
        self.fuzzy_threshold = 0.85  # High threshold for automated mapping
        
    def _load_official_naics_codes(self) -> Dict[str, str]:
        """Load official NAICS codes and descriptions."""
        # This would ideally load from official NAICS CSV/database
        # For now, including key mappings based on your data patterns
        return {
            # Agriculture, Forestry, Fishing and Hunting (11)
            "Apiculture": "112910",
            "All Other Miscellaneous Crop Farming": "111998",
            "Farm": "111000",
            "Crop Production": "111000",
            "Animal Production": "112000",
            "Livestock": "112111",
            "Poultry": "112300",
            "Aquaculture": "112511",
            
            # Manufacturing (31-33)
            "Frozen Fruit, Juice, and Vegetable Manufacturing": "311411",
            "Food Manufacturing": "311000",
            "Engineered Wood Member Manufacturing": "321213",
            "Plastics Plumbing Fixture Manufacturing": "326191",
            "Air-Conditioning and Warm Air Heating Equipment and Commercial and Industrial Refrigeration Equipment Manufacturing": "333415",
            
            # Construction (23)
            "Other Heavy and Civil Engineering Construction": "237990",
            "Heavy Construction": "237000",
            "Building Construction": "236000",
            "Residential Building Construction": "236110",
            "Commercial Building Construction": "236220",
            
            # Transportation (48-49)
            "Scenic and Sightseeing Transportation, Water": "487210",
            "Coastal and Great Lakes Passenger Transportation": "483112",
            "Other Nonscheduled Air Transportation": "481219",
            "General Freight Trucking": "484000",
            
            # Professional Services (54)
            "Environmental Consulting Services": "541620",
            "Management Consulting Services": "541611",
            "Computer Systems Design Services": "541512",
            "Architectural Services": "541310",
            "Engineering Services": "541330",
            
            # Finance and Insurance (52)
            "Financial Transactions Processing, Reserve, and Clearinghouse Activities": "522320",
            "Investment Banking": "523110",
            "Securities Brokerage": "523120",
            
            # Wholesale Trade (42)
            "Industrial and Personal Service Paper Merchant Wholesalers": "424130",
            "Wholesale Trade": "420000",
            
            # Retail Trade (44-45)
            "Retail": "440000",
            "Store": "440000",
            
            # Other Services (81)
            "Automotive Body, Paint, and Interior Repair and Maintenance": "811121",
            "Auto Repair": "811100",
            "Personal Services": "812000",
            
            # Public Administration (92)
            "Political Organizations": "813940",
            "Government": "920000"
        }
    
    def find_naics_code(self, niche: str) -> Optional[str]:
        """Find NAICS code for a given niche description."""
        if not niche or pd.isna(niche):
            return None
            
        # Direct match first
        if niche in self.official_naics:
            return self.official_naics[niche]
        
        # Fuzzy matching
        best_match = self._fuzzy_match_naics(niche)
        if best_match:
            return best_match
        
        # Pattern-based extraction
        pattern_match = self._extract_naics_from_pattern(niche)
        if pattern_match:
            return pattern_match
            
        return None
    
    def _fuzzy_match_naics(self, niche: str) -> Optional[str]:
        """Fuzzy match niche against official NAICS descriptions."""
        best_score = 0
        best_naics = None
        
        niche_clean = self._clean_text(niche)
        
        for naics_desc, naics_code in self.official_naics.items():
            desc_clean = self._clean_text(naics_desc)
            
            # Calculate similarity
            similarity = SequenceMatcher(None, niche_clean, desc_clean).ratio()
            
            if similarity > best_score and similarity >= self.fuzzy_threshold:
                best_score = similarity
                best_naics = naics_code
                
        return best_naics
    
    def _clean_text(self, text: str) -> str:
        """Clean text for better matching."""
        if not text:
            return ""
        
        # Convert to lowercase and remove extra spaces
        text = text.lower().strip()
        
        # Remove common suffixes that don't affect classification
        suffixes_to_remove = [
            " services", " service", " manufacturing", " construction",
            " and maintenance", " repair", " installation"
        ]
        
        for suffix in suffixes_to_remove:
            if text.endswith(suffix):
                text = text[:-len(suffix)].strip()
        
        return text
    
    def _extract_naics_from_pattern(self, niche: str) -> Optional[str]:
        """Extract NAICS code based on industry patterns."""
        niche_lower = niche.lower()
        
        # Manufacturing patterns
        if any(term in niche_lower for term in ['manufacturing', 'production', 'factory']):
            if 'food' in niche_lower or 'fruit' in niche_lower or 'vegetable' in niche_lower:
                return "311000"  # Food Manufacturing
            elif 'wood' in niche_lower or 'lumber' in niche_lower:
                return "321000"  # Wood Product Manufacturing
            elif 'plastic' in niche_lower:
                return "326000"  # Plastics Manufacturing
            else:
                return "330000"  # Manufacturing (general)
        
        # Construction patterns
        if any(term in niche_lower for term in ['construction', 'building', 'contractor']):
            if 'heavy' in niche_lower or 'civil' in niche_lower or 'engineering' in niche_lower:
                return "237000"  # Heavy Construction
            elif 'residential' in niche_lower:
                return "236110"  # Residential Building Construction
            elif 'commercial' in niche_lower:
                return "236220"  # Commercial Building Construction
            else:
                return "236000"  # Construction (general)
        
        # Agriculture patterns
        if any(term in niche_lower for term in ['farm', 'agriculture', 'crop', 'livestock']):
            if 'crop' in niche_lower or 'farming' in niche_lower:
                return "111000"  # Crop Production
            elif 'livestock' in niche_lower or 'animal' in niche_lower:
                return "112000"  # Animal Production
            else:
                return "110000"  # Agriculture (general)
        
        # Transportation patterns
        if any(term in niche_lower for term in ['transportation', 'shipping', 'freight', 'trucking']):
            if 'water' in niche_lower or 'marine' in niche_lower:
                return "483000"  # Water Transportation
            elif 'air' in niche_lower or 'aviation' in niche_lower:
                return "481000"  # Air Transportation
            elif 'truck' in niche_lower or 'freight' in niche_lower:
                return "484000"  # Truck Transportation
            else:
                return "480000"  # Transportation (general)
        
        # Services patterns
        if any(term in niche_lower for term in ['consulting', 'professional', 'services']):
            if 'environmental' in niche_lower:
                return "541620"  # Environmental Consulting
            elif 'management' in niche_lower:
                return "541611"  # Management Consulting
            elif 'computer' in niche_lower or 'software' in niche_lower:
                return "541512"  # Computer Systems Design
            elif 'engineering' in niche_lower:
                return "541330"  # Engineering Services
            elif 'architectural' in niche_lower:
                return "541310"  # Architectural Services
            else:
                return "541000"  # Professional Services (general)
        
        # Repair/Maintenance patterns
        if any(term in niche_lower for term in ['repair', 'maintenance']):
            if 'automotive' in niche_lower or 'auto' in niche_lower or 'car' in niche_lower:
                return "811100"  # Automotive Repair
            else:
                return "811000"  # Repair and Maintenance (general)
        
        return None
    
    def map_all_niches(self, companies_df: pd.DataFrame) -> pd.DataFrame:
        """Map all company niches to NAICS codes."""
        result_df = companies_df.copy()
        
        # Apply NAICS mapping
        result_df['naics_code'] = result_df['niche'].apply(self.find_naics_code)
        result_df['naics_mapped'] = result_df['naics_code'].notna()
        
        # Statistics
        total_companies = len(result_df)
        mapped_companies = result_df['naics_mapped'].sum()
        mapping_rate = mapped_companies / total_companies
        
        print(f"Comprehensive NAICS Mapping Results:")
        print(f"  Total companies: {total_companies}")
        print(f"  Successfully mapped: {mapped_companies}")
        print(f"  Mapping rate: {mapping_rate:.1%}")
        
        # Show unmapped niches
        unmapped = result_df[~result_df['naics_mapped']]['niche'].value_counts()
        if len(unmapped) > 0:
            print(f"\n  Top 10 unmapped niches:")
            for niche, count in unmapped.head(10).items():
                print(f"    {count:3d} companies: {niche}")
        
        return result_df
    
    def get_mapping_summary(self, companies_df: pd.DataFrame) -> pd.DataFrame:
        """Get summary of niche → NAICS mappings."""
        mapped_df = self.map_all_niches(companies_df)
        
        # Group by niche and NAICS
        summary = mapped_df.groupby(['niche', 'naics_code']).size().reset_index(name='company_count')
        summary = summary.sort_values('company_count', ascending=False)
        
        return summary
    
    def add_manual_mappings(self, niche_naics_pairs: List[Tuple[str, str]]):
        """Add manual niche → NAICS mappings."""
        for niche, naics_code in niche_naics_pairs:
            self.official_naics[niche] = naics_code 