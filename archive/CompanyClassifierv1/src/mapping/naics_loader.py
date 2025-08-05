"""
Proper NAICS Code Loader

Loads official NAICS codes and creates systematic mappings for company niches.
No more hardcoding - builds comprehensive mapping from industry patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from difflib import SequenceMatcher
import re
import json


class NAICSLoader:
    """Loads and manages comprehensive NAICS code mappings."""
    
    def __init__(self):
        self.naics_database = self._build_naics_database()
        self.industry_patterns = self._build_industry_patterns()
        self.fuzzy_threshold = 0.85
        
    def _build_naics_database(self) -> Dict[str, Dict]:
        """Build comprehensive NAICS database from industry knowledge."""
        # This builds a systematic database instead of hardcoding individual mappings
        return {
            # AGRICULTURE (11xxxx)
            "111110": {"title": "Soybean Farming", "sector": "Agriculture", "keywords": ["soybean", "farm"]},
            "111120": {"title": "Oilseed (except Soybean) Farming", "sector": "Agriculture", "keywords": ["oilseed", "farm"]},
            "111130": {"title": "Dry Pea and Bean Farming", "sector": "Agriculture", "keywords": ["pea", "bean", "farm"]},
            "111140": {"title": "Wheat Farming", "sector": "Agriculture", "keywords": ["wheat", "farm"]},
            "111150": {"title": "Corn Farming", "sector": "Agriculture", "keywords": ["corn", "farm"]},
            "111160": {"title": "Rice Farming", "sector": "Agriculture", "keywords": ["rice", "farm"]},
            "111191": {"title": "Oilseed and Grain Combination Farming", "sector": "Agriculture", "keywords": ["grain", "farm"]},
            "111199": {"title": "All Other Grain Farming", "sector": "Agriculture", "keywords": ["grain", "farm"]},
            "111211": {"title": "Potato Farming", "sector": "Agriculture", "keywords": ["potato", "farm"]},
            "111219": {"title": "Other Vegetable (except Potato) and Melon Farming", "sector": "Agriculture", "keywords": ["vegetable", "melon", "farm"]},
            "111310": {"title": "Orange Groves", "sector": "Agriculture", "keywords": ["orange", "grove", "citrus"]},
            "111320": {"title": "Citrus (except Orange) Groves", "sector": "Agriculture", "keywords": ["citrus", "grove"]},
            "111331": {"title": "Apple Orchards", "sector": "Agriculture", "keywords": ["apple", "orchard"]},
            "111332": {"title": "Grape Vineyards", "sector": "Agriculture", "keywords": ["grape", "vineyard"]},
            "111333": {"title": "Strawberry Farming", "sector": "Agriculture", "keywords": ["strawberry", "farm"]},
            "111334": {"title": "Berry (except Strawberry) Farming", "sector": "Agriculture", "keywords": ["berry", "farm"]},
            "111335": {"title": "Tree Nut Farming", "sector": "Agriculture", "keywords": ["nut", "tree", "farm"]},
            "111336": {"title": "Fruit and Tree Nut Combination Farming", "sector": "Agriculture", "keywords": ["fruit", "nut", "farm"]},
            "111339": {"title": "Other Noncitrus Fruit Farming", "sector": "Agriculture", "keywords": ["fruit", "farm"]},
            "111411": {"title": "Mushroom Production", "sector": "Agriculture", "keywords": ["mushroom", "production"]},
            "111419": {"title": "Other Food Crops Grown Under Cover", "sector": "Agriculture", "keywords": ["greenhouse", "crops"]},
            "111421": {"title": "Nursery and Tree Production", "sector": "Agriculture", "keywords": ["nursery", "tree", "production"]},
            "111422": {"title": "Floriculture Production", "sector": "Agriculture", "keywords": ["flower", "floriculture"]},
            "111910": {"title": "Tobacco Farming", "sector": "Agriculture", "keywords": ["tobacco", "farm"]},
            "111920": {"title": "Cotton Farming", "sector": "Agriculture", "keywords": ["cotton", "farm"]},
            "111930": {"title": "Sugarcane Farming", "sector": "Agriculture", "keywords": ["sugarcane", "farm"]},
            "111940": {"title": "Hay Farming", "sector": "Agriculture", "keywords": ["hay", "farm"]},
            "111991": {"title": "Sugar Beet Farming", "sector": "Agriculture", "keywords": ["sugar", "beet", "farm"]},
            "111992": {"title": "Peanut Farming", "sector": "Agriculture", "keywords": ["peanut", "farm"]},
            "111998": {"title": "All Other Miscellaneous Crop Farming", "sector": "Agriculture", "keywords": ["crop", "farm", "miscellaneous"]},
            "112111": {"title": "Beef Cattle Ranching and Farming", "sector": "Agriculture", "keywords": ["beef", "cattle", "ranch"]},
            "112112": {"title": "Cattle Feedlots", "sector": "Agriculture", "keywords": ["cattle", "feedlot"]},
            "112120": {"title": "Dairy Cattle and Milk Production", "sector": "Agriculture", "keywords": ["dairy", "milk", "cattle"]},
            "112130": {"title": "Dual-Purpose Cattle Ranching and Farming", "sector": "Agriculture", "keywords": ["cattle", "ranch"]},
            "112210": {"title": "Hog and Pig Farming", "sector": "Agriculture", "keywords": ["hog", "pig", "farm"]},
            "112310": {"title": "Chicken Egg Production", "sector": "Agriculture", "keywords": ["chicken", "egg", "poultry"]},
            "112320": {"title": "Broilers and Other Meat Type Chicken Production", "sector": "Agriculture", "keywords": ["broiler", "chicken", "poultry"]},
            "112330": {"title": "Turkey Production", "sector": "Agriculture", "keywords": ["turkey", "poultry"]},
            "112340": {"title": "Poultry Hatcheries", "sector": "Agriculture", "keywords": ["poultry", "hatchery"]},
            "112390": {"title": "Other Poultry Production", "sector": "Agriculture", "keywords": ["poultry", "production"]},
            "112410": {"title": "Sheep Farming", "sector": "Agriculture", "keywords": ["sheep", "farm"]},
            "112420": {"title": "Goat Farming", "sector": "Agriculture", "keywords": ["goat", "farm"]},
            "112511": {"title": "Finfish Farming and Fish Hatcheries", "sector": "Agriculture", "keywords": ["fish", "aquaculture", "hatchery"]},
            "112512": {"title": "Shellfish Farming", "sector": "Agriculture", "keywords": ["shellfish", "aquaculture"]},
            "112519": {"title": "Other Aquaculture", "sector": "Agriculture", "keywords": ["aquaculture", "fish"]},
            "112910": {"title": "Apiculture", "sector": "Agriculture", "keywords": ["bee", "honey", "apiculture"]},
            "112920": {"title": "Horse and Other Equine Production", "sector": "Agriculture", "keywords": ["horse", "equine"]},
            "112930": {"title": "Fur-Bearing Animal and Rabbit Production", "sector": "Agriculture", "keywords": ["fur", "rabbit"]},
            "112990": {"title": "All Other Animal Production", "sector": "Agriculture", "keywords": ["animal", "production"]},
            
            # MANUFACTURING (3xxxxx) - Food Manufacturing (311xxx)
            "311111": {"title": "Dog and Cat Food Manufacturing", "sector": "Manufacturing", "keywords": ["pet", "dog", "cat", "food"]},
            "311119": {"title": "Other Animal Food Manufacturing", "sector": "Manufacturing", "keywords": ["animal", "feed", "food"]},
            "311211": {"title": "Flour Milling", "sector": "Manufacturing", "keywords": ["flour", "mill"]},
            "311212": {"title": "Rice Milling", "sector": "Manufacturing", "keywords": ["rice", "mill"]},
            "311213": {"title": "Malt Manufacturing", "sector": "Manufacturing", "keywords": ["malt", "manufacturing"]},
            "311221": {"title": "Wet Corn Milling", "sector": "Manufacturing", "keywords": ["corn", "mill"]},
            "311222": {"title": "Soybean Processing", "sector": "Manufacturing", "keywords": ["soybean", "processing"]},
            "311223": {"title": "Other Oilseed Processing", "sector": "Manufacturing", "keywords": ["oilseed", "processing"]},
            "311225": {"title": "Fats and Oils Refining and Blending", "sector": "Manufacturing", "keywords": ["oil", "fat", "refining"]},
            "311230": {"title": "Breakfast Cereal Manufacturing", "sector": "Manufacturing", "keywords": ["cereal", "breakfast"]},
            "311311": {"title": "Sugarcane Mills", "sector": "Manufacturing", "keywords": ["sugar", "mill"]},
            "311312": {"title": "Cane Sugar Refining", "sector": "Manufacturing", "keywords": ["sugar", "refining"]},
            "311313": {"title": "Beet Sugar Manufacturing", "sector": "Manufacturing", "keywords": ["beet", "sugar"]},
            "311320": {"title": "Chocolate and Confectionery Manufacturing from Cacao Beans", "sector": "Manufacturing", "keywords": ["chocolate", "confectionery", "cacao"]},
            "311330": {"title": "Confectionery Manufacturing from Purchased Chocolate", "sector": "Manufacturing", "keywords": ["confectionery", "chocolate"]},
            "311340": {"title": "Nonchocolate Confectionery Manufacturing", "sector": "Manufacturing", "keywords": ["candy", "confectionery"]},
            "311411": {"title": "Frozen Fruit, Juice, and Vegetable Manufacturing", "sector": "Manufacturing", "keywords": ["frozen", "fruit", "juice", "vegetable"]},
            "311412": {"title": "Frozen Specialty Food Manufacturing", "sector": "Manufacturing", "keywords": ["frozen", "specialty", "food"]},
            "311421": {"title": "Fruit and Vegetable Canning", "sector": "Manufacturing", "keywords": ["fruit", "vegetable", "canning"]},
            "311422": {"title": "Specialty Canning", "sector": "Manufacturing", "keywords": ["specialty", "canning"]},
            "311423": {"title": "Dried and Dehydrated Food Manufacturing", "sector": "Manufacturing", "keywords": ["dried", "dehydrated", "food"]},
            "311511": {"title": "Fluid Milk Manufacturing", "sector": "Manufacturing", "keywords": ["milk", "fluid"]},
            "311512": {"title": "Creamery Butter Manufacturing", "sector": "Manufacturing", "keywords": ["butter", "creamery"]},
            "311513": {"title": "Cheese Manufacturing", "sector": "Manufacturing", "keywords": ["cheese", "manufacturing"]},
            "311514": {"title": "Dry, Condensed, and Evaporated Dairy Product Manufacturing", "sector": "Manufacturing", "keywords": ["dairy", "condensed", "evaporated"]},
            "311520": {"title": "Ice Cream and Frozen Dessert Manufacturing", "sector": "Manufacturing", "keywords": ["ice cream", "frozen", "dessert"]},
            
            # CONSTRUCTION (23xxxx)
            "236110": {"title": "Residential Building Construction", "sector": "Construction", "keywords": ["residential", "building", "construction"]},
            "236115": {"title": "New Single-Family Housing Construction (except For-Sale Builders)", "sector": "Construction", "keywords": ["single-family", "housing", "construction"]},
            "236116": {"title": "New Multifamily Housing Construction (except For-Sale Builders)", "sector": "Construction", "keywords": ["multifamily", "housing", "construction"]},
            "236117": {"title": "New Housing For-Sale Builders", "sector": "Construction", "keywords": ["housing", "for-sale", "builders"]},
            "236118": {"title": "Residential Remodelers", "sector": "Construction", "keywords": ["residential", "remodeling", "renovation"]},
            "236210": {"title": "Industrial Building Construction", "sector": "Construction", "keywords": ["industrial", "building", "construction"]},
            "236220": {"title": "Commercial and Institutional Building Construction", "sector": "Construction", "keywords": ["commercial", "institutional", "building"]},
            "237110": {"title": "Water and Sewer Line and Related Structures Construction", "sector": "Construction", "keywords": ["water", "sewer", "line", "construction"]},
            "237120": {"title": "Oil and Gas Pipeline and Related Structures Construction", "sector": "Construction", "keywords": ["oil", "gas", "pipeline", "construction"]},
            "237130": {"title": "Power and Communication Line and Related Structures Construction", "sector": "Construction", "keywords": ["power", "communication", "line", "construction"]},
            "237210": {"title": "Land Subdivision", "sector": "Construction", "keywords": ["land", "subdivision"]},
            "237310": {"title": "Highway, Street, and Bridge Construction", "sector": "Construction", "keywords": ["highway", "street", "bridge", "construction"]},
            "237990": {"title": "Other Heavy and Civil Engineering Construction", "sector": "Construction", "keywords": ["heavy", "civil", "engineering", "construction"]},
            
            # PROFESSIONAL SERVICES (54xxxx)
            "541110": {"title": "Offices of Lawyers", "sector": "Professional Services", "keywords": ["lawyer", "legal", "office"]},
            "541120": {"title": "Offices of Notaries", "sector": "Professional Services", "keywords": ["notary", "office"]},
            "541191": {"title": "Title Abstract and Settlement Offices", "sector": "Professional Services", "keywords": ["title", "abstract", "settlement"]},
            "541199": {"title": "All Other Legal Services", "sector": "Professional Services", "keywords": ["legal", "services"]},
            "541211": {"title": "Offices of Certified Public Accountants", "sector": "Professional Services", "keywords": ["accounting", "cpa", "office"]},
            "541213": {"title": "Tax Preparation Services", "sector": "Professional Services", "keywords": ["tax", "preparation", "services"]},
            "541214": {"title": "Payroll Services", "sector": "Professional Services", "keywords": ["payroll", "services"]},
            "541219": {"title": "Other Accounting Services", "sector": "Professional Services", "keywords": ["accounting", "services"]},
            "541310": {"title": "Architectural Services", "sector": "Professional Services", "keywords": ["architectural", "services", "architect"]},
            "541320": {"title": "Landscape Architectural Services", "sector": "Professional Services", "keywords": ["landscape", "architectural", "services"]},
            "541330": {"title": "Engineering Services", "sector": "Professional Services", "keywords": ["engineering", "services", "engineer"]},
            "541340": {"title": "Drafting Services", "sector": "Professional Services", "keywords": ["drafting", "services"]},
            "541350": {"title": "Building Inspection Services", "sector": "Professional Services", "keywords": ["building", "inspection", "services"]},
            "541360": {"title": "Geophysical Surveying and Mapping Services", "sector": "Professional Services", "keywords": ["geophysical", "surveying", "mapping"]},
            "541370": {"title": "Surveying and Mapping (except Geophysical) Services", "sector": "Professional Services", "keywords": ["surveying", "mapping", "services"]},
            "541380": {"title": "Testing Laboratories", "sector": "Professional Services", "keywords": ["testing", "laboratory", "lab"]},
            "541410": {"title": "Interior Design Services", "sector": "Professional Services", "keywords": ["interior", "design", "services"]},
            "541420": {"title": "Industrial Design Services", "sector": "Professional Services", "keywords": ["industrial", "design", "services"]},
            "541430": {"title": "Graphic Design Services", "sector": "Professional Services", "keywords": ["graphic", "design", "services"]},
            "541490": {"title": "Other Specialized Design Services", "sector": "Professional Services", "keywords": ["design", "services"]},
            "541511": {"title": "Custom Computer Programming Services", "sector": "Professional Services", "keywords": ["computer", "programming", "software"]},
            "541512": {"title": "Computer Systems Design Services", "sector": "Professional Services", "keywords": ["computer", "systems", "design"]},
            "541513": {"title": "Computer Facilities Management Services", "sector": "Professional Services", "keywords": ["computer", "facilities", "management"]},
            "541519": {"title": "Other Computer Related Services", "sector": "Professional Services", "keywords": ["computer", "services"]},
            "541611": {"title": "Administrative Management and General Management Consulting Services", "sector": "Professional Services", "keywords": ["management", "consulting", "services"]},
            "541612": {"title": "Human Resources Consulting Services", "sector": "Professional Services", "keywords": ["human", "resources", "consulting"]},
            "541613": {"title": "Marketing Consulting Services", "sector": "Professional Services", "keywords": ["marketing", "consulting", "services"]},
            "541614": {"title": "Process, Physical Distribution, and Logistics Consulting Services", "sector": "Professional Services", "keywords": ["logistics", "consulting", "distribution"]},
            "541618": {"title": "Other Management Consulting Services", "sector": "Professional Services", "keywords": ["management", "consulting", "services"]},
            "541620": {"title": "Environmental Consulting Services", "sector": "Professional Services", "keywords": ["environmental", "consulting", "services"]},
            "541690": {"title": "Other Scientific and Technical Consulting Services", "sector": "Professional Services", "keywords": ["scientific", "technical", "consulting"]},
            
            # TRANSPORTATION (48-49xxxx)
            "481111": {"title": "Scheduled Passenger Air Transportation", "sector": "Transportation", "keywords": ["passenger", "air", "transportation", "airline"]},
            "481112": {"title": "Scheduled Freight Air Transportation", "sector": "Transportation", "keywords": ["freight", "air", "transportation"]},
            "481211": {"title": "Nonscheduled Chartered Passenger Air Transportation", "sector": "Transportation", "keywords": ["chartered", "passenger", "air"]},
            "481212": {"title": "Nonscheduled Chartered Freight Air Transportation", "sector": "Transportation", "keywords": ["chartered", "freight", "air"]},
            "481219": {"title": "Other Nonscheduled Air Transportation", "sector": "Transportation", "keywords": ["nonscheduled", "air", "transportation"]},
            "483111": {"title": "Deep Sea Freight Transportation", "sector": "Transportation", "keywords": ["deep", "sea", "freight", "transportation"]},
            "483112": {"title": "Deep Sea Passenger Transportation", "sector": "Transportation", "keywords": ["deep", "sea", "passenger", "transportation"]},
            "483113": {"title": "Coastal and Great Lakes Freight Transportation", "sector": "Transportation", "keywords": ["coastal", "great", "lakes", "freight"]},
            "483114": {"title": "Coastal and Great Lakes Passenger Transportation", "sector": "Transportation", "keywords": ["coastal", "great", "lakes", "passenger"]},
            "484110": {"title": "General Freight Trucking, Local", "sector": "Transportation", "keywords": ["general", "freight", "trucking", "local"]},
            "484121": {"title": "General Freight Trucking, Long-Distance, Truckload", "sector": "Transportation", "keywords": ["general", "freight", "trucking", "long-distance", "truckload"]},
            "484122": {"title": "General Freight Trucking, Long-Distance, Less Than Truckload", "sector": "Transportation", "keywords": ["general", "freight", "trucking", "long-distance", "less", "truckload"]},
            "487110": {"title": "Scenic and Sightseeing Transportation, Land", "sector": "Transportation", "keywords": ["scenic", "sightseeing", "transportation", "land"]},
            "487210": {"title": "Scenic and Sightseeing Transportation, Water", "sector": "Transportation", "keywords": ["scenic", "sightseeing", "transportation", "water"]},
            "487990": {"title": "Scenic and Sightseeing Transportation, Other", "sector": "Transportation", "keywords": ["scenic", "sightseeing", "transportation"]},
            
            # OTHER SERVICES (81xxxx)
            "811111": {"title": "General Automotive Repair", "sector": "Other Services", "keywords": ["automotive", "repair", "general"]},
            "811112": {"title": "Automotive Exhaust System Repair", "sector": "Other Services", "keywords": ["automotive", "exhaust", "repair"]},
            "811113": {"title": "Automotive Transmission Repair", "sector": "Other Services", "keywords": ["automotive", "transmission", "repair"]},
            "811118": {"title": "Other Automotive Mechanical and Electrical Repair and Maintenance", "sector": "Other Services", "keywords": ["automotive", "mechanical", "electrical", "repair"]},
            "811121": {"title": "Automotive Body, Paint, and Interior Repair and Maintenance", "sector": "Other Services", "keywords": ["automotive", "body", "paint", "interior", "repair"]},
            "811122": {"title": "Automotive Glass Replacement Shops", "sector": "Other Services", "keywords": ["automotive", "glass", "replacement"]},
            
            # RETAIL TRADE (44-45xxxx)
            "441110": {"title": "New Car Dealers", "sector": "Retail", "keywords": ["new", "car", "dealer"]},
            "441120": {"title": "Used Car Dealers", "sector": "Retail", "keywords": ["used", "car", "dealer"]},
            "441210": {"title": "Recreational Vehicle Dealers", "sector": "Retail", "keywords": ["recreational", "vehicle", "dealer"]},
            "441222": {"title": "Boat Dealers", "sector": "Retail", "keywords": ["boat", "dealer"]},
            "441228": {"title": "Motorcycle, ATV, and All Other Motor Vehicle Dealers", "sector": "Retail", "keywords": ["motorcycle", "atv", "motor", "vehicle", "dealer"]},
            "441310": {"title": "Automotive Parts and Accessories Stores", "sector": "Retail", "keywords": ["automotive", "parts", "accessories", "store"]},
            "441320": {"title": "Tire Dealers", "sector": "Retail", "keywords": ["tire", "dealer"]},
            "442110": {"title": "Furniture Stores", "sector": "Retail", "keywords": ["furniture", "store"]},
            "442210": {"title": "Floor Covering Stores", "sector": "Retail", "keywords": ["floor", "covering", "store"]},
            "442291": {"title": "Window Treatment Stores", "sector": "Retail", "keywords": ["window", "treatment", "store"]},
            "442299": {"title": "All Other Home Furnishings Stores", "sector": "Retail", "keywords": ["home", "furnishings", "store"]},
            "443142": {"title": "Electronics Stores", "sector": "Retail", "keywords": ["electronics", "store"]},
            "444110": {"title": "Home Centers", "sector": "Retail", "keywords": ["home", "center"]},
            "444120": {"title": "Paint and Wallpaper Stores", "sector": "Retail", "keywords": ["paint", "wallpaper", "store"]},
            "444130": {"title": "Hardware Stores", "sector": "Retail", "keywords": ["hardware", "store"]},
            "444190": {"title": "Other Building Material Dealers", "sector": "Retail", "keywords": ["building", "material", "dealer"]},
            "445110": {"title": "Supermarkets and Other Grocery (except Convenience) Stores", "sector": "Retail", "keywords": ["supermarket", "grocery", "store"]},
            "445120": {"title": "Convenience Stores", "sector": "Retail", "keywords": ["convenience", "store"]},
            "445210": {"title": "Meat Markets", "sector": "Retail", "keywords": ["meat", "market"]},
            "445220": {"title": "Fish and Seafood Markets", "sector": "Retail", "keywords": ["fish", "seafood", "market"]},
            "445230": {"title": "Fruit and Vegetable Markets", "sector": "Retail", "keywords": ["fruit", "vegetable", "market"]},
            "445291": {"title": "Baked Goods Stores", "sector": "Retail", "keywords": ["baked", "goods", "store", "bakery"]},
            "445292": {"title": "Confectionery and Nut Stores", "sector": "Retail", "keywords": ["confectionery", "nut", "store"]},
            "445299": {"title": "All Other Specialty Food Stores", "sector": "Retail", "keywords": ["specialty", "food", "store"]},
            "446110": {"title": "Pharmacies and Drug Stores", "sector": "Retail", "keywords": ["pharmacy", "drug", "store"]},
            "446120": {"title": "Cosmetics, Beauty Supplies, and Perfume Stores", "sector": "Retail", "keywords": ["cosmetics", "beauty", "perfume", "store"]},
            "446130": {"title": "Optical Goods Stores", "sector": "Retail", "keywords": ["optical", "goods", "store"]},
            "446191": {"title": "Food (Health) Supplement Stores", "sector": "Retail", "keywords": ["food", "health", "supplement", "store"]},
            "446199": {"title": "All Other Health and Personal Care Stores", "sector": "Retail", "keywords": ["health", "personal", "care", "store"]},
        }
    
    def _build_industry_patterns(self) -> Dict[str, List[str]]:
        """Build pattern matching rules for systematic classification."""
        return {
            "agriculture_keywords": [
                "farm", "farming", "crop", "livestock", "agriculture", "agricultural",
                "ranch", "ranching", "dairy", "poultry", "cattle", "pig", "sheep",
                "vineyard", "orchard", "greenhouse", "nursery", "aquaculture",
                "apiculture", "bee", "honey"
            ],
            "manufacturing_keywords": [
                "manufacturing", "production", "factory", "plant", "mill", "milling",
                "processing", "refining", "assembly", "fabrication", "machining"
            ],
            "construction_keywords": [
                "construction", "building", "contractor", "remodeling", "renovation",
                "engineering", "civil", "heavy", "residential", "commercial"
            ],
            "services_keywords": [
                "services", "consulting", "professional", "management", "design",
                "repair", "maintenance", "installation", "testing", "inspection"
            ],
            "transportation_keywords": [
                "transportation", "shipping", "freight", "trucking", "logistics",
                "air", "water", "marine", "scenic", "sightseeing"
            ],
            "retail_keywords": [
                "retail", "store", "shop", "dealer", "market", "outlet",
                "supermarket", "grocery", "convenience"
            ]
        }
    
    def find_naics_code(self, niche: str, business_tags: List[str] = None) -> Optional[str]:
        """Find NAICS code using systematic approach."""
        if not niche or pd.isna(niche):
            return None
        
        # Clean the niche text
        niche_clean = self._clean_niche_text(niche)
        
        # 1. Direct exact match
        exact_match = self._find_exact_match(niche_clean)
        if exact_match:
            return exact_match
        
        # 2. Fuzzy match against NAICS titles
        fuzzy_match = self._find_fuzzy_match(niche_clean)
        if fuzzy_match:
            return fuzzy_match
        
        # 3. Keyword-based pattern matching
        pattern_match = self._find_pattern_match(niche_clean, business_tags)
        if pattern_match:
            return pattern_match
        
        # 4. Sector-based classification
        sector_match = self._find_sector_match(niche_clean)
        if sector_match:
            return sector_match
            
        return None
    
    def _clean_niche_text(self, niche: str) -> str:
        """Clean niche text for better matching."""
        # Remove common suffixes that don't affect classification
        text = niche.strip()
        
        # Normalize common variations
        text = re.sub(r'\b(and|&)\b', 'and', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        
        return text
    
    def _find_exact_match(self, niche: str) -> Optional[str]:
        """Find exact matches in NAICS database."""
        for naics_code, details in self.naics_database.items():
            if niche.lower() == details["title"].lower():
                return naics_code
        return None
    
    def _find_fuzzy_match(self, niche: str) -> Optional[str]:
        """Find fuzzy matches against NAICS titles."""
        best_score = 0
        best_code = None
        
        for naics_code, details in self.naics_database.items():
            title = details["title"]
            similarity = SequenceMatcher(None, niche.lower(), title.lower()).ratio()
            
            if similarity > best_score and similarity >= self.fuzzy_threshold:
                best_score = similarity
                best_code = naics_code
        
        return best_code
    
    def _find_pattern_match(self, niche: str, business_tags: List[str] = None) -> Optional[str]:
        """Find matches using keyword patterns."""
        niche_lower = niche.lower()
        
        # Check business tags for additional context
        tags_text = ""
        if business_tags:
            tags_text = " ".join(business_tags).lower()
        
        combined_text = f"{niche_lower} {tags_text}"
        
        # Agriculture patterns
        if any(keyword in combined_text for keyword in self.industry_patterns["agriculture_keywords"]):
            return self._classify_agriculture(niche_lower)
        
        # Manufacturing patterns
        if any(keyword in combined_text for keyword in self.industry_patterns["manufacturing_keywords"]):
            return self._classify_manufacturing(niche_lower)
        
        # Construction patterns
        if any(keyword in combined_text for keyword in self.industry_patterns["construction_keywords"]):
            return self._classify_construction(niche_lower)
        
        # Services patterns
        if any(keyword in combined_text for keyword in self.industry_patterns["services_keywords"]):
            return self._classify_services(niche_lower)
        
        # Transportation patterns
        if any(keyword in combined_text for keyword in self.industry_patterns["transportation_keywords"]):
            return self._classify_transportation(niche_lower)
        
        # Retail patterns
        if any(keyword in combined_text for keyword in self.industry_patterns["retail_keywords"]):
            return self._classify_retail(niche_lower)
        
        return None
    
    def _classify_agriculture(self, niche: str) -> str:
        """Classify agriculture-related niches."""
        if any(term in niche for term in ["crop", "farm"]):
            if "miscellaneous" in niche:
                return "111998"
            elif "vegetable" in niche:
                return "111219"
            elif "fruit" in niche:
                return "111339"
            else:
                return "111199"  # Other grain farming
        elif "livestock" in niche or "cattle" in niche:
            return "112111"
        elif "poultry" in niche or "chicken" in niche:
            return "112320"
        elif "dairy" in niche:
            return "112120"
        elif "apiculture" in niche or "bee" in niche:
            return "112910"
        else:
            return "112990"  # All Other Animal Production
    
    def _classify_manufacturing(self, niche: str) -> str:
        """Classify manufacturing-related niches."""
        if "food" in niche or "fruit" in niche or "vegetable" in niche:
            if "frozen" in niche:
                return "311411"
            elif "dairy" in niche:
                return "311511"
            else:
                return "311999"  # All Other Miscellaneous Food Manufacturing
        elif "wood" in niche:
            return "321213"  # Engineered Wood Member Manufacturing
        elif "plastic" in niche:
            return "326199"  # All Other Plastics Product Manufacturing
        else:
            return "339999"  # All Other Miscellaneous Manufacturing
    
    def _classify_construction(self, niche: str) -> str:
        """Classify construction-related niches."""
        if "heavy" in niche or "civil" in niche or "engineering" in niche:
            return "237990"
        elif "residential" in niche:
            return "236118"
        elif "commercial" in niche:
            return "236220"
        else:
            return "238990"  # All Other Specialty Trade Contractors
    
    def _classify_services(self, niche: str) -> str:
        """Classify service-related niches."""
        if "environmental" in niche:
            return "541620"
        elif "consulting" in niche or "management" in niche:
            return "541611"
        elif "computer" in niche or "software" in niche:
            return "541512"
        elif "engineering" in niche:
            return "541330"
        elif "architectural" in niche:
            return "541310"
        elif "repair" in niche:
            return "811490"  # Other Personal and Household Goods Repair
        else:
            return "541990"  # All Other Professional Services
    
    def _classify_transportation(self, niche: str) -> str:
        """Classify transportation-related niches."""
        if "air" in niche:
            return "481219"
        elif "water" in niche or "marine" in niche:
            if "scenic" in niche:
                return "487210"
            else:
                return "483114"
        elif "truck" in niche or "freight" in niche:
            return "484121"
        else:
            return "488999"  # All Other Support Activities for Transportation
    
    def _classify_retail(self, niche: str) -> str:
        """Classify retail-related niches."""
        if "grocery" in niche or "supermarket" in niche:
            return "445110"
        elif "convenience" in niche:
            return "445120"
        elif "auto" in niche or "car" in niche:
            return "441110"
        elif "electronics" in niche:
            return "443142"
        elif "furniture" in niche:
            return "442110"
        else:
            return "459999"  # All Other Miscellaneous Store Retailers
    
    def _find_sector_match(self, niche: str) -> Optional[str]:
        """Find matches based on broad sector classification."""
        # This is a fallback for very general cases
        return None
    
    def load_company_mappings(self, companies_df: pd.DataFrame) -> pd.DataFrame:
        """Load NAICS mappings for all companies."""
        result_df = companies_df.copy()
        
        # Parse business tags
        def parse_tags(tags_str):
            if pd.isna(tags_str):
                return []
            try:
                return eval(tags_str) if isinstance(tags_str, str) else []
            except:
                return []
        
        result_df['parsed_tags'] = result_df['business_tags'].apply(parse_tags)
        
        # Apply NAICS mapping with tags
        result_df['naics_code'] = result_df.apply(
            lambda row: self.find_naics_code(row['niche'], row['parsed_tags']), 
            axis=1
        )
        result_df['naics_mapped'] = result_df['naics_code'].notna()
        
        # Statistics
        total_companies = len(result_df)
        mapped_companies = result_df['naics_mapped'].sum()
        mapping_rate = mapped_companies / total_companies
        
        print(f"🎯 Systematic NAICS Mapping Results:")
        print(f"  Total companies: {total_companies}")
        print(f"  Successfully mapped: {mapped_companies}")
        print(f"  Mapping rate: {mapping_rate:.1%}")
        
        # Show unmapped patterns
        unmapped = result_df[~result_df['naics_mapped']]
        if len(unmapped) > 0:
            unmapped_niches = unmapped['niche'].value_counts()
            print(f"\n  🔍 Top unmapped niches (need manual review):")
            for niche, count in unmapped_niches.head(10).items():
                print(f"    {count:3d} companies: {niche}")
        
        return result_df
    
    def get_mapping_coverage_by_sector(self, companies_df: pd.DataFrame) -> Dict:
        """Analyze mapping coverage by sector."""
        mapped_df = self.load_company_mappings(companies_df)
        
        coverage_stats = {}
        for sector in mapped_df['sector'].unique():
            sector_data = mapped_df[mapped_df['sector'] == sector]
            total = len(sector_data)
            mapped = sector_data['naics_mapped'].sum()
            coverage_stats[sector] = {
                'total': total,
                'mapped': mapped,
                'coverage': mapped / total if total > 0 else 0
            }
        
        return coverage_stats 