"""
Main classifier for insurance company classification using business tags
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

from .embedder import EmbeddingModel
from ..utils.text_utils import TextProcessor
from ..utils.cache_manager import CacheManager

logger = logging.getLogger(__name__)

class BusinessTagsClassifier:
    """Classifier that uses business tags for insurance label classification"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the classifier
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.embedder = EmbeddingModel(model_name)
        self.taxonomy_labels = None
        self.taxonomy_embeddings = None
        self.label_to_index = {}
        self.index_to_label = {}
        
    def load_taxonomy(self, labels: List[str], cache_path: Optional[Path] = None) -> bool:
        """
        Load and embed taxonomy labels
        
        Args:
            labels: List of insurance taxonomy labels
            cache_path: Path to cache embeddings (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.taxonomy_labels = labels
            
            # Create label mappings
            self.label_to_index = {label: i for i, label in enumerate(labels)}
            self.index_to_label = {i: label for i, label in enumerate(labels)}
            
            # Try to load from cache first
            if cache_path and CacheManager.cache_exists(cache_path):
                logger.info("Loading taxonomy embeddings from cache...")
                cached_embeddings, cached_labels = CacheManager.load_embeddings(cache_path)
                
                if cached_embeddings is not None and cached_labels == labels:
                    self.taxonomy_embeddings = cached_embeddings
                    logger.info("Taxonomy embeddings loaded from cache successfully")
                    return True
                else:
                    logger.warning("Cache mismatch, regenerating embeddings...")
            
            # Generate embeddings for taxonomy labels
            logger.info("Generating taxonomy embeddings...")
            self.taxonomy_embeddings = self.embedder.embed_texts(labels)
            
            # Save to cache if path provided
            if cache_path:
                CacheManager.save_embeddings(self.taxonomy_embeddings, labels, cache_path)
                
            logger.info(f"Successfully loaded {len(labels)} taxonomy labels")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load taxonomy: {e}")
            return False
    
    def classify_company(self, business_tags: List[str], 
                        top_k: int = 5, 
                        similarity_threshold: float = 0.3,
                        keyword_boost: float = 0.2,
                        sector: str = None,
                        category: str = None,
                        niche: str = None) -> List[Tuple[str, float]]:
        """
        Classify a company based on its business tags
        
        Args:
            business_tags: List of business tags for the company
            top_k: Number of top labels to return
            similarity_threshold: Minimum similarity score for inclusion
            keyword_boost: Additional score for direct keyword matches
            
        Returns:
            List of (label, confidence_score) tuples
        """
        if not self.taxonomy_labels or self.taxonomy_embeddings is None:
            logger.error("Taxonomy not loaded")
            return []
            
        if not business_tags:
            logger.warning("No business tags provided")
            return []
            
        try:
            # Combine all business tags into a single text
            combined_tags = " ".join(business_tags)
            
            # Generate embedding for combined tags
            tag_embedding = self.embedder.embed_single_text(combined_tags)
            
            if len(tag_embedding) == 0:
                logger.warning("Failed to generate embedding for business tags")
                return []
            
            # Find most similar taxonomy labels
            similarities = self.embedder.find_most_similar(
                tag_embedding, 
                self.taxonomy_embeddings, 
                top_k=top_k
            )
            
            # Process results and apply keyword matching boost
            results = []
            for label_idx, similarity_score in similarities:
                if similarity_score < similarity_threshold:
                    continue
                    
                label = self.index_to_label[label_idx]
                
                # Apply keyword matching boost
                keyword_score = self._calculate_keyword_boost(business_tags, label)
                
                # Apply sector-based validation
                sector_penalty = self._calculate_sector_penalty(label, sector, category, niche)
                
                # Calculate final score with all adjustments
                final_score = similarity_score + (keyword_score * keyword_boost) + sector_penalty
                
                results.append((label, final_score))
            
            # Sort by final score and return
            results.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"Classified company with {len(results)} matching labels")
            return results
            
        except Exception as e:
            logger.error(f"Failed to classify company: {e}")
            return []
    
    def _calculate_keyword_boost(self, business_tags: List[str], taxonomy_label: str) -> float:
        """
        Calculate keyword matching boost between business tags and taxonomy label
        
        Args:
            business_tags: List of business tags
            taxonomy_label: Taxonomy label to compare against
            
        Returns:
            Keyword matching score (0-1)
        """
        try:
            # Combine business tags
            combined_tags = " ".join(business_tags)
            
            # Calculate keyword overlap
            overlap_score = TextProcessor.calculate_keyword_overlap(combined_tags, taxonomy_label)
            
            # Also check for direct substring matches
            normalized_tags = TextProcessor.normalize_for_matching(combined_tags)
            normalized_label = TextProcessor.normalize_for_matching(taxonomy_label)
            
            # Check if any tag words appear in the label
            tag_words = set(normalized_tags.split())
            label_words = set(normalized_label.split())
            
            direct_matches = tag_words.intersection(label_words)
            direct_match_score = len(direct_matches) / max(len(tag_words), 1)
            
            # Return the higher of the two scores
            return max(overlap_score, direct_match_score)
            
        except Exception as e:
            logger.error(f"Failed to calculate keyword boost: {e}")
            return 0.0
    
    def _calculate_sector_penalty(self, insurance_label: str, sector: str, category: str, niche: str) -> float:
        """
        Calculate sector-based penalty/boost for insurance label alignment
        
        Args:
            insurance_label: Insurance taxonomy label
            sector: Business sector (e.g., "Manufacturing", "Services")
            category: Business category 
            niche: Business niche
            
        Returns:
            Penalty/boost score (-0.3 to +0.1)
        """
        if not sector:
            return 0.0
            
        try:
            # Normalize inputs
            sector = sector.lower().strip() if sector else ""
            label = insurance_label.lower().strip()
            
            # Define sector-based validation rules
            sector_keywords = {
                'manufacturing': ['production', 'manufacturing', 'factory', 'processing', 'assembly'],
                'services': ['services', 'consulting', 'repair', 'maintenance', 'installation', 'design'],
                'wholesale': ['wholesale', 'distribution', 'supply', 'trading'],
                'retail': ['retail', 'store', 'shop', 'sales'],
                'government': ['government', 'public', 'municipal', 'state'],
                'education': ['education', 'school', 'training', 'academic'],
                'non profit': ['charity', 'foundation', 'volunteer', 'community']
            }
            
            # Find which sector the insurance label belongs to
            label_sector = None
            for sector_name, keywords in sector_keywords.items():
                if any(keyword in label for keyword in keywords):
                    label_sector = sector_name
                    break
            
            # If we can't determine the label's sector, no penalty/boost
            if not label_sector:
                return 0.0
            
            # Compare business sector with label sector
            if sector in label_sector or label_sector in sector:
                # Perfect sector match - small boost
                return 0.05
            elif self._are_related_sectors(sector, label_sector):
                # Related sectors - neutral
                return 0.0
            else:
                # Mismatched sectors - penalty
                penalty_strength = self._get_penalty_strength(sector, label_sector)
                return penalty_strength
                
        except Exception as e:
            logger.error(f"Failed to calculate sector penalty: {e}")
            return 0.0
    
    def _are_related_sectors(self, sector1: str, sector2: str) -> bool:
        """Check if two sectors are related/compatible"""
        
        # Define related sector groups
        related_groups = [
            ['manufacturing', 'wholesale'],  # Manufacturers often wholesale
            ['services', 'retail'],          # Services and retail often overlap
            ['manufacturing', 'services'],   # Some manufacturing includes services
        ]
        
        sector1 = sector1.lower()
        sector2 = sector2.lower()
        
        for group in related_groups:
            if sector1 in group and sector2 in group:
                return True
                
        return False
    
    def _get_penalty_strength(self, business_sector: str, label_sector: str) -> float:
        """
        Get penalty strength based on how mismatched the sectors are
        
        Returns:
            Negative float representing penalty strength
        """
        
        # Heavy penalties for completely unrelated sectors
        heavy_mismatches = [
            ('manufacturing', 'education'),
            ('manufacturing', 'non profit'), 
            ('services', 'education'),
            ('wholesale', 'education'),
            ('retail', 'government'),
        ]
        
        business_sector = business_sector.lower()
        label_sector = label_sector.lower()
        
        # Check for heavy mismatches
        for b_sector, l_sector in heavy_mismatches:
            if (business_sector == b_sector and label_sector == l_sector) or \
               (business_sector == l_sector and label_sector == b_sector):
                return -0.3  # Heavy penalty
        
        # Default moderate penalty for other mismatches
        return -0.15
    
    def classify_multiple_companies(self, companies_tags: Dict[int, List[str]], 
                                   companies_metadata: Dict[int, Dict[str, str]] = None,
                                   **kwargs) -> Dict[int, List[Tuple[str, float]]]:
        """
        Classify multiple companies
        
        Args:
            companies_tags: Dictionary mapping company_id to business tags
            companies_metadata: Dictionary mapping company_id to metadata (sector, category, niche)
            **kwargs: Additional parameters for classify_company
            
        Returns:
            Dictionary mapping company_id to classification results
        """
        results = {}
        
        for company_id, business_tags in companies_tags.items():
            try:
                # Get metadata for this company if available
                metadata = companies_metadata.get(company_id, {}) if companies_metadata else {}
                
                classification = self.classify_company(
                    business_tags, 
                    sector=metadata.get('sector'),
                    category=metadata.get('category'),
                    niche=metadata.get('niche'),
                    **kwargs
                )
                results[company_id] = classification
                
            except Exception as e:
                logger.error(f"Failed to classify company {company_id}: {e}")
                results[company_id] = []
        
        logger.info(f"Classified {len(results)} companies")
        return results
    
    def get_classification_summary(self, classification_results: Dict[int, List[Tuple[str, float]]]) -> Dict[str, int]:
        """
        Get summary statistics from classification results
        
        Args:
            classification_results: Results from classify_multiple_companies
            
        Returns:
            Dictionary with summary statistics
        """
        total_companies = len(classification_results)
        companies_with_labels = sum(1 for results in classification_results.values() if results)
        total_labels_assigned = sum(len(results) for results in classification_results.values())
        
        # Count unique labels assigned
        unique_labels = set()
        for results in classification_results.values():
            for label, _ in results:
                unique_labels.add(label)
        
        return {
            "total_companies": total_companies,
            "companies_with_labels": companies_with_labels,
            "companies_without_labels": total_companies - companies_with_labels,
            "total_labels_assigned": total_labels_assigned,
            "unique_labels_used": len(unique_labels),
            "avg_labels_per_company": total_labels_assigned / total_companies if total_companies > 0 else 0
        }
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the classifier
        
        Returns:
            Dictionary with classifier information
        """
        embedder_info = self.embedder.get_model_info()
        
        return {
            "embedder_info": embedder_info,
            "taxonomy_loaded": self.taxonomy_labels is not None,
            "num_taxonomy_labels": len(self.taxonomy_labels) if self.taxonomy_labels else 0,
            "embeddings_generated": self.taxonomy_embeddings is not None
        } 