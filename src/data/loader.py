"""
Data loading utilities for reading and processing CSV files
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and initial processing of CSV data"""
    
    @staticmethod
    def load_company_data(file_path: Path, sample_size: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Load company data from CSV file
        
        Args:
            file_path: Path to company data CSV
            sample_size: Number of samples to load (None for all)
            
        Returns:
            DataFrame with company data or None if failed
        """
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Take sample if specified
            if sample_size is not None:
                df = df.head(sample_size)
                logger.info(f"Loaded {len(df)} companies (sample of {sample_size})")
            else:
                logger.info(f"Loaded {len(df)} companies")
                
            # Basic data validation
            required_columns = ['description', 'business_tags', 'sector', 'category', 'niche']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return None
                
            # Add index if not present
            if 'company_id' not in df.columns:
                df['company_id'] = range(len(df))
                
            return df
            
        except Exception as e:
            logger.error(f"Failed to load company data from {file_path}: {e}")
            return None
    
    @staticmethod
    def load_taxonomy_labels(file_path: Path) -> Optional[List[str]]:
        """
        Load insurance taxonomy labels from CSV file
        
        Args:
            file_path: Path to taxonomy CSV
            
        Returns:
            List of taxonomy labels or None if failed
        """
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Extract labels (assuming single column named 'label')
            if 'label' not in df.columns:
                logger.error("Taxonomy file must have a 'label' column")
                return None
                
            labels = df['label'].dropna().tolist()
            logger.info(f"Loaded {len(labels)} taxonomy labels")
            
            return labels
            
        except Exception as e:
            logger.error(f"Failed to load taxonomy labels from {file_path}: {e}")
            return None
    
    @staticmethod
    def get_company_business_tags(df: pd.DataFrame) -> Dict[int, List[str]]:
        """
        Extract business tags for all companies
        
        Args:
            df: DataFrame with company data
            
        Returns:
            Dictionary mapping company_id to list of business tags
        """
        from ..utils.text_utils import TextProcessor
        
        business_tags_dict = {}
        
        for idx, row in df.iterrows():
            company_id = row.get('company_id', idx)
            tags_string = row.get('business_tags', '')
            
            # Parse business tags
            tags = TextProcessor.parse_business_tags(tags_string)
            business_tags_dict[company_id] = tags
            
        logger.info(f"Extracted business tags for {len(business_tags_dict)} companies")
        return business_tags_dict
    
    @staticmethod
    def get_company_metadata(df: pd.DataFrame) -> Dict[int, Dict[str, str]]:
        """
        Extract metadata for all companies
        
        Args:
            df: DataFrame with company data
            
        Returns:
            Dictionary mapping company_id to metadata dict
        """
        metadata_dict = {}
        
        for idx, row in df.iterrows():
            company_id = row.get('company_id', idx)
            
            metadata = {
                'description': row.get('description', ''),
                'sector': row.get('sector', ''),
                'category': row.get('category', ''),
                'niche': row.get('niche', '')
            }
            
            metadata_dict[company_id] = metadata
            
        logger.info(f"Extracted metadata for {len(metadata_dict)} companies")
        return metadata_dict
    
    @staticmethod
    def create_sample_output_format(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create output DataFrame with insurance_label column
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with insurance_label column added
        """
        output_df = df.copy()
        
        # Add empty insurance_label column
        output_df['insurance_label'] = ''
        
        # Reorder columns to put insurance_label at the end
        columns = [col for col in output_df.columns if col != 'insurance_label']
        columns.append('insurance_label')
        output_df = output_df[columns]
        
        return output_df
    
    @staticmethod
    def save_results(df: pd.DataFrame, output_path: Path) -> bool:
        """
        Save results to CSV file
        
        Args:
            df: DataFrame with results
            output_path: Path to save results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV
            df.to_csv(output_path, index=False)
            
            logger.info(f"Results saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save results to {output_path}: {e}")
            return False 