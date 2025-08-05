"""
Export Tools for Insurance Classification System
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any


class ExportManager:
    """
    Utility class for exporting labeled data and creating training datasets
    """
    
    @staticmethod
    def combine_labeled_sessions(data_dir: str = 'data') -> Dict[str, Any]:
        """
        Combine all labeled session files into one dataset
        
        Args:
            data_dir: Directory containing labeled session files
            
        Returns:
            Combined labeled dataset
        """
        data_path = Path(data_dir)
        
        # Find all labeled session files
        session_files = list(data_path.glob('labeled_companies_*.json'))
        
        if not session_files:
            return {
                'error': 'No labeled session files found',
                'files_checked': str(data_path)
            }
        
        all_labeled = []
        session_info = []
        
        for file_path in session_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                session_info.append(data.get('session_info', {}))
                labeled_companies = data.get('labeled_companies', [])
                
                # Add source file info to each record
                for record in labeled_companies:
                    record['source_file'] = file_path.name
                    
                all_labeled.extend(labeled_companies)
                
            except Exception as e:
                print(f"⚠️  Error reading {file_path}: {e}")
        
        return {
            'total_sessions': len(session_files),
            'total_labeled_companies': len(all_labeled),
            'session_files': [f.name for f in session_files],
            'session_info': session_info,
            'labeled_companies': all_labeled
        }
    
    @staticmethod
    def create_training_dataset(companies_df: pd.DataFrame, 
                               labeled_data: Dict[str, Any],
                               output_path: str = None) -> pd.DataFrame:
        """
        Create a training dataset from labeled companies
        
        Args:
            companies_df: Original company dataframe
            labeled_data: Combined labeled data from sessions
            output_path: Optional path to save the dataset
            
        Returns:
            Training dataset DataFrame
        """
        if 'labeled_companies' not in labeled_data:
            raise ValueError("Invalid labeled data format")
        
        training_records = []
        
        for item in labeled_data['labeled_companies']:
            company_idx = item['company_index']
            
            if company_idx >= len(companies_df):
                print(f"⚠️  Skipping invalid company index: {company_idx}")
                continue
                
            company = companies_df.iloc[company_idx]
            
            # Create records for each label assigned to this company
            for label in item['labels']:
                record = {
                    'company_index': company_idx,
                    'description': company['description'],
                    'business_tags': company['business_tags'],
                    'sector': company['sector'],
                    'category': company['category'],
                    'niche': company['niche'],
                    'insurance_label': label,
                    'confidence': item.get('confidence', 1.0),
                    'labeling_method': item.get('method', 'unknown'),
                    'source_session': item.get('source_file', 'unknown')
                }
                training_records.append(record)
        
        training_df = pd.DataFrame(training_records)
        
        if output_path:
            training_df.to_csv(output_path, index=False)
            print(f"💾 Training dataset saved to: {output_path}")
        
        return training_df
    
    @staticmethod
    def generate_labeling_report(labeled_data: Dict[str, Any], 
                                companies_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive report on labeling progress
        
        Args:
            labeled_data: Combined labeled data
            companies_df: Original company dataframe
            
        Returns:
            Labeling progress report
        """
        if 'labeled_companies' not in labeled_data:
            return {'error': 'Invalid labeled data format'}
        
        labeled_companies = labeled_data['labeled_companies']
        
        # Basic statistics
        total_companies = len(companies_df)
        unique_labeled_companies = len(set(item['company_index'] for item in labeled_companies))
        total_labels_assigned = len(labeled_companies)
        
        # Label distribution
        label_counts = {}
        confidence_scores = []
        method_counts = {}
        
        for item in labeled_companies:
            for label in item['labels']:
                label_counts[label] = label_counts.get(label, 0) + 1
            
            confidence_scores.append(item.get('confidence', 1.0))
            method = item.get('method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
        
        # Session analysis
        session_analysis = {}
        if 'session_info' in labeled_data:
            for session in labeled_data['session_info']:
                approach = session.get('approach', 'unknown')
                session_analysis[approach] = session_analysis.get(approach, 0) + 1
        
        report = {
            'overview': {
                'total_companies_in_dataset': total_companies,
                'unique_labeled_companies': unique_labeled_companies,
                'total_label_assignments': total_labels_assigned,
                'labeling_progress': unique_labeled_companies / total_companies,
                'avg_labels_per_company': total_labels_assigned / unique_labeled_companies if unique_labeled_companies > 0 else 0
            },
            'label_distribution': {
                'most_common_labels': sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:10],
                'total_unique_labels_used': len(label_counts),
                'label_coverage': len(label_counts)  # Would need taxonomy size for percentage
            },
            'quality_metrics': {
                'avg_confidence': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
                'high_confidence_labels': sum(1 for score in confidence_scores if score > 0.8),
                'method_breakdown': method_counts
            },
            'session_analysis': session_analysis,
            'recommendations': ExportManager._generate_recommendations(labeled_data, companies_df)
        }
        
        return report
    
    @staticmethod
    def _generate_recommendations(labeled_data: Dict[str, Any], 
                                 companies_df: pd.DataFrame) -> List[str]:
        """Generate recommendations based on labeling progress"""
        recommendations = []
        
        labeled_companies = labeled_data.get('labeled_companies', [])
        unique_labeled = len(set(item['company_index'] for item in labeled_companies))
        total_companies = len(companies_df)
        
        progress = unique_labeled / total_companies if total_companies > 0 else 0
        
        if progress < 0.01:
            recommendations.append("🚀 Just getting started! Try to label 50-100 companies to build initial patterns")
        elif progress < 0.05:
            recommendations.append("📈 Good progress! Focus on diverse company types to improve coverage")
        elif progress < 0.1:
            recommendations.append("💪 Solid foundation! Consider running validation to check consistency")
        else:
            recommendations.append("🎯 Excellent progress! Ready for batch classification of remaining companies")
        
        # Check confidence distribution
        confidence_scores = [item.get('confidence', 1.0) for item in labeled_companies]
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            if avg_confidence < 0.6:
                recommendations.append("⚠️  Low average confidence - consider adjusting similarity weights")
            elif avg_confidence > 0.8:
                recommendations.append("✅ High confidence scores - your approach is working well!")
        
        # Check method distribution
        method_counts = {}
        for item in labeled_companies:
            method = item.get('method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
        
        if method_counts.get('custom', 0) > method_counts.get('ai_suggestion', 0):
            recommendations.append("🤔 Many custom labels - AI suggestions might need weight adjustment")
        
        return recommendations
    
    @staticmethod
    def export_for_external_tools(training_df: pd.DataFrame, 
                                 format_type: str = 'huggingface',
                                 output_dir: str = 'data/exports') -> Dict[str, str]:
        """
        Export training data in formats suitable for external ML tools
        
        Args:
            training_df: Training dataset
            format_type: Export format ('huggingface', 'sklearn', 'csv')
            output_dir: Output directory
            
        Returns:
            Dictionary with export file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exports = {}
        
        if format_type == 'huggingface':
            # Format for Hugging Face datasets
            hf_data = training_df[['description', 'business_tags', 'insurance_label']].copy()
            hf_data['text'] = hf_data['description'] + ' ' + hf_data['business_tags'].astype(str)
            hf_data = hf_data[['text', 'insurance_label']].rename(columns={'insurance_label': 'label'})
            
            hf_path = output_path / f'huggingface_dataset_{timestamp}.csv'
            hf_data.to_csv(hf_path, index=False)
            exports['huggingface'] = str(hf_path)
            
        elif format_type == 'sklearn':
            # Format for sklearn classifiers
            sklearn_data = training_df.copy()
            sklearn_data['combined_features'] = (
                sklearn_data['description'].astype(str) + ' ' + 
                sklearn_data['business_tags'].astype(str) + ' ' +
                sklearn_data['sector'].astype(str)
            )
            
            sklearn_path = output_path / f'sklearn_dataset_{timestamp}.csv'
            sklearn_data.to_csv(sklearn_path, index=False)
            exports['sklearn'] = str(sklearn_path)
            
        # Always export full CSV
        csv_path = output_path / f'full_training_dataset_{timestamp}.csv'
        training_df.to_csv(csv_path, index=False)
        exports['csv'] = str(csv_path)
        
        return exports 