"""
NAICS-Based Validation Framework
Uses NAICS codes as external ground truth to validate insurance taxonomy classifications.
This addresses the task requirement for "real-world effectiveness" validation.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import os
import sys

# Add src to path for imports
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


class NAICSValidator:
    """Validates insurance taxonomy classifications against NAICS ground truth."""
    
    def __init__(self, 
                 companies_with_naics_file="data/output/companies_with_naics.csv",
                 insurance_naics_mappings_file="data/cache/naics_mappings/insurance_to_naics_mappings.csv"):
        """Initialize with pre-computed NAICS mappings."""
        self.companies_with_naics = None
        self.insurance_naics_mappings = None
        self.naics_to_insurance_dict = {}
        self.validation_pairs = []
        
        self.load_data(companies_with_naics_file, insurance_naics_mappings_file)
        self.create_naics_to_insurance_lookup()
        
    def load_data(self, companies_file, mappings_file):
        """Load companies with NAICS codes and insurance-NAICS mappings."""
        print("Loading companies with NAICS codes...")
        self.companies_with_naics = pd.read_csv(companies_file)
        print(f"Loaded {len(self.companies_with_naics)} companies with NAICS codes")
        
        print("Loading insurance to NAICS mappings...")
        self.insurance_naics_mappings = pd.read_csv(mappings_file)
        print(f"Loaded {len(self.insurance_naics_mappings)} insurance label mappings")
        
        # Fix data type issue: convert NAICS codes to strings for consistent matching
        self.companies_with_naics['naics_code'] = self.companies_with_naics['naics_code'].astype(str)
        self.insurance_naics_mappings['best_naics_code'] = self.insurance_naics_mappings['best_naics_code'].astype(str)
        print("Converted NAICS codes to strings for consistent matching")
        
    def create_naics_to_insurance_lookup(self):
        """Create lookup from NAICS codes to insurance labels."""
        print("Creating NAICS → Insurance label lookup...")
        
        # Only use high-confidence mappings for validation (>0.6 similarity)
        high_confidence_mappings = self.insurance_naics_mappings[
            self.insurance_naics_mappings['best_similarity_score'] > 0.6
        ]
        
        for _, row in high_confidence_mappings.iterrows():
            naics_code = str(row['best_naics_code'])  # Ensure string type
            insurance_label = row['insurance_label']
            confidence_score = row['best_similarity_score']
            
            if naics_code not in self.naics_to_insurance_dict:
                self.naics_to_insurance_dict[naics_code] = []
            
            self.naics_to_insurance_dict[naics_code].append({
                'insurance_label': insurance_label,
                'confidence_score': confidence_score
            })
        
        print(f"Created lookup for {len(self.naics_to_insurance_dict)} NAICS codes")
        
        # Debug: show overlap
        company_naics = set(self.companies_with_naics['naics_code'].unique())
        insurance_naics = set(self.naics_to_insurance_dict.keys())
        overlap = company_naics.intersection(insurance_naics)
        print(f"Company NAICS codes: {len(company_naics)}")
        print(f"Insurance NAICS codes (high conf): {len(insurance_naics)}")
        print(f"Overlap found: {len(overlap)} NAICS codes")
        
        if len(overlap) > 0:
            print(f"Sample overlapping codes: {list(overlap)[:5]}")
        else:
            print("No overlap found - adjusting confidence threshold...")
            # Try lower confidence threshold
            medium_confidence_mappings = self.insurance_naics_mappings[
                self.insurance_naics_mappings['best_similarity_score'] > 0.5
            ]
            
            for _, row in medium_confidence_mappings.iterrows():
                naics_code = str(row['best_naics_code'])
                insurance_label = row['insurance_label']
                confidence_score = row['best_similarity_score']
                
                if naics_code not in self.naics_to_insurance_dict:
                    self.naics_to_insurance_dict[naics_code] = []
                
                self.naics_to_insurance_dict[naics_code].append({
                    'insurance_label': insurance_label,
                    'confidence_score': confidence_score
                })
            
            # Recheck overlap
            insurance_naics = set(self.naics_to_insurance_dict.keys())
            overlap = company_naics.intersection(insurance_naics)
            print(f"With lower threshold - Overlap: {len(overlap)} NAICS codes")
            if len(overlap) > 0:
                print(f"Sample overlapping codes: {list(overlap)[:5]}")
        
    def create_validation_ground_truth(self):
        """Create ground truth pairs: (Company, Expected_Insurance_Label) based on NAICS."""
        print("Creating validation ground truth using NAICS bridge...")
        
        validation_pairs = []
        companies_with_ground_truth = 0
        
        for idx, company in self.companies_with_naics.iterrows():
            company_naics = str(company['naics_code'])  # Ensure string type
            
            # Check if this NAICS code has corresponding insurance labels
            if company_naics in self.naics_to_insurance_dict:
                expected_labels = self.naics_to_insurance_dict[company_naics]
                
                for label_info in expected_labels:
                    validation_pairs.append({
                        'company_index': idx,
                        'company_description': company['description'],
                        'company_sector': company['sector'],
                        'company_category': company['category'],
                        'company_niche': company['niche'],
                        'company_naics_code': company_naics,
                        'company_naics_description': company['naics_description'],
                        'expected_insurance_label': label_info['insurance_label'],
                        'naics_mapping_confidence': label_info['confidence_score'],
                        'company_naics_confidence': company['naics_similarity_score']
                    })
                
                companies_with_ground_truth += 1
        
        self.validation_pairs = pd.DataFrame(validation_pairs)
        
        print(f"Created {len(self.validation_pairs)} validation pairs")
        print(f"Companies with ground truth: {companies_with_ground_truth}")
        
        if len(self.validation_pairs) > 0:
            print(f"Unique insurance labels in ground truth: {self.validation_pairs['expected_insurance_label'].nunique()}")
        else:
            print("No validation pairs created - checking data...")
            # Debug information
            print("Sample company NAICS codes:", self.companies_with_naics['naics_code'].head(5).tolist())
            print("Sample insurance NAICS codes:", list(self.naics_to_insurance_dict.keys())[:5])
        
        return self.validation_pairs
    
    def validate_classifications(self, predicted_classifications):
        """
        Validate predicted classifications against NAICS ground truth.
        
        Args:
            predicted_classifications: DataFrame with columns ['company_index', 'predicted_labels']
                where predicted_labels is a list of insurance labels
        """
        print("Validating classifications against NAICS ground truth...")
        
        if self.validation_pairs is None or len(self.validation_pairs) == 0:
            self.create_validation_ground_truth()
        
        validation_results = []
        
        # Create lookup for predicted classifications
        predictions_dict = {}
        for _, row in predicted_classifications.iterrows():
            predictions_dict[row['company_index']] = row['predicted_labels']
        
        # Validate each ground truth pair
        for _, ground_truth in self.validation_pairs.iterrows():
            company_idx = ground_truth['company_index']
            expected_label = ground_truth['expected_insurance_label']
            
            # Get predicted labels for this company
            predicted_labels = predictions_dict.get(company_idx, [])
            
            # Check if expected label is in predictions
            is_correct = expected_label in predicted_labels
            
            validation_results.append({
                'company_index': company_idx,
                'expected_label': expected_label,
                'predicted_labels': predicted_labels,
                'is_correct': is_correct,
                'naics_code': ground_truth['company_naics_code'],
                'naics_mapping_confidence': ground_truth['naics_mapping_confidence'],
                'company_naics_confidence': ground_truth['company_naics_confidence']
            })
        
        validation_results_df = pd.DataFrame(validation_results)
        return validation_results_df
    
    def calculate_naics_agreement_metrics(self, validation_results):
        """Calculate NAICS-based validation metrics."""
        print("\n=== NAICS-Based Validation Metrics ===")
        
        total_validations = len(validation_results)
        correct_predictions = validation_results['is_correct'].sum()
        naics_agreement_rate = correct_predictions / total_validations
        
        print(f"Total validation pairs: {total_validations}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"NAICS Agreement Rate: {naics_agreement_rate:.3f} ({naics_agreement_rate*100:.1f}%)")
        
        # Agreement by confidence levels
        high_conf_mask = (validation_results['naics_mapping_confidence'] > 0.7) & \
                        (validation_results['company_naics_confidence'] > 0.7)
        high_conf_results = validation_results[high_conf_mask]
        
        if len(high_conf_results) > 0:
            high_conf_agreement = high_conf_results['is_correct'].mean()
            print(f"High confidence agreement rate: {high_conf_agreement:.3f} ({high_conf_agreement*100:.1f}%)")
        
        # Agreement by NAICS code (industry analysis)
        naics_agreement = validation_results.groupby('naics_code')['is_correct'].agg(['count', 'sum', 'mean']).round(3)
        naics_agreement.columns = ['total_validations', 'correct_predictions', 'agreement_rate']
        naics_agreement = naics_agreement.sort_values('agreement_rate', ascending=False)
        
        print(f"\n=== Top 10 Industries (Best Agreement) ===")
        top_industries = naics_agreement.head(10)
        for naics_code, row in top_industries.iterrows():
            if row['total_validations'] >= 3:  # Only show industries with sufficient data
                print(f"NAICS {naics_code}: {row['agreement_rate']:.3f} agreement ({row['correct_predictions']:.0f}/{row['total_validations']:.0f})")
        
        print(f"\n=== Bottom 10 Industries (Worst Agreement) ===")
        bottom_industries = naics_agreement.tail(10)
        for naics_code, row in bottom_industries.iterrows():
            if row['total_validations'] >= 3:
                print(f"NAICS {naics_code}: {row['agreement_rate']:.3f} agreement ({row['correct_predictions']:.0f}/{row['total_validations']:.0f})")
        
        return {
            'naics_agreement_rate': naics_agreement_rate,
            'total_validations': total_validations,
            'correct_predictions': correct_predictions,
            'agreement_by_industry': naics_agreement
        }
    
    def analyze_classification_errors(self, validation_results):
        """Analyze classification errors to identify patterns."""
        print("\n=== Classification Error Analysis ===")
        
        errors = validation_results[~validation_results['is_correct']]
        print(f"Total classification errors: {len(errors)}")
        
        if len(errors) == 0:
            print("No errors to analyze!")
            return
        
        # Most common error patterns
        print("\n=== Most Common Missing Labels ===")
        error_labels = errors['expected_label'].value_counts().head(10)
        for label, count in error_labels.items():
            print(f"'{label}': {count} misses")
        
        # Error patterns by NAICS confidence
        print(f"\n=== Errors by Confidence Level ===")
        low_conf_errors = errors[errors['naics_mapping_confidence'] <= 0.6]
        med_conf_errors = errors[(errors['naics_mapping_confidence'] > 0.6) & 
                               (errors['naics_mapping_confidence'] <= 0.7)]
        high_conf_errors = errors[errors['naics_mapping_confidence'] > 0.7]
        
        print(f"Low confidence errors (<0.6): {len(low_conf_errors)}")
        print(f"Medium confidence errors (0.6-0.7): {len(med_conf_errors)}")  
        print(f"High confidence errors (>0.7): {len(high_conf_errors)}")
        
        return errors
    
    def generate_validation_report(self, validation_results, output_file="data/output/naics_validation_report.csv"):
        """Generate comprehensive validation report."""
        print(f"\nGenerating validation report...")
        
        # Calculate metrics
        metrics = self.calculate_naics_agreement_metrics(validation_results)
        
        # Analyze errors
        errors = self.analyze_classification_errors(validation_results)
        
        # Save detailed results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        validation_results.to_csv(output_file, index=False)
        print(f"Detailed validation results saved to: {output_file}")
        
        # Save summary
        summary_file = output_file.replace('.csv', '_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("NAICS-Based Validation Summary\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total validation pairs: {metrics['total_validations']}\n")
            f.write(f"Correct predictions: {metrics['correct_predictions']}\n")
            f.write(f"NAICS Agreement Rate: {metrics['naics_agreement_rate']:.3f}\n")
            f.write(f"Classification Errors: {len(errors) if errors is not None else 0}\n")
        
        print(f"Validation summary saved to: {summary_file}")
        
        return metrics


def run_naics_validation_demo():
    """Demo function showing how to use NAICS validation."""
    print("=== NAICS Validation Framework Demo ===")
    
    # Initialize validator
    validator = NAICSValidator()
    
    # Create ground truth
    ground_truth = validator.create_validation_ground_truth()
    
    # For demo: create mock predictions (you would replace this with your actual model predictions)
    print("\nCreating demo predictions...")
    unique_companies = ground_truth['company_index'].unique()
    mock_predictions = []
    
    for company_idx in unique_companies:
        # For demo: randomly predict some labels (replace with your actual predictions)
        company_ground_truth = ground_truth[ground_truth['company_index'] == company_idx]
        expected_labels = company_ground_truth['expected_insurance_label'].tolist()
        
        # Mock prediction: sometimes get it right, sometimes wrong
        if np.random.random() > 0.3:  # 70% accuracy demo
            predicted = expected_labels[:1]  # Predict first expected label
        else:
            predicted = ['Random Wrong Label']  # Wrong prediction
            
        mock_predictions.append({
            'company_index': company_idx,
            'predicted_labels': predicted
        })
    
    mock_predictions_df = pd.DataFrame(mock_predictions)
    
    # Validate predictions
    validation_results = validator.validate_classifications(mock_predictions_df)
    
    # Generate report
    metrics = validator.generate_validation_report(validation_results)
    
    print(f"\n=== Demo Complete ===")
    print(f"This framework provides the 'real-world effectiveness' validation")
    print(f"that the task requires, using official NAICS industry codes as ground truth.")
    
    return validator, ground_truth, validation_results


if __name__ == "__main__":
    run_naics_validation_demo() 