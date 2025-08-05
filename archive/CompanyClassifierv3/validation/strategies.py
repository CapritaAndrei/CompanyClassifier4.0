"""
Advanced Validation Strategies for Insurance Classification
Without ground truth, we need creative validation approaches
"""

import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from collections import Counter
import json
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer


class ValidationStrategies:
    """
    Multiple validation strategies for insurance classification
    without predefined ground truth
    """
    
    def __init__(self, classifier):
        """Initialize with trained classifier"""
        self.classifier = classifier
        self.model = classifier.model
        
    def consistency_validation(self, companies_df: pd.DataFrame, n_samples: int = 100) -> Dict:
        """
        Validation Strategy 1: Consistency Check
        
        Test if similar companies get similar labels
        - Find pairs of very similar companies
        - Check if they receive similar insurance labels
        - High consistency = good classifier
        
        Args:
            companies_df: DataFrame with company data
            n_samples: Number of company pairs to test
            
        Returns:
            Consistency metrics
        """
        print("🔄 Running Consistency Validation...")
        
        # Sample random companies
        sample_indices = np.random.choice(len(companies_df), min(n_samples, len(companies_df)), replace=False)
        
        consistency_scores = []
        examples = []
        
        for idx in sample_indices:
            company1 = companies_df.iloc[idx]
            company1_dict = company1.to_dict()
            company1_text = self.classifier.get_company_features(company1_dict)
            
            # Find most similar company
            company1_embedding = self.model.encode(company1_text)
            
            # Compute similarities with all other companies
            similarities = []
            for other_idx in range(len(companies_df)):
                if other_idx != idx:
                    company2 = companies_df.iloc[other_idx]
                    company2_dict = company2.to_dict()
                    company2_text = self.classifier.get_company_features(company2_dict)
                    company2_embedding = self.model.encode(company2_text)
                    
                    similarity = np.dot(company1_embedding, company2_embedding) / (
                        np.linalg.norm(company1_embedding) * np.linalg.norm(company2_embedding)
                    )
                    similarities.append((other_idx, similarity))
            
            # Get most similar company
            similarities.sort(key=lambda x: x[1], reverse=True)
            most_similar_idx, similarity_score = similarities[0]
            
            if similarity_score > 0.8:  # Only check highly similar pairs
                company2 = companies_df.iloc[most_similar_idx]
                company2_dict = company2.to_dict()
                
                # Get labels for both companies
                labels1 = self.classifier.get_similarity_suggestions(company1_dict, top_k=3)
                labels2 = self.classifier.get_similarity_suggestions(company2_dict, top_k=3)
                
                # Check label overlap
                labels1_set = set([l[0] for l in labels1])
                labels2_set = set([l[0] for l in labels2])
                
                overlap = len(labels1_set.intersection(labels2_set))
                consistency_score = overlap / len(labels1_set.union(labels2_set))
                
                consistency_scores.append(consistency_score)
                
                if len(examples) < 5:
                    examples.append({
                        'company1': company1['description'][:100] + '...',
                        'company2': company2['description'][:100] + '...',
                        'similarity': similarity_score,
                        'labels1': labels1[:3],
                        'labels2': labels2[:3],
                        'consistency': consistency_score
                    })
        
        return {
            'average_consistency': np.mean(consistency_scores) if consistency_scores else 0,
            'std_consistency': np.std(consistency_scores) if consistency_scores else 0,
            'n_pairs_tested': len(consistency_scores),
            'examples': examples
        }
    
    def cluster_coherence_validation(self, companies_df: pd.DataFrame, n_clusters: int = 20) -> Dict:
        """
        Validation Strategy 2: Cluster Coherence
        
        - Cluster companies based on their features
        - Check if companies in same cluster get similar insurance labels
        - High coherence = good classifier
        
        Args:
            companies_df: DataFrame with company data
            n_clusters: Number of clusters to create
            
        Returns:
            Cluster coherence metrics
        """
        print("🎯 Running Cluster Coherence Validation...")
        
        # Create embeddings for all companies
        embeddings = []
        for idx, company in companies_df.iterrows():
            company_dict = company.to_dict()
            company_text = self.classifier.get_company_features(company_dict)
            embedding = self.model.encode(company_text)
            embeddings.append(embedding)
            
        embeddings = np.array(embeddings)
        
        # Cluster companies
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(embeddings, cluster_labels)
        
        # Analyze label coherence within clusters
        cluster_coherence_scores = []
        cluster_examples = []
        
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            
            if len(cluster_indices) < 2:
                continue
                
            # Get insurance labels for all companies in cluster
            all_labels = []
            for idx in cluster_indices[:10]:  # Sample up to 10 companies per cluster
                company = companies_df.iloc[idx]
                company_dict = company.to_dict()
                suggestions = self.classifier.get_similarity_suggestions(company_dict, top_k=3)
                all_labels.extend([l[0] for l in suggestions])
            
            # Calculate label diversity (lower = more coherent)
            label_counts = Counter(all_labels)
            total_labels = len(all_labels)
            
            if total_labels > 0:
                # Entropy-based coherence
                entropy = 0
                for count in label_counts.values():
                    p = count / total_labels
                    if p > 0:
                        entropy -= p * np.log(p)
                
                # Normalize by maximum possible entropy
                max_entropy = np.log(len(label_counts))
                coherence = 1 - (entropy / max_entropy) if max_entropy > 0 else 1
                
                cluster_coherence_scores.append(coherence)
                
                if len(cluster_examples) < 3:
                    cluster_examples.append({
                        'cluster_id': cluster_id,
                        'size': len(cluster_indices),
                        'top_labels': label_counts.most_common(5),
                        'coherence': coherence
                    })
        
        return {
            'silhouette_score': silhouette_avg,
            'average_coherence': np.mean(cluster_coherence_scores),
            'std_coherence': np.std(cluster_coherence_scores),
            'cluster_examples': cluster_examples
        }
    
    def business_logic_validation(self, companies_df: pd.DataFrame) -> Dict:
        """
        Validation Strategy 2: Business Logic Rules
        
        Check if classifications follow insurance industry logic:
        - Construction companies → Construction-related insurance labels
        - Food companies → Food safety/processing labels
        - etc.
        
        Returns:
            Business logic validation results
        """
        print("💼 Running Business Logic Validation...")
        
        # Define business rules
        business_rules = {
            'construction': {
                'keywords': ['construction', 'building', 'contractor', 'civil engineering'],
                'expected_labels': [
                    'Commercial Construction Services',
                    'Residential Construction Services',
                    'Building Cleaning Services'
                ]
            },
            'food': {
                'keywords': ['food', 'restaurant', 'catering', 'beverage', 'bakery'],
                'expected_labels': [
                    'Food Processing Services',
                    'Catering Services',
                    'Food Safety Services',
                    'Frozen Food Processing'
                ]
            },
            'technology': {
                'keywords': ['software', 'technology', 'IT', 'digital', 'web'],
                'expected_labels': [
                    'Software Development Services',
                    'Technology Consulting',
                    'Data Analysis Services',
                    'Website Development Services'
                ]
            }
        }
        
        validation_results = {}
        
        for domain, rules in business_rules.items():
            # Find companies matching domain keywords
            matching_companies = []
            
            for idx, company in companies_df.iterrows():
                description = company['description'].lower()
                if any(keyword in description for keyword in rules['keywords']):
                    matching_companies.append((idx, company))
            
            if matching_companies:
                # Check if they get appropriate labels
                correct_classifications = 0
                examples = []
                
                for idx, company in matching_companies[:20]:  # Sample up to 20
                    company_dict = company.to_dict()
                    suggestions = self.classifier.get_similarity_suggestions(company_dict, top_k=5)
                    suggested_labels = [l[0] for l in suggestions]
                    
                    # Check if any expected label appears in suggestions
                    if any(expected in suggested_labels for expected in rules['expected_labels']):
                        correct_classifications += 1
                        
                    if len(examples) < 2:
                        examples.append({
                            'company': company['description'][:100] + '...',
                            'suggested_labels': suggestions[:3],
                            'matches_expectation': any(expected in suggested_labels for expected in rules['expected_labels'])
                        })
                
                accuracy = correct_classifications / len(matching_companies)
                
                validation_results[domain] = {
                    'n_companies': len(matching_companies),
                    'accuracy': accuracy,
                    'examples': examples
                }
        
        return validation_results
    
    def cross_validation_with_confidence(self, companies_df: pd.DataFrame, n_folds: int = 5) -> Dict:
        """
        Validation Strategy 4: Pseudo Cross-Validation
        
        - Use high-confidence classifications as pseudo-labels
        - Train on subset, test on another
        - Measure consistency
        
        Args:
            companies_df: DataFrame with company data
            n_folds: Number of folds for cross-validation
            
        Returns:
            Cross-validation metrics
        """
        print("🔀 Running Pseudo Cross-Validation...")
        
        # Get high-confidence classifications
        results = self.classifier.batch_classify(companies_df, confidence_threshold=0.85)
        
        # Filter only high-confidence results
        high_conf_indices = []
        high_conf_labels = []
        
        for _, row in results.iterrows():
            if row['high_confidence_labels']:
                high_conf_indices.append(row['company_index'])
                high_conf_labels.append(row['top_label'])
        
        if len(high_conf_indices) < 10:
            return {
                'status': 'Not enough high-confidence samples',
                'n_samples': len(high_conf_indices)
            }
        
        # Simple train-test split validation
        n_samples = len(high_conf_indices)
        test_size = n_samples // n_folds
        
        fold_scores = []
        
        for fold in range(n_folds):
            test_start = fold * test_size
            test_end = test_start + test_size if fold < n_folds - 1 else n_samples
            
            test_indices = high_conf_indices[test_start:test_end]
            test_labels = high_conf_labels[test_start:test_end]
            
            # Test consistency
            correct = 0
            for i, idx in enumerate(test_indices):
                company = companies_df.iloc[idx]
                company_dict = company.to_dict()
                
                # Get top prediction
                suggestions = self.classifier.get_similarity_suggestions(company_dict, top_k=1)
                if suggestions and suggestions[0][0] == test_labels[i]:
                    correct += 1
            
            accuracy = correct / len(test_indices) if test_indices else 0
            fold_scores.append(accuracy)
        
        return {
            'average_accuracy': np.mean(fold_scores),
            'std_accuracy': np.std(fold_scores),
            'n_high_confidence_samples': len(high_conf_indices),
            'fold_scores': fold_scores
        }
    
    def human_interpretability_check(self, companies_df: pd.DataFrame, n_samples: int = 10) -> List[Dict]:
        """
        Validation Strategy 5: Human Interpretability
        
        Generate examples for human review with explanations
        
        Args:
            companies_df: DataFrame with company data
            n_samples: Number of examples to generate
            
        Returns:
            List of examples with explanations
        """
        print("👁️ Generating Human Interpretability Examples...")
        
        # Sample diverse companies
        sample_indices = np.random.choice(len(companies_df), min(n_samples, len(companies_df)), replace=False)
        
        examples = []
        
        for idx in sample_indices:
            company = companies_df.iloc[idx]
            company_dict = company.to_dict()
            
            # Get suggestions
            suggestions = self.classifier.get_similarity_suggestions(company_dict, top_k=5)
            
            # Extract key features that led to classification
            description_words = company['description'].lower().split()
            tag_words = []
            if isinstance(company['business_tags'], str):
                try:
                    tags = eval(company['business_tags'])
                    tag_words = ' '.join(tags).lower().split()
                except:
                    tag_words = company['business_tags'].lower().split()
            
            # Find matching words between company and suggested labels
            explanations = []
            for label, score in suggestions[:3]:
                label_words = label.lower().split()
                matching_words = set(description_words + tag_words).intersection(set(label_words))
                
                explanation = {
                    'label': label,
                    'score': score,
                    'matching_keywords': list(matching_words),
                    'reasoning': self._generate_reasoning(company_dict, label)
                }
                explanations.append(explanation)
            
            examples.append({
                'company_description': company['description'],
                'business_tags': company['business_tags'],
                'sector': company['sector'],
                'category': company['category'],
                'classifications': explanations
            })
        
        return examples
    
    def _generate_reasoning(self, company_dict: Dict, label: str) -> str:
        """Generate human-readable reasoning for a classification"""
        
        # Simple keyword-based reasoning
        company_text = self.classifier.get_company_features(company_dict).lower()
        label_lower = label.lower()
        
        reasons = []
        
        # Check for direct keyword matches
        if 'construction' in company_text and 'construction' in label_lower:
            reasons.append("Company operates in construction industry")
        if 'food' in company_text and 'food' in label_lower:
            reasons.append("Company is involved in food-related business")
        if 'technology' in company_text and ('technology' in label_lower or 'software' in label_lower):
            reasons.append("Company provides technology services")
        
        # Check for service matches
        if 'services' in company_text and 'services' in label_lower:
            reasons.append("Both company and label indicate service provision")
        
        # Sector-based reasoning
        if 'sector' in company_dict:
            if company_dict['sector'] == 'Manufacturing' and 'manufacturing' in label_lower:
                reasons.append(f"Company sector ({company_dict['sector']}) matches label category")
        
        if not reasons:
            reasons.append("Semantic similarity between company description and insurance category")
        
        return "; ".join(reasons)
    
    def generate_validation_report(self, companies_df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive validation report using all strategies
        
        Args:
            companies_df: DataFrame with company data
            
        Returns:
            Complete validation report
        """
        print("\n📊 Generating Comprehensive Validation Report...\n")
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'n_companies': len(companies_df),
            'n_labels': len(self.classifier.labels),
            'validation_results': {}
        }
        
        # Run all validation strategies
        print("1/5", end=" ")
        report['validation_results']['consistency'] = self.consistency_validation(companies_df, n_samples=50)
        
        print("2/5", end=" ")
        report['validation_results']['cluster_coherence'] = self.cluster_coherence_validation(companies_df, n_clusters=15)
        
        print("3/5", end=" ")
        report['validation_results']['business_logic'] = self.business_logic_validation(companies_df)
        
        print("4/5", end=" ")
        report['validation_results']['cross_validation'] = self.cross_validation_with_confidence(companies_df)
        
        print("5/5", end=" ")
        report['validation_results']['interpretability_examples'] = self.human_interpretability_check(companies_df, n_samples=5)
        
        print("\n✅ Validation report complete!")
        
        # Save report
        report_path = Path(f'data/validation_report_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.json')
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📁 Report saved to: {report_path}")
        
        # Generate summary
        self._print_validation_summary(report)
        
        return report
    
    def _print_validation_summary(self, report: Dict):
        """Print a human-readable summary of validation results"""
        
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        results = report['validation_results']
        
        # Consistency
        if 'consistency' in results:
            cons = results['consistency']
            print(f"\n📊 Consistency Validation:")
            print(f"   Average consistency: {cons['average_consistency']:.2%}")
            print(f"   Based on {cons['n_pairs_tested']} similar company pairs")
        
        # Cluster Coherence
        if 'cluster_coherence' in results:
            cluster = results['cluster_coherence']
            print(f"\n🎯 Cluster Coherence:")
            print(f"   Silhouette score: {cluster['silhouette_score']:.3f}")
            print(f"   Average coherence: {cluster['average_coherence']:.2%}")
        
        # Business Logic
        if 'business_logic' in results:
            print(f"\n💼 Business Logic Validation:")
            for domain, metrics in results['business_logic'].items():
                print(f"   {domain.capitalize()}: {metrics['accuracy']:.2%} accuracy ({metrics['n_companies']} companies)")
        
        # Cross Validation
        if 'cross_validation' in results:
            cv = results['cross_validation']
            if 'average_accuracy' in cv:
                print(f"\n🔀 Pseudo Cross-Validation:")
                print(f"   Average accuracy: {cv['average_accuracy']:.2%}")
                print(f"   Based on {cv['n_high_confidence_samples']} high-confidence samples")
        
        print("\n" + "="*60)
        
        # Overall assessment
        print("\n🎯 OVERALL ASSESSMENT:")
        
        # Calculate overall score
        scores = []
        if 'consistency' in results:
            scores.append(results['consistency']['average_consistency'])
        if 'cluster_coherence' in results:
            scores.append(results['cluster_coherence']['average_coherence'])
        if 'business_logic' in results:
            bl_scores = [m['accuracy'] for m in results['business_logic'].values()]
            if bl_scores:
                scores.append(np.mean(bl_scores))
        
        if scores:
            overall_score = np.mean(scores)
            print(f"   Composite validation score: {overall_score:.2%}")
            
            if overall_score > 0.7:
                print("   ✅ Classifier shows GOOD performance")
            elif overall_score > 0.5:
                print("   ⚠️  Classifier shows MODERATE performance")
            else:
                print("   ❌ Classifier needs improvement")
        
        print("\n" + "="*60) 