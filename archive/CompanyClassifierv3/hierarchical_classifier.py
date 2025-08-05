#!/usr/bin/env python3
"""
Hierarchical Classification: Domain → Specific Label
This should improve accuracy by:
1. First classifying into broad domains (easier task)
2. Then finding specific labels within that domain (focused search)
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter, defaultdict
import re
import json
from typing import Dict, List, Tuple


class HierarchicalClassifier:
    """
    Two-stage classification:
    Stage 1: Classify into broad domains (Software, Manufacturing, etc.)
    Stage 2: Find specific label within that domain using similarity
    """
    
    def __init__(self, training_data_path: str = "data/processed/training_data.csv"):
        self.df = pd.read_csv(training_data_path)
        # Remove rows with NaN primary_labels
        self.df = self.df.dropna(subset=['primary_label'])
        self.df = self.df[self.df['primary_label'].str.len() > 0]  # Remove empty strings
        self.sentence_model = None
        self.domain_classifier = None
        self.tfidf_vectorizer = None
        self.domain_mappings = {}
        self.domain_embeddings = {}
        self.label_to_domain = {}
        
        # Initialize the system
        self.initialize()
    
    def analyze_domains(self) -> Dict[str, List[str]]:
        """Analyze existing labels to identify natural domains"""
        print("🔍 Analyzing labels to identify natural domains...")
        
        # Get unique labels, excluding NaN values
        labels = self.df['primary_label'].dropna().unique()
        labels = [label for label in labels if isinstance(label, str)]
        print(f"Total unique labels: {len(labels)}")
        
        # Define domain keywords and patterns
        domain_patterns = {
            'Software & Technology': [
                'software', 'application', 'web', 'mobile', 'tech', 'digital', 'cyber',
                'computer', 'system', 'platform', 'app', 'development', 'programming',
                'IT', 'internet', 'online', 'electronic', 'database', 'cloud'
            ],
            'Manufacturing & Production': [
                'manufacturing', 'production', 'factory', 'industrial', 'assembly',
                'machinery', 'equipment', 'fabrication', 'processing', 'parts',
                'component', 'automotive', 'metal', 'plastic', 'chemical', 'textile'
            ],
            'Food & Beverage': [
                'food', 'restaurant', 'catering', 'beverage', 'drink', 'kitchen',
                'culinary', 'dining', 'bar', 'cafe', 'bakery', 'grocery', 'nutrition'
            ],
            'Healthcare & Medical': [
                'medical', 'health', 'hospital', 'clinic', 'pharmaceutical', 'dental',
                'therapy', 'care', 'wellness', 'fitness', 'veterinary', 'diagnostic'
            ],
            'Construction & Real Estate': [
                'construction', 'building', 'real estate', 'property', 'contractor',
                'architecture', 'engineering', 'renovation', 'infrastructure', 'housing'
            ],
            'Financial Services': [
                'financial', 'banking', 'insurance', 'investment', 'loan', 'credit',
                'accounting', 'tax', 'finance', 'money', 'payment', 'wealth'
            ],
            'Transportation & Logistics': [
                'transportation', 'logistics', 'shipping', 'delivery', 'freight',
                'warehouse', 'distribution', 'supply chain', 'trucking', 'cargo'
            ],
            'Retail & Commerce': [
                'retail', 'store', 'shop', 'market', 'sales', 'merchant', 'commerce',
                'trading', 'wholesale', 'distribution', 'vendor'
            ],
            'Education & Training': [
                'education', 'school', 'training', 'teaching', 'learning', 'academic',
                'university', 'college', 'course', 'instruction', 'tutoring'
            ],
            'Professional Services': [
                'consulting', 'advisory', 'legal', 'law', 'attorney', 'audit',
                'compliance', 'business', 'professional', 'service', 'management'
            ],
            'Entertainment & Media': [
                'entertainment', 'media', 'broadcasting', 'publishing', 'advertising',
                'marketing', 'creative', 'design', 'content', 'production', 'studio'
            ],
            'Energy & Utilities': [
                'energy', 'utility', 'power', 'electricity', 'gas', 'water', 'oil',
                'renewable', 'solar', 'wind', 'nuclear', 'coal'
            ],
            'Agriculture': [
                'agriculture', 'farming', 'crop', 'livestock', 'farm', 'agricultural',
                'produce', 'harvest', 'cultivation', 'dairy', 'poultry'
            ],
            'Government & Non-Profit': [
                'government', 'public', 'municipal', 'federal', 'state', 'non-profit',
                'nonprofit', 'charity', 'foundation', 'association', 'organization'
            ]
        }
        
        # Map each label to domains
        label_domains = {}
        domain_counts = defaultdict(int)
        
        for label in labels:
            # Skip NaN labels
            if pd.isna(label) or not isinstance(label, str):
                continue
                
            label_lower = label.lower()
            matched_domains = []
            
            for domain, keywords in domain_patterns.items():
                for keyword in keywords:
                    if keyword in label_lower:
                        matched_domains.append(domain)
                        break
            
            # If no domain matched, classify as "Other"
            if not matched_domains:
                matched_domains = ['Other Services']
            
            # Take the first matched domain (could be improved with scoring)
            assigned_domain = matched_domains[0]
            label_domains[label] = assigned_domain
            domain_counts[assigned_domain] += 1
        
        # Group labels by domain
        domain_to_labels = defaultdict(list)
        for label, domain in label_domains.items():
            domain_to_labels[domain].append(label)
        
        # Print domain analysis
        print(f"\n📊 Domain Analysis:")
        for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {domain}: {count} labels")
            if count <= 5:  # Show labels for small domains
                print(f"    Labels: {domain_to_labels[domain][:5]}")
        
        return dict(domain_to_labels), label_domains
    
    def initialize(self):
        """Initialize the hierarchical classifier"""
        print("🏗️ Initializing hierarchical classifier...")
        
        # Load sentence transformer
        print("Loading semantic model...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Analyze domains
        self.domain_mappings, self.label_to_domain = self.analyze_domains()
        
        # Create domain classifier (Stage 1)
        print("\n🎯 Training domain classifier...")
        self.train_domain_classifier()
        
        # Create domain-specific embeddings (Stage 2)
        print("🔗 Creating domain-specific embeddings...")
        self.create_domain_embeddings()
        
        print("✅ Hierarchical classifier initialized!")
    
    def train_domain_classifier(self):
        """Train a classifier to predict broad domains"""
        
        # Prepare training data
        X_text = []
        y_domains = []
        
        for idx, row in self.df.iterrows():
            # Get domain for this label first
            label = row['primary_label']
            if pd.isna(label) or not isinstance(label, str) or label == '':
                continue  # Skip invalid labels
            
            # Combine description with business tags
            desc = str(row.get('description', '')).strip()
            tags = row.get('business_tags', '[]')
            
            if isinstance(tags, str) and tags.startswith('['):
                try:
                    tags = eval(tags)
                    tags_text = ' '.join(tags) if isinstance(tags, list) else ''
                except:
                    tags_text = ''
            else:
                tags_text = str(tags)
            
            combined_text = f"{desc} {tags_text}"
            X_text.append(combined_text)
            
            domain = self.label_to_domain.get(label, 'Other Services')
            y_domains.append(domain)
        
        # Create TF-IDF features
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            stop_words='english'
        )
        
        X_tfidf = self.tfidf_vectorizer.fit_transform(X_text)
        
        # Train Random Forest for domain classification
        self.domain_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        
        self.domain_classifier.fit(X_tfidf, y_domains)
        
        print(f"  Trained on {len(X_text)} examples")
        print(f"  {len(set(y_domains))} unique domains")
    
    def create_domain_embeddings(self):
        """Create separate embeddings for each domain"""
        
        for domain, labels in self.domain_mappings.items():
            print(f"  Creating embeddings for {domain} ({len(labels)} labels)...")
            
            # Get companies in this domain
            domain_companies = self.df[self.df['primary_label'].isin(labels)]
            
            if len(domain_companies) == 0:
                continue
            
            # Create embeddings for this domain
            descriptions = []
            for idx, row in domain_companies.iterrows():
                desc = str(row.get('description', '')).strip()
                tags = row.get('business_tags', '[]')
                
                if isinstance(tags, str) and tags.startswith('['):
                    try:
                        tags = eval(tags)
                        tags_text = ' '.join(tags) if isinstance(tags, list) else ''
                    except:
                        tags_text = ''
                else:
                    tags_text = str(tags)
                
                combined_text = f"{desc} {tags_text}"
                descriptions.append(combined_text)
            
            # Create embeddings
            embeddings = self.sentence_model.encode(descriptions, show_progress_bar=False)
            
            self.domain_embeddings[domain] = {
                'embeddings': embeddings,
                'companies': domain_companies.reset_index(drop=True),
                'labels': labels
            }
    
    def predict_domain(self, company_data: Dict) -> Tuple[str, float]:
        """Stage 1: Predict the broad domain"""
        
        # Prepare text
        desc = company_data.get('description', '')
        tags = company_data.get('business_tags', [])
        
        if isinstance(tags, list):
            tags_text = ' '.join(tags)
        else:
            tags_text = str(tags)
        
        combined_text = f"{desc} {tags_text}"
        
        # Transform to TF-IDF
        X_tfidf = self.tfidf_vectorizer.transform([combined_text])
        
        # Predict domain probabilities
        domain_probs = self.domain_classifier.predict_proba(X_tfidf)[0]
        domain_classes = self.domain_classifier.classes_
        
        # Get top domain
        top_idx = np.argmax(domain_probs)
        predicted_domain = domain_classes[top_idx]
        confidence = domain_probs[top_idx]
        
        return predicted_domain, confidence
    
    def predict_label_in_domain(self, company_data: Dict, domain: str, top_k: int = 5) -> List[Dict]:
        """Stage 2: Find specific label within the predicted domain"""
        
        if domain not in self.domain_embeddings:
            return []
        
        domain_data = self.domain_embeddings[domain]
        
        # Create query embedding
        desc = company_data.get('description', '')
        tags = company_data.get('business_tags', [])
        
        if isinstance(tags, list):
            tags_text = ' '.join(tags)
        else:
            tags_text = str(tags)
        
        query_text = f"{desc} {tags_text}"
        query_embedding = self.sentence_model.encode([query_text])
        
        # Calculate similarities within domain
        similarities = cosine_similarity(query_embedding, domain_data['embeddings'])[0]
        
        # Get top-k most similar
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            similarity = similarities[idx]
            similar_company = domain_data['companies'].iloc[idx]
            
            # Handle missing descriptions
            desc_sim = similar_company.get('description', '')
            if pd.isna(desc_sim) or not isinstance(desc_sim, str):
                desc_sim = 'No description available'
            
            results.append({
                'label': similar_company['primary_label'],
                'similarity': float(similarity),
                'description': desc_sim[:200] + '...' if len(desc_sim) > 200 else desc_sim,
                'domain': domain
            })
        
        return results
    
    def predict(self, company_data: Dict, top_k: int = 5) -> Dict:
        """Full hierarchical prediction: Domain → Specific Label"""
        
        # Stage 1: Predict domain
        predicted_domain, domain_confidence = self.predict_domain(company_data)
        
        # Stage 2: Find specific labels within domain
        specific_predictions = self.predict_label_in_domain(
            company_data, predicted_domain, top_k=top_k*2
        )
        
        # Aggregate predictions by label
        label_scores = {}
        for pred in specific_predictions:
            label = pred['label']
            similarity = pred['similarity']
            
            if label not in label_scores:
                label_scores[label] = {'total_similarity': 0, 'count': 0, 'max_similarity': 0}
            
            label_scores[label]['total_similarity'] += similarity
            label_scores[label]['count'] += 1
            label_scores[label]['max_similarity'] = max(
                label_scores[label]['max_similarity'], similarity
            )
        
        # Sort predictions
        sorted_predictions = []
        for label, data in sorted(label_scores.items(), 
                                key=lambda x: x[1]['max_similarity'], 
                                reverse=True)[:top_k]:
            sorted_predictions.append({
                'label': label,
                'confidence': data['max_similarity'],
                'domain_confidence': domain_confidence,
                'combined_confidence': data['max_similarity'] * domain_confidence,
                'support': data['count']
            })
        
        # Result
        if sorted_predictions:
            primary = sorted_predictions[0]
            result = {
                'primary_prediction': primary['label'],
                'primary_confidence': primary['combined_confidence'],
                'predicted_domain': predicted_domain,
                'domain_confidence': domain_confidence,
                'predictions': sorted_predictions,
                'similar_companies': specific_predictions[:5],
                'method': 'hierarchical'
            }
        else:
            result = {
                'primary_prediction': 'Unknown',
                'primary_confidence': 0.0,
                'predicted_domain': predicted_domain,
                'domain_confidence': domain_confidence,
                'predictions': [],
                'similar_companies': [],
                'method': 'hierarchical'
            }
        
        return result


def evaluate_hierarchical_approach():
    """Test hierarchical vs flat similarity approach"""
    print("🧪 EVALUATING HIERARCHICAL CLASSIFICATION")
    print("=" * 60)
    
    # Load data and create test split
    df = pd.read_csv('data/processed/training_data.csv')
    
    # Simple split for testing
    test_size = min(200, len(df) // 10)  # Test on up to 200 examples
    test_indices = np.random.choice(len(df), size=test_size, replace=False)
    test_df = df.iloc[test_indices].reset_index(drop=True)
    
    print(f"Testing on {len(test_df)} examples...")
    
    # Initialize hierarchical classifier
    hierarchical = HierarchicalClassifier()
    
    # Test hierarchical approach
    correct_hierarchical = 0
    correct_domain = 0
    top3_hierarchical = 0
    
    results = []
    
    for idx, row in test_df.iterrows():
        if idx % 50 == 0:
            print(f"  Progress: {idx}/{len(test_df)}")
        
        company_data = {
            'description': row['description'],
            'business_tags': row['business_tags']
        }
        
        # Get prediction
        result = hierarchical.predict(company_data)
        
        true_label = row['primary_label']
        true_domain = hierarchical.label_to_domain.get(true_label, 'Other Services')
        
        # Check accuracies
        pred_label = result['primary_prediction']  
        pred_domain = result['predicted_domain']
        
        if pred_label == true_label:
            correct_hierarchical += 1
        
        if pred_domain == true_domain:
            correct_domain += 1
        
        # Top-3 accuracy
        top_labels = [p['label'] for p in result['predictions']]
        if true_label in top_labels[:3]:
            top3_hierarchical += 1
        
        results.append({
            'true_label': true_label,
            'true_domain': true_domain,
            'pred_label': pred_label,
            'pred_domain': pred_domain,
            'domain_correct': pred_domain == true_domain,
            'label_correct': pred_label == true_label,
            'confidence': result['primary_confidence']
        })
    
    # Calculate accuracies
    domain_accuracy = correct_domain / len(test_df)
    label_accuracy = correct_hierarchical / len(test_df)
    top3_accuracy = top3_hierarchical / len(test_df)
    
    print(f"\n📊 HIERARCHICAL RESULTS:")
    print(f"Domain Accuracy: {domain_accuracy:.3f} ({correct_domain}/{len(test_df)})")
    print(f"Label Accuracy: {label_accuracy:.3f} ({correct_hierarchical}/{len(test_df)})")
    print(f"Top-3 Accuracy: {top3_accuracy:.3f} ({top3_hierarchical}/{len(test_df)})")
    
    # Analyze domain performance
    domain_analysis = defaultdict(lambda: {'correct': 0, 'total': 0})
    for result in results:
        domain = result['true_domain']
        domain_analysis[domain]['total'] += 1
        if result['label_correct']:
            domain_analysis[domain]['correct'] += 1
    
    print(f"\n🔍 DOMAIN-WISE PERFORMANCE:")
    for domain, stats in sorted(domain_analysis.items(), 
                               key=lambda x: x[1]['correct']/max(x[1]['total'],1), 
                               reverse=True):
        if stats['total'] >= 3:  # Only show domains with 3+ examples
            acc = stats['correct'] / stats['total']
            print(f"  {domain}: {acc:.2f} ({stats['correct']}/{stats['total']})")
    
    return domain_accuracy, label_accuracy, top3_accuracy


if __name__ == "__main__":
    print("🏗️ HIERARCHICAL CLASSIFICATION APPROACH")
    print("=" * 60)
    print("This approach:")
    print("1. First classifies companies into broad domains (Software, Manufacturing, etc.)")
    print("2. Then finds specific labels within that domain using similarity")
    print("3. Should improve accuracy by focusing the search")
    
    # Run evaluation
    domain_acc, label_acc, top3_acc = evaluate_hierarchical_approach()
    
    print(f"\n🎯 SUMMARY:")
    print(f"Domain classification: {domain_acc:.1%}")
    print(f"Final label accuracy: {label_acc:.1%}")
    print(f"Top-3 accuracy: {top3_acc:.1%}")
    
    print(f"\nThis hierarchical approach should be more accurate than flat similarity!") 