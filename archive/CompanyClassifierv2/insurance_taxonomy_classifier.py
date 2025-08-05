#!/usr/bin/env python3
"""
Insurance Taxonomy Few-Shot Classifier
Interactive labeling system with active learning for multi-label classification
"""

import pandas as pd
import numpy as np
import json
import re
import csv
from collections import defaultdict, Counter
from pathlib import Path
import pickle
from datetime import datetime

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, multilabel_confusion_matrix

import warnings
warnings.filterwarnings('ignore')

class InsuranceTaxonomyClassifier:
    """Few-shot learning system for insurance taxonomy classification"""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.labels = []
        self.label_patterns = {}
        self.companies = []
        self.labeled_data = []
        self.classifier = None
        self.label_embeddings = None
        
        # Create data directories
        Path("data/labeled").mkdir(exist_ok=True)
        Path("data/models").mkdir(exist_ok=True)
        
    def load_taxonomy(self):
        """Load and analyze insurance taxonomy labels"""
        print("📋 Loading insurance taxonomy...")
        
        # Load labels
        df = pd.read_csv('data/input/insurance_taxonomy - insurance_taxonomy.csv')
        self.labels = df['label'].tolist()
        
        print(f"✅ Loaded {len(self.labels)} taxonomy labels")
        
        # Analyze label patterns
        self.analyze_label_patterns()
        
        # Create label embeddings for semantic search
        print("🚀 Creating label embeddings...")
        self.label_embeddings = self.model.encode(self.labels)
        print("✅ Label embeddings created")
        
    def analyze_label_patterns(self):
        """Extract patterns from labels to understand the taxonomy structure"""
        print("🔍 Analyzing label patterns...")
        
        # Extract service types, modifiers, and activities
        service_types = set()
        modifiers = set()
        activities = set()
        
        for label in self.labels:
            words = label.lower().split()
            
            # Common service types
            for word in words:
                if word in ['plumbing', 'electrical', 'construction', 'roofing', 'hvac', 
                           'welding', 'cleaning', 'painting', 'installation', 'manufacturing']:
                    service_types.add(word)
                    
                # Common modifiers  
                if word in ['residential', 'commercial', 'industrial', 'low-rise', 'high-rise',
                           'single', 'multi', 'new', 'general']:
                    modifiers.add(word)
                    
                # Common activities
                if word in ['installation', 'repair', 'maintenance', 'services', 'construction',
                           'cleaning', 'inspection', 'manufacturing', 'processing']:
                    activities.add(word)
        
        self.label_patterns = {
            'service_types': list(service_types),
            'modifiers': list(modifiers), 
            'activities': list(activities)
        }
        
        print(f"   🏷️  Service types: {len(service_types)} ({list(service_types)[:5]}...)")
        print(f"   🏷️  Modifiers: {len(modifiers)} ({list(modifiers)[:5]}...)")
        print(f"   🏷️  Activities: {len(activities)} ({list(activities)[:5]}...)")
        
    def load_companies(self, sample_size=None):
        """Load company data efficiently"""
        print("🏢 Loading company data...")
        
        companies = []
        with open('data/input/ml_insurance_challenge.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, company in enumerate(reader):
                if sample_size and i >= sample_size:
                    break
                companies.append(company)
        
        self.companies = companies
        print(f"✅ Loaded {len(companies)} companies")
        
        # Show sample company
        if companies:
            sample = companies[0]
            print(f"\n📋 Sample company:")
            print(f"   Description: {sample['description'][:100]}...")
            print(f"   Business Tags: {sample['business_tags'][:50]}...")
            print(f"   Category: {sample['category']}")
            
    def create_company_representation(self, company):
        """Create comprehensive text representation of company"""
        parts = []
        
        # Description
        if company.get('description'):
            parts.append(f"Description: {company['description']}")
            
        # Business tags
        if company.get('business_tags'):
            # Parse tags (handle different formats)
            tags_str = company['business_tags']
            if tags_str.startswith('[') and tags_str.endswith(']'):
                try:
                    import ast
                    tags = ast.literal_eval(tags_str)
                    tags_text = ' '.join(tags)
                except:
                    tags_text = tags_str.strip('[]').replace("'", "").replace('"', '')
            else:
                tags_text = tags_str
            parts.append(f"Business Tags: {tags_text}")
            
        # Category and other fields
        for field in ['sector', 'category', 'niche']:
            if company.get(field) and pd.notna(company[field]):
                parts.append(f"{field.title()}: {company[field]}")
                
        return '. '.join(parts)
    
    def find_labeling_candidates(self, label_name, n_candidates=10, min_similarity=0.3):
        """Find companies that are good candidates for a specific label"""
        print(f"\n🔍 Finding candidates for: '{label_name}'")
        
        # Get label embedding
        label_idx = self.labels.index(label_name)
        label_embedding = self.label_embeddings[label_idx].reshape(1, -1)
        
        # Create company embeddings (sample for efficiency)
        sample_companies = self.companies[:1000]  # First 1000 for speed
        company_texts = [self.create_company_representation(c) for c in sample_companies]
        company_embeddings = self.model.encode(company_texts)
        
        # Calculate similarities
        similarities = cosine_similarity(label_embedding, company_embeddings)[0]
        
        # Get top candidates above threshold
        candidate_indices = np.argsort(similarities)[::-1]
        good_candidates = []
        
        for idx in candidate_indices:
            if len(good_candidates) >= n_candidates:
                break
            if similarities[idx] >= min_similarity:
                good_candidates.append({
                    'company': sample_companies[idx],
                    'similarity': similarities[idx],
                    'text': company_texts[idx]
                })
        
        print(f"   Found {len(good_candidates)} candidates (similarity >= {min_similarity})")
        return good_candidates
    
    def get_high_priority_labels(self, n_labels=20):
        """Identify high-priority labels to start with"""
        print("🎯 Identifying high-priority labels...")
        
        # Prioritize based on common service types
        priority_keywords = [
            'plumbing', 'electrical', 'construction', 'roofing', 'hvac',
            'cleaning', 'painting', 'landscaping', 'welding', 'installation',
            'manufacturing', 'consulting', 'management', 'services'
        ]
        
        scored_labels = []
        for label in self.labels:
            score = 0
            label_lower = label.lower()
            
            # Score based on priority keywords
            for keyword in priority_keywords:
                if keyword in label_lower:
                    score += 2
                    
            # Prefer shorter, clearer labels
            if len(label.split()) <= 4:
                score += 1
                
            # Boost residential/commercial services (common)
            if any(word in label_lower for word in ['residential', 'commercial']):
                score += 1
                
            scored_labels.append((label, score))
        
        # Sort by score and return top N
        scored_labels.sort(key=lambda x: x[1], reverse=True)
        top_labels = [label for label, score in scored_labels[:n_labels]]
        
        print(f"✅ Selected {len(top_labels)} high-priority labels:")
        for i, label in enumerate(top_labels[:10]):
            print(f"   {i+1:2d}. {label}")
        if len(top_labels) > 10:
            print(f"   ... and {len(top_labels)-10} more")
            
        return top_labels
    
    def interactive_labeling_session(self, labels_to_label=None, examples_per_label=5):
        """Interactive labeling session with smart candidate selection"""
        print("\n" + "="*60)
        print("🏷️  INTERACTIVE LABELING SESSION")
        print("="*60)
        print("Instructions:")
        print("  y = Yes, this company matches the label")
        print("  n = No, this company doesn't match") 
        print("  s = Skip this company")
        print("  q = Quit labeling session")
        print("  ? = Show more details about the company")
        print("="*60)
        
        if labels_to_label is None:
            labels_to_label = self.get_high_priority_labels()
            
        total_labeled = 0
        session_data = []
        
        for label_idx, label in enumerate(labels_to_label):
            print(f"\n📋 LABEL {label_idx + 1}/{len(labels_to_label)}: {label}")
            print("-" * 50)
            
            # Find candidates for this label
            candidates = self.find_labeling_candidates(label, n_candidates=examples_per_label * 3)
            
            if not candidates:
                print(f"⚠️  No good candidates found for '{label}'. Skipping...")
                continue
                
            labeled_for_this_label = 0
            candidate_idx = 0
            
            while labeled_for_this_label < examples_per_label and candidate_idx < len(candidates):
                candidate = candidates[candidate_idx]
                company = candidate['company']
                similarity = candidate['similarity']
                
                print(f"\n📊 Candidate {candidate_idx + 1} (similarity: {similarity:.3f})")
                print(f"Description: {company['description'][:200]}...")
                if company.get('business_tags'):
                    print(f"Tags: {company['business_tags'][:100]}...")
                print(f"Category: {company.get('category', 'N/A')}")
                
                while True:
                    response = input(f"\nDoes this match '{label}'? (y/n/s/q/?): ").lower().strip()
                    
                    if response == 'y':
                        session_data.append({
                            'company_id': company.get('id', f"unknown_{total_labeled}"),
                            'company_text': self.create_company_representation(company),
                            'label': label,
                            'match': True,
                            'similarity': similarity,
                            'timestamp': datetime.now().isoformat()
                        })
                        labeled_for_this_label += 1
                        total_labeled += 1
                        print("✅ Labeled as MATCH")
                        break
                        
                    elif response == 'n':
                        session_data.append({
                            'company_id': company.get('id', f"unknown_{total_labeled}"),
                            'company_text': self.create_company_representation(company),
                            'label': label,
                            'match': False,
                            'similarity': similarity,
                            'timestamp': datetime.now().isoformat()
                        })
                        total_labeled += 1
                        print("❌ Labeled as NO MATCH")
                        break
                        
                    elif response == 's':
                        print("⏭️  Skipped")
                        break
                        
                    elif response == 'q':
                        print("🛑 Quitting labeling session...")
                        self.save_labeling_session(session_data)
                        return session_data
                        
                    elif response == '?':
                        print(f"\n📋 Full company details:")
                        for key, value in company.items():
                            if value and str(value).strip():
                                print(f"   {key}: {value}")
                        print()
                        
                    else:
                        print("Please enter y, n, s, q, or ?")
                
                candidate_idx += 1
            
            print(f"✅ Completed labeling for '{label}': {labeled_for_this_label} examples")
            
            # Ask if user wants to continue
            if label_idx < len(labels_to_label) - 1:
                continue_response = input(f"\nContinue to next label? (y/n): ").lower().strip()
                if continue_response == 'n':
                    break
        
        print(f"\n🎉 Labeling session complete!")
        print(f"   Total examples labeled: {total_labeled}")
        print(f"   Labels worked on: {len(set(item['label'] for item in session_data))}")
        
        # Save the session data
        self.save_labeling_session(session_data)
        return session_data
    
    def save_labeling_session(self, session_data):
        """Save labeling session data"""
        if not session_data:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/labeled/session_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
            
        print(f"💾 Saved labeling session to {filename}")
        
        # Also append to master file
        master_file = "data/labeled/all_labels.json"
        if Path(master_file).exists():
            with open(master_file, 'r') as f:
                all_data = json.load(f)
        else:
            all_data = []
            
        all_data.extend(session_data)
        
        with open(master_file, 'w') as f:
            json.dump(all_data, f, indent=2)
            
        print(f"💾 Updated master file: {len(all_data)} total examples")
    
    def load_labeled_data(self):
        """Load all labeled data"""
        master_file = "data/labeled/all_labels.json"
        if not Path(master_file).exists():
            print("⚠️  No labeled data found")
            return []
            
        with open(master_file, 'r') as f:
            self.labeled_data = json.load(f)
            
        print(f"📥 Loaded {len(self.labeled_data)} labeled examples")
        
        # Show statistics
        label_counts = Counter(item['label'] for item in self.labeled_data)
        match_counts = Counter(item['match'] for item in self.labeled_data)
        
        print(f"   📊 Match distribution: {dict(match_counts)}")
        print(f"   📊 Top labels: {dict(label_counts.most_common(5))}")
        
        return self.labeled_data
    
    def train_few_shot_classifier(self):
        """Train classifier with labeled examples"""
        if not self.labeled_data:
            self.load_labeled_data()
            
        if len(self.labeled_data) < 10:
            print("⚠️  Need at least 10 labeled examples to train")
            return None
            
        print(f"\n🚀 Training few-shot classifier with {len(self.labeled_data)} examples...")
        
        # Prepare training data
        texts = [item['company_text'] for item in self.labeled_data]
        labels_matrix = []
        
        # Create binary matrix for multi-label classification
        unique_labels = list(set(item['label'] for item in self.labeled_data))
        print(f"   📋 Training on {len(unique_labels)} unique labels")
        
        for item in self.labeled_data:
            label_vector = [0] * len(unique_labels)
            if item['match']:  # Only set to 1 if it's a positive match
                label_idx = unique_labels.index(item['label'])
                label_vector[label_idx] = 1
            labels_matrix.append(label_vector)
        
        # Create embeddings
        print("   🔄 Creating embeddings...")
        embeddings = self.model.encode(texts)
        
        # Train classifier
        print("   🔄 Training classifier...")
        self.classifier = MultiOutputClassifier(LogisticRegression(max_iter=1000))
        self.classifier.fit(embeddings, labels_matrix)
        
        # Store label mapping
        self.trained_labels = unique_labels
        
        print("✅ Classifier trained successfully!")
        
        # Save model
        model_path = "data/models/few_shot_classifier.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'classifier': self.classifier,
                'labels': self.trained_labels,
                'label_embeddings': self.label_embeddings
            }, f)
        print(f"💾 Model saved to {model_path}")
        
        return self.classifier
    
    def predict_company_labels(self, company, threshold=0.5):
        """Predict labels for a company"""
        if self.classifier is None:
            print("⚠️  No trained classifier. Train first with train_few_shot_classifier()")
            return []
            
        # Create company representation
        company_text = self.create_company_representation(company)
        company_embedding = self.model.encode([company_text])
        
        # Get predictions
        probabilities = self.classifier.predict_proba(company_embedding)[0]
        
        # Extract predictions above threshold
        predictions = []
        for i, prob_positive in enumerate(probabilities):
            # prob_positive is array [prob_negative, prob_positive]
            if len(prob_positive) > 1 and prob_positive[1] >= threshold:
                predictions.append({
                    'label': self.trained_labels[i],
                    'confidence': prob_positive[1],
                    'reasoning': f'Classifier confidence: {prob_positive[1]:.3f}'
                })
        
        # Sort by confidence
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        return predictions
    
    def evaluate_on_test_set(self, test_companies=None):
        """Evaluate classifier performance"""
        if test_companies is None:
            # Use some companies from our dataset
            test_companies = self.companies[:50]  # First 50 for evaluation
            
        print(f"🔍 Evaluating classifier on {len(test_companies)} companies...")
        
        results = []
        for company in test_companies:
            predictions = self.predict_company_labels(company)
            results.append({
                'company': company,
                'predictions': predictions,
                'company_text': self.create_company_representation(company)
            })
        
        # Show sample results
        print(f"\n📊 Sample Results:")
        for i, result in enumerate(results[:5]):
            print(f"\n{i+1}. Company: {result['company']['description'][:60]}...")
            if result['predictions']:
                for pred in result['predictions'][:3]:  # Top 3 predictions
                    print(f"   🏷️  {pred['label']} (confidence: {pred['confidence']:.3f})")
            else:
                print("   🏷️  No confident predictions")
        
        return results

def main():
    """Main function to run the few-shot learning system"""
    print("🎯 Insurance Taxonomy Few-Shot Classifier")
    print("=" * 50)
    
    # Initialize classifier
    classifier = InsuranceTaxonomyClassifier()
    
    # Load data
    classifier.load_taxonomy()
    classifier.load_companies(sample_size=2000)  # Load subset for faster processing
    
    print("\n🚀 Ready for interactive labeling!")
    print("Commands:")
    print("  1. Start labeling session")
    print("  2. Train classifier with existing labels")
    print("  3. Evaluate classifier")
    print("  4. Show sample predictions")
    
    while True:
        choice = input("\nEnter choice (1-4) or 'q' to quit: ").strip()
        
        if choice == '1':
            # Interactive labeling
            classifier.interactive_labeling_session()
            
        elif choice == '2':
            print("⚠️  Training functionality coming in next update!")
            
        elif choice == '3':
            print("⚠️  Evaluation functionality coming in next update!")
            
        elif choice == '4':
            print("⚠️  Prediction functionality coming in next update!")
            
        elif choice == 'q':
            print("👋 Goodbye!")
            break
            
        else:
            print("Invalid choice. Please enter 1-4 or 'q'")

if __name__ == "__main__":
    main() 