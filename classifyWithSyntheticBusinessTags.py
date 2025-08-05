"""
Automated Data Cleaning System
Implements two-tier labeling strategy and semantic keyword extraction
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from collections import Counter, defaultdict
import ast
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.embedder import EmbeddingModel
from src.models.classifier import BusinessTagsClassifier
from src.data.loader import DataLoader
from src.config.settings import *

class AutomatedDataCleaner:
    """
    Comprehensive automated data cleaning system
    - Force-assign best matches for unlabeled companies  
    - Semantic keyword extraction for companies without tags
    - Two-tier confidence marking
    - Verification analytics
    """
    
    def __init__(self):
        self.embedder = EmbeddingModel()
        self.classifier = BusinessTagsClassifier()
        self.existing_tags_embeddings = None
        self.existing_tags_list = None
        
        print("🤖 AUTOMATED DATA CLEANER INITIALIZED")
        print("="*60)
        
        # Load taxonomy
        print("📋 Loading insurance taxonomy...")
        taxonomy_labels = DataLoader.load_taxonomy_labels(TAXONOMY_FILE)
        if taxonomy_labels:
            success = self.classifier.load_taxonomy(taxonomy_labels, cache_path=TAXONOMY_EMBEDDINGS_CACHE)
            if success:
                print(f"✅ Loaded {len(taxonomy_labels)} taxonomy labels")
            else:
                raise Exception("Failed to load taxonomy embeddings")
        else:
            raise Exception("Failed to load taxonomy labels")
    
    def load_current_results(self):
        """Load current classification results"""
        results_path = Path("src/data/output/full_classification_results.csv")
        
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")
            
        print("📂 Loading current classification results...")
        self.df = pd.read_csv(results_path)
        
        # Separate into tiers
        self.tier1_companies = self.df[self.df['num_labels_assigned'] > 0].copy()  # High confidence
        self.tier2_companies = self.df[self.df['num_labels_assigned'] == 0].copy()  # Need labels
        
        print(f"✅ Loaded {len(self.df):,} companies")
        print(f"   📊 Tier 1 (labeled): {len(self.tier1_companies):,} companies ({len(self.tier1_companies)/len(self.df)*100:.1f}%)")
        print(f"   📊 Tier 2 (unlabeled): {len(self.tier2_companies):,} companies ({len(self.tier2_companies)/len(self.df)*100:.1f}%)")
        
        return self.df
    
    def extract_existing_business_tags(self):
        """Extract all unique business tags from labeled companies"""
        print("\n🏷️ Extracting existing business tags corpus...")
        
        all_tags = set()
        
        for _, row in self.tier1_companies.iterrows():
            if pd.notna(row['business_tags']) and row['business_tags'] not in ['', '[]']:
                try:
                    tags = ast.literal_eval(row['business_tags'])
                    if isinstance(tags, list):
                        all_tags.update(tags)
                except:
                    # Handle string format
                    if isinstance(row['business_tags'], str):
                        # Simple split approach for string tags
                        tags = [tag.strip() for tag in row['business_tags'].split(',')]
                        all_tags.update(tags)
        
        self.existing_tags_list = list(all_tags)
        print(f"✅ Extracted {len(self.existing_tags_list):,} unique business tags")
        
        # Create embeddings for existing tags
        print("🧠 Creating embeddings for existing business tags...")
        self.existing_tags_embeddings = self.embedder.embed_texts(self.existing_tags_list)
        print(f"✅ Created embeddings for {len(self.existing_tags_list):,} tags")
        
        return self.existing_tags_list
    
    def semantic_keyword_extraction(self, description, top_k=5):
        """
        Extract most relevant keywords from description using semantic similarity
        to existing business tags corpus
        """
        if not description or pd.isna(description):
            return []
            
        # Embed the description
        desc_embedding = self.embedder.embed_single_text(str(description))
        
        # Calculate similarities to all existing tags
        similarities = cosine_similarity(
            desc_embedding.reshape(1, -1), 
            self.existing_tags_embeddings
        )[0]
        
        # Get top-k most similar tags
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        extracted_tags = []
        for idx in top_indices:
            tag = self.existing_tags_list[idx]
            similarity = float(similarities[idx])  # Convert numpy float32 to Python float
            
            # Only include if similarity is reasonable
            if similarity > 0.3:  # Threshold for relevance
                extracted_tags.append({
                    'tag': tag,
                    'similarity': similarity
                })
        
        return extracted_tags
    
    def force_assign_best_labels(self):
        """
        Force-assign best available labels to unlabeled companies
        Two strategies:
        1. Companies with business tags: Use existing classification with lower threshold
        2. Companies without tags: Use semantic keyword extraction + classification
        """
        print(f"\n🎯 FORCE-ASSIGNING LABELS TO {len(self.tier2_companies):,} UNLABELED COMPANIES")
        print("="*70)
        
        # Separate companies with/without business tags
        tier2_with_tags = self.tier2_companies[
            self.tier2_companies['business_tags'].notna() & 
            (self.tier2_companies['business_tags'] != '') & 
            (self.tier2_companies['business_tags'] != '[]')
        ].copy()
        
        tier2_without_tags = self.tier2_companies[
            self.tier2_companies['business_tags'].isna() | 
            (self.tier2_companies['business_tags'] == '') | 
            (self.tier2_companies['business_tags'] == '[]')
        ].copy()
        
        print(f"📊 Companies with business tags: {len(tier2_with_tags):,}")
        print(f"📊 Companies without business tags: {len(tier2_without_tags):,}")
        
        new_assignments = []
        
        # Strategy 1: Companies with business tags - lower threshold
        print(f"\n🔄 Processing companies WITH business tags...")
        
        for idx, (_, row) in enumerate(tier2_with_tags.iterrows()):
            if idx % 100 == 0:
                print(f"   Progress: {idx:,}/{len(tier2_with_tags):,}")
            
            try:
                business_tags = ast.literal_eval(row['business_tags'])
                if not isinstance(business_tags, list):
                    continue
                    
            except:
                continue
            
            # Handle NaN values in sector/category/niche
            sector = row['sector'] if pd.notna(row['sector']) else None
            category = row['category'] if pd.notna(row['category']) else None
            niche = row['niche'] if pd.notna(row['niche']) else None
            
            # Classify with very low threshold to force assignment
            results = self.classifier.classify_company(
                business_tags=business_tags,
                top_k=15,  # More options
                similarity_threshold=0.25,  # Much lower threshold
                keyword_boost=0.0,  # Pure semantic
                sector=sector,
                category=category, 
                niche=niche
            )
            
            if results and len(results) > 0:
                # Take the best match (results are tuples: (label, score))
                best_label = results[0]
                label_name = best_label[0]
                confidence_score = best_label[1]
                
                new_assignments.append({
                    'company_id': row['company_id'],
                    'method': 'force_assign_with_tags',
                    'tier': 'tier2_forced',
                    'insurance_labels': label_name,
                    'confidence_score': confidence_score,
                    'num_labels_assigned': 1,
                    'all_labels_with_scores': f"{label_name} ({confidence_score:.3f})",
                    'original_business_tags': row['business_tags']
                })
        
        # Strategy 2: Companies without business tags - semantic extraction
        print(f"\n🧠 Processing companies WITHOUT business tags...")
        
        for idx, (_, row) in enumerate(tier2_without_tags.iterrows()):
            if idx % 50 == 0:
                print(f"   Progress: {idx:,}/{len(tier2_without_tags):,}")
            
            # Extract semantic keywords from description
            extracted_tags = self.semantic_keyword_extraction(row['description'], top_k=5)
            
            if not extracted_tags:
                continue
                
            # Use extracted tags as business tags
            synthetic_tags = [tag_info['tag'] for tag_info in extracted_tags]
            
            # Handle NaN values in sector/category/niche
            sector = row['sector'] if pd.notna(row['sector']) else None
            category = row['category'] if pd.notna(row['category']) else None
            niche = row['niche'] if pd.notna(row['niche']) else None
            
            # Classify using synthetic tags
            results = self.classifier.classify_company(
                business_tags=synthetic_tags,
                top_k=10,
                similarity_threshold=0.3,  # Slightly higher threshold for quality
                keyword_boost=0.0,
                sector=sector,
                category=category,
                niche=niche
            )
            
            if results and len(results) > 0:
                # Take the best match (results are tuples: (label, score))
                best_label = results[0]
                label_name = best_label[0]
                confidence_score = best_label[1]
                
                new_assignments.append({
                    'company_id': row['company_id'],
                    'method': 'semantic_keyword_extraction',
                    'tier': 'tier2_synthetic',
                    'insurance_labels': label_name,
                    'confidence_score': confidence_score,
                    'num_labels_assigned': 1,
                    'all_labels_with_scores': f"{label_name} ({confidence_score:.3f})",
                    'synthetic_business_tags': json.dumps(synthetic_tags),
                    'extraction_details': json.dumps(extracted_tags)
                })
        
        print(f"\n✅ Created {len(new_assignments):,} new label assignments")
        return new_assignments
    
    def remove_noise_labels(self):
        """Remove labels used very rarely (likely noise)"""
        print(f"\n🧹 REMOVING NOISE LABELS")
        print("="*40)
        
        # Count label frequencies
        all_labels = []
        for _, row in self.tier1_companies.iterrows():
            if pd.notna(row['insurance_labels']) and row['insurance_labels'] != '':
                labels = row['insurance_labels'].split('; ')
                all_labels.extend(labels)
        
        label_counter = Counter(all_labels)
        
        # Identify noise labels (used ≤3 times)
        noise_labels = [label for label, count in label_counter.items() if count <= 3]
        
        print(f"📊 Identified {len(noise_labels)} noise labels (used ≤3 times):")
        for label in noise_labels:
            count = label_counter[label]
            print(f"   • {label}: {count} uses")
        
        return noise_labels
    
    def create_cleaned_dataset(self, new_assignments, noise_labels):
        """Create the final cleaned dataset with tier markings"""
        print(f"\n📊 CREATING CLEANED DATASET")
        print("="*40)
        
        # Start with original dataset
        cleaned_df = self.df.copy()
        
        # Add tier markings to existing data
        cleaned_df['data_tier'] = 'tier1_confident' 
        cleaned_df.loc[cleaned_df['num_labels_assigned'] == 0, 'data_tier'] = 'unlabeled'
        
        # Add method column
        cleaned_df['assignment_method'] = 'original_classification'
        
        # Apply new assignments
        for assignment in new_assignments:
            company_id = assignment['company_id']
            mask = cleaned_df['company_id'] == company_id
            
            if mask.any():
                # Update the unlabeled company with new assignment
                for key, value in assignment.items():
                    if key != 'company_id':
                        cleaned_df.loc[mask, key] = value
        
        # Remove noise labels from all assignments
        if noise_labels:
            print(f"🗑️ Removing noise labels from all assignments...")
            
            def clean_labels(labels_str):
                if pd.isna(labels_str) or labels_str == '':
                    return labels_str
                    
                labels = labels_str.split('; ')
                clean_labels = [label for label in labels if label not in noise_labels]
                return '; '.join(clean_labels) if clean_labels else ''
            
            def clean_scores(scores_str):
                if pd.isna(scores_str) or scores_str == '':
                    return scores_str
                    
                score_parts = scores_str.split('; ')
                clean_parts = []
                for part in score_parts:
                    label = part.split(' (')[0] if ' (' in part else part
                    if label not in noise_labels:
                        clean_parts.append(part)
                return '; '.join(clean_parts) if clean_parts else ''
            
            # Apply cleaning
            cleaned_df['insurance_labels'] = cleaned_df['insurance_labels'].apply(clean_labels)
            cleaned_df['all_labels_with_scores'] = cleaned_df['all_labels_with_scores'].apply(clean_scores)
            
            # Recalculate num_labels_assigned
            def count_labels(labels_str):
                if pd.isna(labels_str) or labels_str == '':
                    return 0
                return len([l for l in labels_str.split('; ') if l.strip()])
            
            cleaned_df['num_labels_assigned'] = cleaned_df['insurance_labels'].apply(count_labels)
        
        # Calculate final statistics
        total_companies = len(cleaned_df)
        companies_with_labels = len(cleaned_df[cleaned_df['num_labels_assigned'] > 0])
        coverage = (companies_with_labels / total_companies) * 100
        
        tier1_count = len(cleaned_df[cleaned_df['data_tier'] == 'tier1_confident'])
        tier2_forced_count = len(cleaned_df[cleaned_df['data_tier'] == 'tier2_forced'])
        tier2_synthetic_count = len(cleaned_df[cleaned_df['data_tier'] == 'tier2_synthetic'])
        still_unlabeled = len(cleaned_df[cleaned_df['num_labels_assigned'] == 0])
        
        print(f"\n📈 FINAL DATASET STATISTICS:")
        print(f"="*50)
        print(f"Total companies: {total_companies:,}")
        print(f"Companies with labels: {companies_with_labels:,} ({coverage:.1f}%)")
        print(f"Companies still unlabeled: {still_unlabeled:,} ({still_unlabeled/total_companies*100:.1f}%)")
        print(f"")
        print(f"📊 Breakdown by tier:")
        print(f"   Tier 1 (confident): {tier1_count:,} companies")
        print(f"   Tier 2 (forced): {tier2_forced_count:,} companies") 
        print(f"   Tier 2 (synthetic): {tier2_synthetic_count:,} companies")
        
        return cleaned_df
    
    def generate_verification_analytics(self, cleaned_df):
        """Generate verification analytics for the cleaned dataset"""
        print(f"\n🔍 GENERATING VERIFICATION ANALYTICS")
        print("="*50)
        
        verification_stats = {
            'sector_alignment': {},
            'category_alignment': {},
            'niche_alignment': {},
            'confidence_by_tier': {},
            'label_frequency_analysis': {}
        }
        
        # Analyze label frequencies by sector/category/niche
        for tier in ['tier1_confident', 'tier2_forced', 'tier2_synthetic']:
            tier_data = cleaned_df[cleaned_df['data_tier'] == tier]
            if len(tier_data) == 0:
                continue
                
            tier_stats = {
                'total_companies': len(tier_data),
                'total_assignments': tier_data['num_labels_assigned'].sum(),
                'avg_confidence': 0,
                'confidence_distribution': {}
            }
            
            # Extract confidence scores for this tier
            confidences = []
            for _, row in tier_data.iterrows():
                if pd.notna(row['all_labels_with_scores']):
                    scores_str = row['all_labels_with_scores']
                    score_parts = scores_str.split('; ')
                    for part in score_parts:
                        if '(' in part and ')' in part:
                            try:
                                score = float(part.split('(')[-1].replace(')', ''))
                                confidences.append(score)
                            except:
                                pass
            
            if confidences:
                tier_stats['avg_confidence'] = np.mean(confidences)
                tier_stats['confidence_distribution'] = {
                    'high_conf_pct': sum(1 for c in confidences if c >= 0.6) / len(confidences) * 100,
                    'med_conf_pct': sum(1 for c in confidences if 0.5 <= c < 0.6) / len(confidences) * 100,
                    'low_conf_pct': sum(1 for c in confidences if c < 0.5) / len(confidences) * 100
                }
            
            verification_stats['confidence_by_tier'][tier] = tier_stats
        
        # Save verification analytics
        with open('cleaned_dataset_analytics.json', 'w') as f:
            json.dump(verification_stats, f, indent=2, default=str)
        
        print(f"✅ Verification analytics saved to: cleaned_dataset_analytics.json")
        return verification_stats
    
    def run_full_cleaning(self):
        """Run the complete automated cleaning process"""
        print(f"\n🚀 STARTING AUTOMATED DATA CLEANING")
        print("="*80)
        
        try:
            # Step 1: Load current results
            self.load_current_results()
            
            # Step 2: Extract existing business tags corpus
            self.extract_existing_business_tags()
            
            # Step 3: Force-assign labels to unlabeled companies
            new_assignments = self.force_assign_best_labels()
            
            # Step 4: Identify and remove noise labels
            noise_labels = self.remove_noise_labels()
            
            # Step 5: Create cleaned dataset
            cleaned_df = self.create_cleaned_dataset(new_assignments, noise_labels)
            
            # Step 6: Generate verification analytics
            verification_stats = self.generate_verification_analytics(cleaned_df)
            
            # Step 7: Save cleaned dataset
            output_path = Path("src/data/output/cleaned_classification_results.csv")
            cleaned_df.to_csv(output_path, index=False)
            print(f"\n💾 Cleaned dataset saved to: {output_path}")
            
            print(f"\n🎉 AUTOMATED CLEANING COMPLETE!")
            print(f"✅ Ready for heatmap curation and ML pipeline development")
            
            return cleaned_df, verification_stats
            
        except Exception as e:
            print(f"❌ Automated cleaning failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None

def main():
    """Main execution function"""
    cleaner = AutomatedDataCleaner()
    cleaned_data, analytics = cleaner.run_full_cleaning()
    
    if cleaned_data is not None:
        print(f"\n📊 SUMMARY:")
        print(f"Input: 9,494 companies (63.3% labeled)")
        coverage = len(cleaned_data[cleaned_data['num_labels_assigned'] > 0]) / len(cleaned_data) * 100
        print(f"Output: 9,494 companies ({coverage:.1f}% labeled)")
        print(f"Improvement: +{coverage-63.3:.1f} percentage points")
        
        print(f"\n🎯 Next Steps:")
        print(f"1. Review cleaned_dataset_analytics.json")
        print(f"2. Run heatmap curation on cleaned dataset")
        print(f"3. Develop ML pipeline with tier-aware training")

if __name__ == "__main__":
    main()