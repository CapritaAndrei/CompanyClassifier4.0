#!/usr/bin/env python3
"""
Heatmap-based Data Cleaner
==========================
Applies frequency filtering and sector-based cleaning to remove noise and misplacements.

Rules:
1. Remove labels appearing ≤5 times total (frequency filtering)
2. Remove sector misplacements (if label appears 85%+ in one sector, remove from others)
3. Maintain "at least one label per company" requirement
"""

import pandas as pd
import json
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

class HeatmapCleaner:
    def __init__(self):
        self.df = None
        self.original_companies = 0
        self.original_labels = 0
        self.cleaning_stats = {
            'frequency_removed': 0,
            'sector_misplacements_removed': 0,
            'companies_affected': 0,
            'final_coverage': 0
        }
        
    def load_data(self, file_path="src/data/output/cleaned_classification_results.csv"):
        """Load the 99% coverage dataset"""
        print(f"🎯 HEATMAP CLEANER INITIALIZED")
        print("="*60)
        print(f"Goal: Apply frequency + sector-based cleaning")
        print(f"Rules: Remove ≤5 frequency labels + 85% sector misplacements")
        print("="*60)
        
        print(f"📂 Loading dataset: {file_path}")
        self.df = pd.read_csv(file_path)
        
        self.original_companies = len(self.df)
        
        # Count original labels
        original_label_count = 0
        for _, row in self.df.iterrows():
            if pd.notna(row['insurance_labels']) and str(row['insurance_labels']).strip():
                try:
                    # Parse semicolon-separated labels
                    labels_str = str(row['insurance_labels']).strip()
                    if labels_str and labels_str != '[]':
                        labels = [label.strip() for label in labels_str.split(';') if label.strip()]
                        original_label_count += len(labels)
                except:
                    pass
        
        self.original_labels = original_label_count
        
        print(f"✅ Loaded {self.original_companies:,} companies")
        print(f"✅ Total original labels: {self.original_labels:,}")
        
        return self.df
    
    def analyze_label_frequency(self):
        """Analyze frequency of each label across the entire dataset"""
        print(f"\n🔍 ANALYZING LABEL FREQUENCY")
        print("="*40)
        
        label_frequency = Counter()
        label_to_companies = defaultdict(list)
        
        for idx, row in self.df.iterrows():
            if pd.notna(row['insurance_labels']) and str(row['insurance_labels']).strip():
                try:
                    # Parse semicolon-separated labels
                    labels_str = str(row['insurance_labels']).strip()
                    if labels_str and labels_str != '[]':
                        labels = [label.strip() for label in labels_str.split(';') if label.strip()]
                        for label in labels:
                            label_frequency[label] += 1
                            label_to_companies[label].append(idx)
                except:
                    pass
        
        print(f"✅ Found {len(label_frequency):,} unique labels")
        
        # Find rare labels (≤5 occurrences) 
        rare_labels = {label for label, count in label_frequency.items() if count <= 5}
        
        print(f"📊 Frequency analysis:")
        print(f"   Labels with 1-2 occurrences: {sum(1 for c in label_frequency.values() if c <= 2):,}")
        print(f"   Labels with 3-5 occurrences: {sum(1 for c in label_frequency.values() if 3 <= c <= 5):,}")
        print(f"   Labels with >5 occurrences: {sum(1 for c in label_frequency.values() if c > 5):,}")
        print(f"   🗑️ Rare labels to remove (≤5): {len(rare_labels):,}")
        
        return label_frequency, label_to_companies, rare_labels
    
    def analyze_sector_distribution(self, label_frequency, label_to_companies):
        """Analyze sector distribution for each label to find misplacements"""
        print(f"\n🎯 ANALYZING SECTOR DISTRIBUTION")
        print("="*40)
        
        sector_misplacements = {}
        SECTOR_THRESHOLD = 0.90  # 85% threshold for sector dominance
        
        for label, company_indices in label_to_companies.items():
            if label_frequency[label] <= 5:  # Skip rare labels (will be removed anyway)
                continue
                
            # Count sector distribution for this label
            sector_counts = Counter()
            
            for idx in company_indices:
                sector = self.df.iloc[idx]['sector']
                if pd.notna(sector):
                    sector_counts[sector] += 1
            
            if len(sector_counts) <= 1:  # Only one sector, no misplacement possible
                continue
            
            total_count = sum(sector_counts.values())
            if total_count < 3:  # Need at least 3 occurrences to detect meaningful patterns
                continue
                
            # Find dominant sector
            dominant_sector, dominant_count = sector_counts.most_common(1)[0]
            dominant_percentage = dominant_count / total_count
            
            if dominant_percentage >= SECTOR_THRESHOLD:
                # This label is misplaced in other sectors
                misplaced_sectors = []
                for sector, count in sector_counts.items():
                    if sector != dominant_sector:
                        misplaced_sectors.append((sector, count))
                
                if misplaced_sectors:
                    sector_misplacements[label] = {
                        'dominant_sector': dominant_sector,
                        'dominant_count': dominant_count,
                        'dominant_percentage': dominant_percentage,
                        'misplaced_sectors': misplaced_sectors,
                        'total_misplaced': sum(count for _, count in misplaced_sectors)
                    }
        
        print(f"✅ Analyzed sector distribution for {len(label_to_companies):,} labels")
        print(f"📊 Sector misplacement analysis:")
        print(f"   Labels with sector misplacements: {len(sector_misplacements):,}")
        
        # Show top misplacements
        if sector_misplacements:
            print(f"   🔍 Top sector misplacements:")
            sorted_misplacements = sorted(
                sector_misplacements.items(), 
                key=lambda x: x[1]['total_misplaced'], 
                reverse=True
            )
            
            for i, (label, data) in enumerate(sorted_misplacements[:10]):
                print(f"      {i+1}. {label}")
                print(f"         Dominant: {data['dominant_sector']} ({data['dominant_percentage']:.1%})")
                print(f"         Misplaced: {data['total_misplaced']} companies in {len(data['misplaced_sectors'])} sectors")
        
        return sector_misplacements
    
    def apply_cleaning(self, rare_labels, sector_misplacements):
        """Apply cleaning rules to remove rare labels and sector misplacements"""
        print(f"\n🧹 APPLYING CLEANING RULES")
        print("="*40)
        
        companies_modified = 0
        frequency_removals = 0
        sector_removals = 0
        
        for idx, row in self.df.iterrows():
            if pd.notna(row['insurance_labels']) and str(row['insurance_labels']).strip():
                try:
                    # Parse semicolon-separated labels
                    labels_str = str(row['insurance_labels']).strip()
                    if labels_str and labels_str != '[]':
                        original_labels = [label.strip() for label in labels_str.split(';') if label.strip()]
                        cleaned_labels = []
                        company_sector = row['sector']
                        
                        for label in original_labels:
                            # Rule 1: Remove rare labels (≤5 occurrences)
                            if label in rare_labels:
                                frequency_removals += 1
                                continue
                            
                            # Rule 2: Remove sector misplacements
                            if (label in sector_misplacements and 
                                pd.notna(company_sector)):
                                
                                misplacement_data = sector_misplacements[label]
                                dominant_sector = misplacement_data['dominant_sector']
                                
                                if company_sector != dominant_sector:
                                    # This label is misplaced in this sector
                                    sector_removals += 1
                                    continue
                            
                            # Keep this label
                            cleaned_labels.append(label)
                        
                        # Ensure at least one label per company
                        if not cleaned_labels and original_labels:
                            # If all labels were removed, keep the first original label
                            cleaned_labels = [original_labels[0]]
                            print(f"⚠️ Kept 1 label for company {idx} to maintain coverage")
                        
                        # Update if changes were made
                        if cleaned_labels != original_labels:
                            companies_modified += 1
                            # Save as semicolon-separated string
                            self.df.at[idx, 'insurance_labels'] = '; '.join(cleaned_labels)
                
                except Exception as e:
                    print(f"⚠️ Error processing company {idx}: {e}")
                    continue
        
        self.cleaning_stats.update({
            'frequency_removed': frequency_removals,
            'sector_misplacements_removed': sector_removals,
            'companies_affected': companies_modified
        })
        
        print(f"✅ Cleaning completed:")
        print(f"   Companies modified: {companies_modified:,}")
        print(f"   Rare labels removed: {frequency_removals:,}")
        print(f"   Sector misplacements removed: {sector_removals:,}")
        print(f"   Total labels removed: {frequency_removals + sector_removals:,}")
    
    def calculate_final_stats(self):
        """Calculate final statistics after cleaning"""
        print(f"\n📊 CALCULATING FINAL STATISTICS")
        print("="*40)
        
        companies_with_labels = 0
        total_final_labels = 0
        final_label_frequency = Counter()
        
        for _, row in self.df.iterrows():
            if pd.notna(row['insurance_labels']) and str(row['insurance_labels']).strip():
                try:
                    # Parse semicolon-separated labels
                    labels_str = str(row['insurance_labels']).strip()
                    if labels_str and labels_str != '[]':
                        labels = [label.strip() for label in labels_str.split(';') if label.strip()]
                        if labels:
                            companies_with_labels += 1
                            total_final_labels += len(labels)
                            for label in labels:
                                final_label_frequency[label] += 1
                except:
                    pass
        
        coverage = companies_with_labels / self.original_companies if self.original_companies > 0 else 0
        self.cleaning_stats['final_coverage'] = coverage
        
        print(f"✅ Final statistics:")
        print(f"   Companies with labels: {companies_with_labels:,}")
        print(f"   Final coverage: {coverage:.1%}")
        print(f"   Total final labels: {total_final_labels:,}")
        print(f"   Unique labels remaining: {len(final_label_frequency):,}")
        print(f"   Average labels per company: {total_final_labels/companies_with_labels:.1f}")
        
        return final_label_frequency
    
    def save_results(self):
        """Save cleaned results and statistics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save cleaned dataset
        output_file = f"src/data/output/heatmap_cleaned_results_{timestamp}.csv"
        self.df.to_csv(output_file, index=False)
        
        # Save cleaning statistics
        stats_file = f"heatmap_cleaning_stats_{timestamp}.json"
        
        detailed_stats = {
            'cleaning_timestamp': timestamp,
            'original_companies': self.original_companies,
            'original_labels': self.original_labels,
            'cleaning_rules': {
                'frequency_threshold': '≤5 occurrences',
                'sector_threshold': '85% dominance'
            },
            'cleaning_results': self.cleaning_stats,
            'labels_removed_total': self.cleaning_stats['frequency_removed'] + self.cleaning_stats['sector_misplacements_removed'],
            'final_coverage': f"{self.cleaning_stats['final_coverage']:.1%}"
        }
        
        with open(stats_file, 'w') as f:
            json.dump(detailed_stats, f, indent=2)
        
        print(f"\n💾 RESULTS SAVED")
        print("="*30)
        print(f"📊 Cleaned dataset: {output_file}")
        print(f"📈 Cleaning stats: {stats_file}")
        
        return output_file
    
    def run_heatmap_cleaning(self):
        """Execute the complete heatmap cleaning pipeline"""
        try:
            # Load data
            self.load_data()
            
            # Analyze label frequency
            label_frequency, label_to_companies, rare_labels = self.analyze_label_frequency()
            
            # Analyze sector distribution
            sector_misplacements = self.analyze_sector_distribution(label_frequency, label_to_companies)
            
            # Apply cleaning
            self.apply_cleaning(rare_labels, sector_misplacements)
            
            # Calculate final stats
            self.calculate_final_stats()
            
            # Save results
            output_file = self.save_results()
            
            print(f"\n🎉 HEATMAP CLEANING COMPLETE!")
            print("="*50)
            print(f"✅ Original: {self.original_companies:,} companies, {self.original_labels:,} labels")
            print(f"✅ Final: {self.cleaning_stats['final_coverage']:.1%} coverage")
            print(f"✅ Removed: {self.cleaning_stats['frequency_removed'] + self.cleaning_stats['sector_misplacements_removed']:,} noisy labels")
            print(f"✅ Output: {output_file}")
            
            return output_file
            
        except Exception as e:
            print(f"❌ Heatmap cleaning failed: {e}")
            raise

if __name__ == "__main__":
    cleaner = HeatmapCleaner()
    cleaner.run_heatmap_cleaning()