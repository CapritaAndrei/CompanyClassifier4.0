#!/usr/bin/env python3
"""
Test Exact Matching for Insurance Labels
========================================

Simple script to test exact matching for the first few insurance taxonomy labels
against NAICS codes, showing results layer by layer.
"""

import pandas as pd
import re
from difflib import SequenceMatcher
from typing import List, Tuple

def clean_text(text: str) -> str:
    """Clean and normalize text for matching"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase and remove special characters
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def find_exact_matches(insurance_label: str, naics_df: pd.DataFrame, threshold: float = 0.6) -> List[Tuple[str, str, float]]:
    """Find exact matches for a single insurance label"""
    matches = []
    cleaned_label = clean_text(insurance_label)
    
    print(f"\n🔍 Searching for: '{insurance_label}'")
    print(f"   Cleaned: '{cleaned_label}'")
    
    for _, row in naics_df.iterrows():
        cleaned_desc = clean_text(row['description'])
        
        # Calculate similarity ratio
        similarity = SequenceMatcher(None, cleaned_label, cleaned_desc).ratio()
        
        if similarity >= threshold:
            matches.append((row['naics_code'], row['description'], similarity))
    
    # Sort by similarity
    matches.sort(key=lambda x: x[2], reverse=True)
    return matches

def main():
    print("🔄 Loading data...")
    
    # Load insurance taxonomy labels (first 6 for testing)
    insurance_df = pd.read_csv('data/input/insurance_taxonomy - insurance_taxonomy.csv')
    test_labels = insurance_df['label'].head(6).tolist()
    
    print(f"✅ Testing with {len(test_labels)} labels:")
    for i, label in enumerate(test_labels, 1):
        print(f"   {i}. {label}")
    
    # Load NAICS detailed index
    naics_index = pd.read_excel('2017_NAICS_Index_File.xlsx')
    naics_index.columns = ['naics_code', 'description']
    naics_index = naics_index.dropna()
    naics_index['naics_code'] = naics_index['naics_code'].astype(str)
    
    print(f"✅ Loaded {len(naics_index)} NAICS descriptions")
    
    # Load official NAICS codes
    naics_codes = pd.read_excel('6-digit_2017_Codes.xlsx')
    naics_codes = naics_codes.dropna()
    naics_codes.columns = ['naics_code', 'title', 'extra']
    naics_codes = naics_codes[['naics_code', 'title']]
    naics_codes['naics_code'] = naics_codes['naics_code'].astype(int).astype(str)
    naics_codes = naics_codes.rename(columns={'title': 'description'})
    
    print(f"✅ Loaded {len(naics_codes)} official NAICS titles")
    
    print("\n" + "="*80)
    print("🎯 EXACT MATCHING ANALYSIS")
    print("="*80)
    
    # Test each label
    for label in test_labels:
        print(f"\n{'='*60}")
        print(f"📋 TESTING: {label}")
        print(f"{'='*60}")
        
        # Layer 1: Search in official NAICS titles (high precision)
        print(f"\n🔍 Layer 1: Official NAICS Titles (6-digit codes)")
        official_matches = find_exact_matches(label, naics_codes, threshold=0.6)
        
        if official_matches:
            print(f"   ✅ Found {len(official_matches)} matches:")
            for code, desc, sim in official_matches[:5]:  # Show top 5
                print(f"      {code}: {desc} (similarity: {sim:.3f})")
        else:
            print(f"   ❌ No matches found in official titles")
        
        # Layer 2: Search in detailed NAICS index (broader coverage)
        print(f"\n🔍 Layer 2: Detailed NAICS Index (20K+ descriptions)")
        detailed_matches = find_exact_matches(label, naics_index, threshold=0.6)
        
        if detailed_matches:
            print(f"   ✅ Found {len(detailed_matches)} matches:")
            for code, desc, sim in detailed_matches[:10]:  # Show top 10
                print(f"      {code}: {desc[:80]}... (similarity: {sim:.3f})")
        else:
            print(f"   ❌ No matches found in detailed index")
        
        # Layer 3: Lower threshold search for potential matches
        print(f"\n🔍 Layer 3: Lower Threshold Search (similarity > 0.4)")
        low_threshold_matches = find_exact_matches(label, naics_index, threshold=0.4)
        
        if low_threshold_matches:
            # Filter out matches we already found
            new_matches = [m for m in low_threshold_matches if m[2] < 0.6]
            if new_matches:
                print(f"   ⚠️  Found {len(new_matches)} potential matches:")
                for code, desc, sim in new_matches[:5]:  # Show top 5
                    print(f"      {code}: {desc[:80]}... (similarity: {sim:.3f})")
            else:
                print(f"   ✅ No additional low-threshold matches")
        else:
            print(f"   ❌ No low-threshold matches found")
        
        # Summary for this label
        total_matches = len(official_matches) + len(detailed_matches)
        print(f"\n📊 SUMMARY for '{label}':")
        print(f"   • Official title matches: {len(official_matches)}")
        print(f"   • Detailed index matches: {len(detailed_matches)}")
        print(f"   • Total high-quality matches: {total_matches}")
        
        if total_matches > 0:
            best_match = official_matches[0] if official_matches else detailed_matches[0]
            print(f"   🎯 Best match: {best_match[0]} - {best_match[1][:60]}... ({best_match[2]:.3f})")
    
    print(f"\n{'='*80}")
    print("🏁 EXACT MATCHING TEST COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main() 