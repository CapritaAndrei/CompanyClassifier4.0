#!/usr/bin/env python3
"""
Filter Matches by Count
=======================

Apply filtering rule: If a label has more than 7 matches, keep only the first 5.
This removes the noise that comes from overly broad matching.
"""

import json

def filter_matches_by_count():
    """Filter matches according to the count rule"""
    
    # Load the detailed matches
    with open('data/processed/detailed_exact_matches_for_review.json', 'r') as f:
        detailed_matches = json.load(f)
    
    filtered_matches = {}
    filtering_stats = {
        'labels_processed': 0,
        'labels_filtered': 0,
        'total_matches_before': 0,
        'total_matches_after': 0
    }
    
    print("🔧 FILTERING MATCHES BY COUNT RULE")
    print("=" * 50)
    print("Rule: If >7 matches, keep only first 5")
    print()
    
    for label, matches in detailed_matches.items():
        filtering_stats['labels_processed'] += 1
        filtering_stats['total_matches_before'] += len(matches)
        
        if len(matches) > 7:
            # Too many matches - keep only first 5
            filtered_matches[label] = matches[:5]
            filtering_stats['labels_filtered'] += 1
            filtering_stats['total_matches_after'] += 5
            
            print(f"🔽 {label}: {len(matches)} → 5 matches (filtered)")
        else:
            # Good number of matches - keep all
            filtered_matches[label] = matches
            filtering_stats['total_matches_after'] += len(matches)
            
            print(f"✅ {label}: {len(matches)} matches (kept all)")
    
    return filtered_matches, filtering_stats

def analyze_filtered_results(filtered_matches, stats):
    """Analyze the filtering results"""
    
    print(f"\n📊 FILTERING STATISTICS:")
    print(f"   • Labels processed: {stats['labels_processed']}")
    print(f"   • Labels filtered: {stats['labels_filtered']}")
    print(f"   • Total matches before: {stats['total_matches_before']:,}")
    print(f"   • Total matches after: {stats['total_matches_after']:,}")
    print(f"   • Matches removed: {stats['total_matches_before'] - stats['total_matches_after']:,}")
    print(f"   • Reduction: {(1 - stats['total_matches_after']/stats['total_matches_before'])*100:.1f}%")
    
    # Show distribution of match counts
    match_counts = [len(matches) for matches in filtered_matches.values()]
    from collections import Counter
    count_distribution = Counter(match_counts)
    
    print(f"\n📈 MATCH COUNT DISTRIBUTION (after filtering):")
    for count in sorted(count_distribution.keys()):
        print(f"   {count} matches: {count_distribution[count]} labels")
    
    # Show examples of filtered labels
    print(f"\n📋 EXAMPLES OF FILTERED LABELS:")
    filtered_examples = [(label, len(matches)) for label, matches in filtered_matches.items() if len(matches) == 5]
    
    # Load original to show what was filtered
    with open('data/processed/detailed_exact_matches_for_review.json', 'r') as f:
        original_matches = json.load(f)
    
    for label, filtered_count in filtered_examples[:10]:
        original_count = len(original_matches[label])
        if original_count > 7:
            print(f"   • {label}: {original_count} → {filtered_count}")
            # Show first 5 kept
            for i, match in enumerate(filtered_matches[label], 1):
                print(f"      {i}. {match['naics_code']}: {match['description'][:50]}...")

def main():
    """Run the filtering process"""
    
    # Apply filtering
    filtered_matches, stats = filter_matches_by_count()
    
    # Analyze results
    analyze_filtered_results(filtered_matches, stats)
    
    # Save filtered results
    with open('data/processed/filtered_exact_matches_for_review.json', 'w') as f:
        json.dump(filtered_matches, f, indent=2)
    
    # Create summary for easier review
    summary = {}
    for label, matches in filtered_matches.items():
        summary[label] = {
            'match_count': len(matches),
            'first_match': f"{matches[0]['naics_code']}: {matches[0]['description']}" if matches else "No matches",
            'all_matches': [f"{m['naics_code']}: {m['description']}" for m in matches]
        }
    
    with open('data/processed/filtered_matches_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 FILES SAVED:")
    print(f"   • data/processed/filtered_exact_matches_for_review.json (filtered matches)")
    print(f"   • data/processed/filtered_matches_summary.json (summary for review)")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. Review filtered_matches_summary.json for easier manual selection")
    print(f"   2. Much cleaner data with {stats['total_matches_after']:,} matches vs {stats['total_matches_before']:,}")
    print(f"   3. Quality should be much better with first-match priority")

if __name__ == "__main__":
    main() 