"""
Master Map Consolidation System
================================

Priority-based consolidation of NAICS mappings:
1. Baseline: Embedding mappings (assume all correct)
2. Supplement: Exact matches (only if not in embedding)
3. Supplement: Hierarchical mappings (only if not in embedding or exact)

For conflicts, embedding mappings always win.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from datetime import datetime
from collections import defaultdict


class MasterMapConsolidator:
    """
    Consolidates multiple NAICS mapping sources into a unified master map
    """
    
    def __init__(self):
        self.embedding_mappings = {}
        self.exact_mappings = {}
        self.hierarchical_mappings = {}
        self.master_map = {}
        self.stats = {
            'embedding_labels': 0,
            'exact_supplements': 0,
            'hierarchical_supplements': 0,
            'total_labels': 0,
            'total_naics_codes': 0,
            'conflicts_resolved': 0
        }
    
    def load_mappings(self):
        """Load all mapping files"""
        print("📥 Loading mapping files...")
        
        # Load embedding mappings (baseline)
        with open('data/processed/embedding_naics_mappings.json', 'r') as f:
            embedding_data = json.load(f)
        
        # Load exact match mappings
        with open('exact_match_mappings.json', 'r') as f:
            exact_data = json.load(f)
        
        # Load hierarchical mappings
        with open('data/processed/enhanced_hierarchical_mappings.json', 'r') as f:
            hierarchical_data = json.load(f)
        
        # Process embedding mappings (baseline)
        print("🔧 Processing embedding mappings...")
        self.embedding_mappings = self._process_embedding_mappings(embedding_data)
        
        # Process exact match mappings
        print("🔧 Processing exact match mappings...")
        self.exact_mappings = self._process_exact_mappings(exact_data)
        
        # Process hierarchical mappings
        print("🔧 Processing hierarchical mappings...")
        self.hierarchical_mappings = self._process_hierarchical_mappings(hierarchical_data)
        
        print(f"✅ Loaded mappings:")
        print(f"   Embedding: {len(self.embedding_mappings)} labels")
        print(f"   Exact: {len(self.exact_mappings)} labels")
        print(f"   Hierarchical: {len(self.hierarchical_mappings)} labels")
    
    def _process_embedding_mappings(self, data: List[Dict]) -> Dict[str, List[Dict]]:
        """Process embedding mappings (baseline - assume all correct)"""
        processed = {}
        
        for item in data:
            label = item['label']
            naics_codes = []
            
            # Extract all approved NAICS codes
            for match in item.get('approved_matches', []):
                # Get unique codes from subordinate_codes
                for sub_code in match.get('subordinate_codes', []):
                    code_info = {
                        'naics_code': sub_code['code'],
                        'description': sub_code['description'],
                        'naics_version': sub_code['naics_version'],
                        'match_type': match['match_type'],
                        'similarity_score': match.get('similarity_score', 1.0),
                        'source': 'embedding_baseline'
                    }
                    
                    # Avoid duplicates
                    if not any(existing['naics_code'] == code_info['naics_code'] and 
                             existing['naics_version'] == code_info['naics_version'] 
                             for existing in naics_codes):
                        naics_codes.append(code_info)
            
            if naics_codes:
                processed[label] = naics_codes
        
        return processed
    
    def _process_exact_mappings(self, data: Dict) -> Dict[str, List[Dict]]:
        """Process exact match mappings"""
        processed = {}
        
        for label, matches in data.get('successful_mappings', {}).items():
            naics_codes = []
            
            for match in matches:
                # Clean up NAICS code (remove .0 suffix)
                naics_code = str(match['naics_code']).replace('.0', '')
                
                code_info = {
                    'naics_code': naics_code,
                    'description': match['match_text'],
                    'naics_version': 'unknown',  # Exact matches don't specify version
                    'match_type': 'exact_text',
                    'similarity_score': 1.0,
                    'source': 'exact_match'
                }
                naics_codes.append(code_info)
            
            if naics_codes:
                processed[label] = naics_codes
        
        return processed
    
    def _process_hierarchical_mappings(self, data: List[Dict]) -> Dict[str, List[Dict]]:
        """Process hierarchical mappings"""
        processed = {}
        
        for item in data:
            label = item['label']
            naics_codes = []
            
            # Extract all approved NAICS codes
            for match in item.get('approved_matches', []):
                # Get unique codes from subordinate_codes
                for sub_code in match.get('subordinate_codes', []):
                    code_info = {
                        'naics_code': sub_code['code'],
                        'description': sub_code['description'],
                        'naics_version': sub_code['naics_version'],
                        'match_type': match['match_type'],
                        'similarity_score': 1.0,  # Hierarchical matches are exact
                        'source': 'hierarchical_exact'
                    }
                    
                    # Avoid duplicates
                    if not any(existing['naics_code'] == code_info['naics_code'] and 
                             existing['naics_version'] == code_info['naics_version'] 
                             for existing in naics_codes):
                        naics_codes.append(code_info)
            
            if naics_codes:
                processed[label] = naics_codes
        
        return processed
    
    def consolidate_mappings(self):
        """
        Consolidate all mappings with priority:
        1. Embedding mappings (baseline)
        2. Exact matches (supplement)
        3. Hierarchical mappings (supplement)
        """
        print("🔄 Consolidating mappings...")
        
        # Start with embedding mappings as baseline
        self.master_map = dict(self.embedding_mappings)
        self.stats['embedding_labels'] = len(self.embedding_mappings)
        
        # Add exact matches for labels not in embedding
        for label, codes in self.exact_mappings.items():
            if label not in self.master_map:
                self.master_map[label] = codes
                self.stats['exact_supplements'] += 1
                print(f"   ➕ Added exact match for: {label}")
            else:
                # Check if exact match has additional codes
                existing_codes = {code['naics_code'] for code in self.master_map[label]}
                new_codes = [code for code in codes if code['naics_code'] not in existing_codes]
                
                if new_codes:
                    self.master_map[label].extend(new_codes)
                    print(f"   ➕ Added supplementary codes for: {label}")
                    self.stats['exact_supplements'] += len(new_codes)
                else:
                    self.stats['conflicts_resolved'] += 1
        
        # Add hierarchical mappings for labels not in embedding or exact
        for label, codes in self.hierarchical_mappings.items():
            if label not in self.master_map:
                self.master_map[label] = codes
                self.stats['hierarchical_supplements'] += 1
                print(f"   ➕ Added hierarchical match for: {label}")
            else:
                # Check if hierarchical match has additional codes
                existing_codes = {code['naics_code'] for code in self.master_map[label]}
                new_codes = [code for code in codes if code['naics_code'] not in existing_codes]
                
                if new_codes:
                    self.master_map[label].extend(new_codes)
                    print(f"   ➕ Added supplementary hierarchical codes for: {label}")
                    self.stats['hierarchical_supplements'] += len(new_codes)
                else:
                    self.stats['conflicts_resolved'] += 1
        
        # Calculate final statistics
        self.stats['total_labels'] = len(self.master_map)
        self.stats['total_naics_codes'] = sum(len(codes) for codes in self.master_map.values())
        
        print(f"\n✅ Consolidation complete!")
        print(f"   Total labels: {self.stats['total_labels']}")
        print(f"   Total NAICS codes: {self.stats['total_naics_codes']}")
        print(f"   Conflicts resolved: {self.stats['conflicts_resolved']}")
    
    def analyze_coverage(self):
        """Analyze the coverage and quality of the master map"""
        print("\n📊 Master Map Coverage Analysis")
        print("=" * 50)
        
        # Source distribution
        source_counts = defaultdict(int)
        for label, codes in self.master_map.items():
            for code in codes:
                source_counts[code['source']] += 1
        
        print(f"📈 Source Distribution:")
        for source, count in source_counts.items():
            print(f"   {source}: {count} mappings")
        
        # NAICS version distribution
        version_counts = defaultdict(int)
        for label, codes in self.master_map.items():
            for code in codes:
                version_counts[code['naics_version']] += 1
        
        print(f"\n📅 NAICS Version Distribution:")
        for version, count in version_counts.items():
            print(f"   {version}: {count} mappings")
        
        # Quality metrics
        high_quality = 0
        medium_quality = 0
        for label, codes in self.master_map.items():
            max_score = max(code.get('similarity_score', 0) for code in codes)
            if max_score >= 0.9:
                high_quality += 1
            elif max_score >= 0.75:
                medium_quality += 1
        
        print(f"\n🎯 Quality Distribution:")
        print(f"   High quality (≥0.9): {high_quality} labels")
        print(f"   Medium quality (0.75-0.9): {medium_quality} labels")
        print(f"   Lower quality (<0.75): {len(self.master_map) - high_quality - medium_quality} labels")
        
        # Top mapped labels
        top_labels = sorted(self.master_map.items(), key=lambda x: len(x[1]), reverse=True)[:10]
        print(f"\n🔝 Top 10 Labels by NAICS Code Count:")
        for label, codes in top_labels:
            print(f"   {label}: {len(codes)} codes")
    
    def save_master_map(self, output_path: str = "data/processed/master_insurance_to_naics_mapping.json"):
        """Save the consolidated master map"""
        print(f"\n💾 Saving master map to {output_path}")
        
        # Create comprehensive output
        output_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "consolidation_strategy": "embedding_baseline_with_supplements",
                "priority_order": [
                    "embedding_mappings (baseline)",
                    "exact_match_mappings (supplement)",
                    "hierarchical_mappings (supplement)"
                ],
                "statistics": self.stats
            },
            "mappings": self.master_map
        }
        
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save master map
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Also save a simplified version for easy use
        simplified_map = {}
        for label, codes in self.master_map.items():
            simplified_map[label] = [
                {
                    'naics_code': code['naics_code'],
                    'description': code['description'],
                    'source': code['source'],
                    'similarity_score': code.get('similarity_score', 1.0)
                }
                for code in codes
            ]
        
        simplified_path = output_path.replace('.json', '_simplified.json')
        with open(simplified_path, 'w') as f:
            json.dump(simplified_map, f, indent=2)
        
        print(f"✅ Master map saved!")
        print(f"   Full version: {output_path}")
        print(f"   Simplified: {simplified_path}")
    
    def generate_report(self):
        """Generate a comprehensive consolidation report"""
        print("\n" + "="*70)
        print("🎯 MASTER MAP CONSOLIDATION REPORT")
        print("="*70)
        
        print(f"\n📊 CONSOLIDATION STATISTICS:")
        print(f"   Embedding baseline labels: {self.stats['embedding_labels']}")
        print(f"   Exact match supplements: {self.stats['exact_supplements']}")
        print(f"   Hierarchical supplements: {self.stats['hierarchical_supplements']}")
        print(f"   Total labels in master map: {self.stats['total_labels']}")
        print(f"   Total NAICS codes: {self.stats['total_naics_codes']}")
        print(f"   Conflicts resolved (embedding priority): {self.stats['conflicts_resolved']}")
        
        # Coverage improvement
        baseline_coverage = self.stats['embedding_labels']
        total_coverage = self.stats['total_labels']
        improvement = ((total_coverage - baseline_coverage) / baseline_coverage) * 100
        
        print(f"\n📈 COVERAGE IMPROVEMENT:")
        print(f"   Baseline coverage: {baseline_coverage} labels")
        print(f"   Final coverage: {total_coverage} labels")
        print(f"   Improvement: +{improvement:.1f}%")
        
        self.analyze_coverage()
        
        print("\n" + "="*70)
        print("🚀 READY FOR ACTIVE LEARNING PIPELINE!")
        print("="*70)


def main():
    """Main consolidation workflow"""
    print("🎯 Master Map Consolidation System")
    print("=" * 50)
    
    # Initialize consolidator
    consolidator = MasterMapConsolidator()
    
    # Load all mapping files
    consolidator.load_mappings()
    
    # Consolidate mappings
    consolidator.consolidate_mappings()
    
    # Analyze coverage
    consolidator.analyze_coverage()
    
    # Save master map
    consolidator.save_master_map()
    
    # Generate report
    consolidator.generate_report()


if __name__ == "__main__":
    main() 