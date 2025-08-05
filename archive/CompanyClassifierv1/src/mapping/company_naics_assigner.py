"""
Company NAICS Assigner
Assigns NAICS codes to companies based on their Sector → Category → Niche combinations.
"""

import pandas as pd
import os


class CompanyNAICSAssigner:
    """Assigns NAICS codes to companies using pre-computed mappings."""
    
    def __init__(self, mappings_file="data/cache/naics_mappings/naics_mappings.csv"):
        """Initialize with NAICS mappings."""
        self.mappings_file = mappings_file
        self.mappings = None
        self.load_mappings()
        
    def load_mappings(self):
        """Load the NAICS mappings."""
        print(f"Loading NAICS mappings from {self.mappings_file}...")
        self.mappings = pd.read_csv(self.mappings_file)
        print(f"Loaded {len(self.mappings)} NAICS mappings")
        
        # Create lookup dictionary for faster assignment
        self.mapping_dict = {}
        for _, row in self.mappings.iterrows():
            key = (row['sector'], row['category'], row['niche'])
            self.mapping_dict[key] = {
                'naics_code': row['best_naics_code'],
                'naics_description': row['best_naics_description'],
                'similarity_score': row['best_similarity_score']
            }
        
    def assign_naics_to_companies(self, companies_file="data/input/ml_insurance_challenge.csv"):
        """Assign NAICS codes to all companies."""
        print(f"Loading companies from {companies_file}...")
        companies = pd.read_csv(companies_file)
        print(f"Loaded {len(companies)} companies")
        
        # Assign NAICS codes
        print("Assigning NAICS codes to companies...")
        
        naics_codes = []
        naics_descriptions = []
        naics_similarity_scores = []
        
        for _, company in companies.iterrows():
            key = (company['sector'], company['category'], company['niche'])
            
            if key in self.mapping_dict:
                mapping = self.mapping_dict[key]
                naics_codes.append(mapping['naics_code'])
                naics_descriptions.append(mapping['naics_description'])
                naics_similarity_scores.append(mapping['similarity_score'])
            else:
                # This shouldn't happen if mappings are complete
                naics_codes.append(None)
                naics_descriptions.append(None)
                naics_similarity_scores.append(None)
        
        # Add NAICS columns to companies dataframe
        companies_with_naics = companies.copy()
        companies_with_naics['naics_code'] = naics_codes
        companies_with_naics['naics_description'] = naics_descriptions
        companies_with_naics['naics_similarity_score'] = naics_similarity_scores
        
        return companies_with_naics
    
    def get_assignment_statistics(self, companies_with_naics):
        """Get statistics about NAICS assignments."""
        total_companies = len(companies_with_naics)
        assigned_companies = companies_with_naics['naics_code'].notna().sum()
        assignment_rate = assigned_companies / total_companies
        
        # Confidence level distribution
        high_confidence = (companies_with_naics['naics_similarity_score'] > 0.7).sum()
        medium_confidence = ((companies_with_naics['naics_similarity_score'] >= 0.5) & 
                           (companies_with_naics['naics_similarity_score'] <= 0.7)).sum()
        low_confidence = (companies_with_naics['naics_similarity_score'] < 0.5).sum()
        
        stats = {
            'total_companies': total_companies,
            'assigned_companies': assigned_companies,
            'assignment_rate': assignment_rate,
            'high_confidence_assignments': high_confidence,
            'medium_confidence_assignments': medium_confidence,
            'low_confidence_assignments': low_confidence,
            'average_similarity_score': companies_with_naics['naics_similarity_score'].mean()
        }
        
        return stats
    
    def save_companies_with_naics(self, companies_with_naics, output_file="data/output/companies_with_naics.csv"):
        """Save companies with NAICS codes to file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        companies_with_naics.to_csv(output_file, index=False)
        print(f"Saved companies with NAICS codes to {output_file}")
        return output_file
    
    def analyze_naics_distribution(self, companies_with_naics):
        """Analyze the distribution of NAICS codes."""
        print("\n=== NAICS Distribution Analysis ===")
        
        # Top 20 most common NAICS codes
        naics_counts = companies_with_naics['naics_code'].value_counts()
        print(f"Total unique NAICS codes assigned: {len(naics_counts)}")
        
        print("\nTop 20 most common NAICS codes:")
        for i, (naics_code, count) in enumerate(naics_counts.head(20).items(), 1):
            description = companies_with_naics[companies_with_naics['naics_code'] == naics_code]['naics_description'].iloc[0]
            percentage = (count / len(companies_with_naics)) * 100
            print(f"{i:2d}. {naics_code} - {description[:60]}... ({count} companies, {percentage:.1f}%)")
        
        # Sector distribution
        print("\n=== NAICS by Sector ===")
        sector_naics = companies_with_naics.groupby('sector')['naics_code'].nunique().sort_values(ascending=False)
        for sector, unique_naics in sector_naics.items():
            total_companies_in_sector = (companies_with_naics['sector'] == sector).sum()
            print(f"{sector}: {unique_naics} unique NAICS codes ({total_companies_in_sector} companies)")


def assign_naics_to_all_companies():
    """Main function to assign NAICS codes to all companies."""
    assigner = CompanyNAICSAssigner()
    
    # Assign NAICS codes
    companies_with_naics = assigner.assign_naics_to_companies()
    
    # Get statistics
    stats = assigner.get_assignment_statistics(companies_with_naics)
    
    # Print results
    print("\n=== NAICS Assignment Results ===")
    print(f"Total companies: {stats['total_companies']}")
    print(f"Successfully assigned NAICS codes: {stats['assigned_companies']}")
    print(f"Assignment rate: {stats['assignment_rate']:.1%}")
    print(f"Average similarity score: {stats['average_similarity_score']:.3f}")
    print(f"High confidence assignments (>0.7): {stats['high_confidence_assignments']} ({stats['high_confidence_assignments']/stats['total_companies']*100:.1f}%)")
    print(f"Medium confidence assignments (0.5-0.7): {stats['medium_confidence_assignments']} ({stats['medium_confidence_assignments']/stats['total_companies']*100:.1f}%)")
    print(f"Low confidence assignments (<0.5): {stats['low_confidence_assignments']} ({stats['low_confidence_assignments']/stats['total_companies']*100:.1f}%)")
    
    # Analyze distribution
    assigner.analyze_naics_distribution(companies_with_naics)
    
    # Save results
    output_file = assigner.save_companies_with_naics(companies_with_naics)
    
    print(f"\n=== Complete! ===")
    print(f"Companies with NAICS codes saved to: {output_file}")
    print("Each company now has:")
    print("  - naics_code: Official NAICS industry code")
    print("  - naics_description: Human-readable industry description")
    print("  - naics_similarity_score: Confidence score (0-1)")
    
    return companies_with_naics


if __name__ == "__main__":
    companies_with_naics = assign_naics_to_all_companies() 