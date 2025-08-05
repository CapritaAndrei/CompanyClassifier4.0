"""
Insurance Company Classification System - Main Entry Point
Clean, modular implementation with weighted approach and few-shot learning
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List

from classifier import WeightedInsuranceClassifier


class InsuranceClassificationInterface:
    """
    User interface for the insurance classification system
    Handles interactive labeling and batch processing
    """
    
    def __init__(self, taxonomy_path: str, data_path: str):
        """
        Initialize the classification interface
        
        Args:
            taxonomy_path: Path to insurance taxonomy CSV
            data_path: Path to company data CSV
        """
        self.taxonomy_path = taxonomy_path
        self.data_path = data_path
        
        # Initialize classifier with weighted approach
        print("🚀 Initializing Weighted Insurance Classifier...")
        self.classifier = WeightedInsuranceClassifier(taxonomy_path)
        
        # Load company data
        print("📊 Loading company data...")
        self.companies_df = pd.read_csv(data_path)
        print(f"✅ Loaded {len(self.companies_df)} companies")
        
        print(f"\n📋 Current Approach:")
        weights = self.classifier.get_current_weights()
        print(f"   • Tag Embedding: {weights['tag_embedding']:.0%} (semantic similarity of business tags)")
        print(f"   • Tag TF-IDF: {weights['tag_tfidf']:.0%} (word matches in business tags)")
        print(f"   • Description Embedding: {weights['desc_embedding']:.0%} (semantic similarity of description)")
        print(f"   • Description TF-IDF: {weights['desc_tfidf']:.0%} (word matches in description)")
        
    def interactive_labeling_session(self, start_idx: int = 0):
        """
        Interactive session for manual labeling with weighted AI suggestions
        
        Args:
            start_idx: Index to start from (for continuing sessions)
        """
        print(f"\n🏷️  Starting Interactive Labeling Session")
        print(f"Total companies: {len(self.companies_df)}")
        print(f"Starting from index: {start_idx}")
        print("\n📚 WEIGHTED APPROACH METHODOLOGY:")
        print("   Your strategy: Business tags matter much more than descriptions")
        print("   • Heavy emphasis on tag similarity (70% of total score)")
        print("   • Moderate emphasis on description similarity (30% of total score)")
        print("   • System learns from your manual validations (few-shot learning)")
        print("\nCommands:")
        print("  [number] - Accept suggested label")
        print("  's' - Skip this company")
        print("  'q' - Quit session")
        print("  'custom' - Enter custom label(s)")
        print("  'explain' - Show detailed scoring breakdown")
        print("  'weights' - View/adjust similarity weights")
        print("-" * 80)
        
        labeled_data = []
        
        for idx in range(start_idx, len(self.companies_df)):
            company = self.companies_df.iloc[idx]
            company_dict = company.to_dict()
            
            # Display full company information
            self.classifier.display_company_info(company, idx, len(self.companies_df))
            
            # Get weighted AI suggestions
            suggestions = self.classifier.get_suggestions(company_dict, top_k=10)
            
            # Display suggestions with weighted scoring
            self.classifier.display_weighted_suggestions(suggestions)
            
            # Get user input
            user_input = input(f"\n🎯 Your choice: ").strip().lower()
            
            if user_input == 'q':
                print("💾 Saving session and exiting...")
                break
            elif user_input == 's':
                print("⏭️  Skipping company...")
                continue
            elif user_input == 'explain':
                self._show_detailed_explanation(company_dict)
                continue
            elif user_input == 'weights':
                self._weight_adjustment_interface()
                continue
            elif user_input == 'custom':
                self._handle_custom_labels(company_dict, idx, labeled_data)
            elif user_input.isdigit():
                self._handle_suggestion_selection(company_dict, suggestions, user_input, idx, labeled_data)
            else:
                print("❌ Invalid input. Try again.")
                continue
        
        # Save labeled data
        self._save_labeled_data(labeled_data, start_idx)
        self._show_session_summary()
        
    def _show_detailed_explanation(self, company_dict: Dict):
        """Show detailed breakdown of how suggestions were calculated"""
        print(f"\n🔍 DETAILED SCORING EXPLANATION")
        print(f"{'='*60}")
        
        explanation = self.classifier.explain_suggestion(company_dict, top_k=3)
        
        # Company features
        features = explanation['company_features']
        print(f"\n1. COMPANY FEATURES ANALYZED:")
        print(f"   Description: {features['description']}")
        print(f"   Business Tags: {features['business_tags']}")
        print(f"   Tags Combined: {features['tags_text']}")
        
        # Methodology
        method = explanation['methodology']
        print(f"\n2. SCORING METHODOLOGY:")
        print(f"   Approach: {method['approach']}")
        print(f"   Weights: {method['weights']}")
        for component in method['components']:
            print(f"   • {component}")
        
        # Top matches breakdown
        print(f"\n3. TOP 3 MATCHES BREAKDOWN:")
        for i, match in enumerate(explanation['top_matches'], 1):
            print(f"\n   {i}. {match['label']} (Total: {match['total_score']:.3f})")
            breakdown = match['breakdown']
            print(f"      • Tag Embedding: {breakdown['tag_embedding']:.3f}")
            print(f"      • Tag TF-IDF: {breakdown['tag_tfidf']:.3f}")
            print(f"      • Desc Embedding: {breakdown['desc_embedding']:.3f}")
            print(f"      • Desc TF-IDF: {breakdown['desc_tfidf']:.3f}")
            print(f"      • Interpretation: {match['interpretation']}")
        
        input("\nPress Enter to continue...")
        
    def _weight_adjustment_interface(self):
        """Interface for adjusting similarity weights"""
        print(f"\n⚖️  SIMILARITY WEIGHTS ADJUSTMENT")
        print(f"{'='*50}")
        
        current_weights = self.classifier.get_current_weights()
        print(f"\nCurrent weights:")
        for component, weight in current_weights.items():
            print(f"   {component}: {weight:.1%}")
        
        print(f"\nOptions:")
        print(f"   1. Increase tag emphasis (Tag: 80%, Desc: 20%)")
        print(f"   2. Balanced approach (Tag: 60%, Desc: 40%)")
        print(f"   3. Reset to default (Tag: 70%, Desc: 30%)")
        print(f"   4. Custom weights")
        print(f"   5. Keep current weights")
        
        choice = input(f"\nSelect option (1-5): ").strip()
        
        if choice == '1':
            new_weights = {
                'tag_embedding': 0.5, 'tag_tfidf': 0.3,
                'desc_embedding': 0.15, 'desc_tfidf': 0.05
            }
        elif choice == '2':
            new_weights = {
                'tag_embedding': 0.35, 'tag_tfidf': 0.25,
                'desc_embedding': 0.25, 'desc_tfidf': 0.15
            }
        elif choice == '3':
            new_weights = {
                'tag_embedding': 0.4, 'tag_tfidf': 0.3,
                'desc_embedding': 0.2, 'desc_tfidf': 0.1
            }
        elif choice == '4':
            # Custom weights input
            try:
                print("Enter weights (must sum to 1.0):")
                tag_emb = float(input("Tag Embedding (0.0-1.0): "))
                tag_tfidf = float(input("Tag TF-IDF (0.0-1.0): "))
                desc_emb = float(input("Description Embedding (0.0-1.0): "))
                desc_tfidf = float(input("Description TF-IDF (0.0-1.0): "))
                
                new_weights = {
                    'tag_embedding': tag_emb, 'tag_tfidf': tag_tfidf,
                    'desc_embedding': desc_emb, 'desc_tfidf': desc_tfidf
                }
            except ValueError:
                print("❌ Invalid input. Keeping current weights.")
                return
        else:
            print("✅ Keeping current weights.")
            return
        
        try:
            self.classifier.adjust_weights(new_weights)
            print("✅ Weights updated successfully!")
        except ValueError as e:
            print(f"❌ Error: {e}")
            
    def _handle_custom_labels(self, company_dict: Dict, idx: int, labeled_data: List):
        """Handle custom label input"""
        custom_labels = input("Enter label(s) separated by comma: ").strip()
        labels = [l.strip() for l in custom_labels.split(',')]
        
        # Validate labels exist in taxonomy
        valid_labels = [l for l in labels if l in self.classifier.labels]
        invalid_labels = [l for l in labels if l not in self.classifier.labels]
        
        if invalid_labels:
            print(f"❌ Invalid labels: {', '.join(invalid_labels)}")
            print("Available labels can be found in the taxonomy file")
        
        if valid_labels:
            labeled_data.append({
                'company_index': idx,
                'labels': valid_labels,
                'confidence': 1.0,  # Manual selection
                'method': 'custom'
            })
            
            # Record as positive examples for few-shot learning
            for label in valid_labels:
                self.classifier.record_validation(company_dict, label, True)
                
            print(f"✅ Labeled with: {', '.join(valid_labels)}")
            print(f"🧠 System learned from your decision!")
        else:
            print("❌ No valid labels provided. Skipping...")
            
    def _handle_suggestion_selection(self, company_dict: Dict, suggestions: List, user_input: str, idx: int, labeled_data: List):
        """Handle selection of AI suggestion"""
        choice = int(user_input) - 1
        if 0 <= choice < len(suggestions):
            selected_label, confidence = suggestions[choice]
            
            # Confirm selection
            print(f"\n🔍 Selected: {selected_label}")
            print(f"   Weighted Score: {confidence:.3f}")
            confirm = input(f"   Confirm this label? (y/n): ").strip().lower()
            
            if confirm == 'y':
                labeled_data.append({
                    'company_index': idx,
                    'labels': [selected_label],
                    'confidence': confidence,
                    'method': 'ai_suggestion'
                })
                
                # Record as positive example for few-shot learning
                self.classifier.record_validation(company_dict, selected_label, True)
                
                # Record other high-confidence suggestions as negative examples
                for label, score in suggestions[:3]:
                    if label != selected_label and score > 0.6:
                        self.classifier.record_validation(company_dict, label, False)
                        
                print(f"✅ Labeled with: {selected_label}")
                print(f"🧠 System learned from your decision!")
                
                # Show learning effect
                total_validations = self.classifier.get_validation_stats()['total_validations']
                print(f"📈 Total validations: {total_validations}")
            else:
                print("❌ Selection cancelled.")
        else:
            print("❌ Invalid choice number.")
            
    def _save_labeled_data(self, labeled_data: List, start_idx: int):
        """Save labeled data to file"""
        if labeled_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f'data/labeled_companies_weighted_{timestamp}.json')
            output_path.parent.mkdir(exist_ok=True)
            
            session_data = {
                'session_info': {
                    'timestamp': timestamp,
                    'total_labeled': len(labeled_data),
                    'start_index': start_idx,
                    'approach': 'weighted_similarity_with_tag_emphasis',
                    'weights_used': self.classifier.get_current_weights()
                },
                'labeled_companies': labeled_data
            }
            
            import json
            with open(output_path, 'w') as f:
                json.dump(session_data, f, indent=2)
                
            print(f"\n💾 Saved {len(labeled_data)} labeled companies to {output_path}")
        
    def _show_session_summary(self):
        """Show summary of labeling session"""
        stats = self.classifier.get_validation_stats()
        
        print(f"\n📊 SESSION SUMMARY:")
        print(f"  Total validations: {stats['total_validations']}")
        print(f"  Labels with positive examples: {stats['labels_with_positive_examples']}")
        print(f"  Labels with negative examples: {stats['labels_with_negative_examples']}")
        print(f"  Approach: {stats['approach']}")
        
        if stats['top_validated_labels']:
            print(f"\n🏆 Most validated labels:")
            for item in stats['top_validated_labels'][:5]:
                print(f"    {item['label']}: {item['examples']} examples")
        
    def test_company(self, company_idx: int):
        """Test classification on a specific company"""
        if 0 <= company_idx < len(self.companies_df):
            company = self.companies_df.iloc[company_idx]
            company_dict = company.to_dict()
            
            # Display company info
            self.classifier.display_company_info(company, company_idx, len(self.companies_df))
            
            # Get and display suggestions
            suggestions = self.classifier.get_suggestions(company_dict, top_k=10)
            self.classifier.display_weighted_suggestions(suggestions, show_details=True)
            
            # Offer detailed explanation
            show_explanation = input(f"\nShow detailed explanation? (y/n): ").strip().lower()
            if show_explanation == 'y':
                self._show_detailed_explanation(company_dict)
        else:
            print(f"❌ Invalid company index. Must be between 0 and {len(self.companies_df)-1}")


def main():
    """Main entry point for the insurance classification system"""
    
    # File paths
    taxonomy_path = 'data/input/insurance_taxonomy - insurance_taxonomy.csv'
    data_path = 'data/input/ml_insurance_challenge.csv'
    
    # Initialize interface
    interface = InsuranceClassificationInterface(taxonomy_path, data_path)
    
    while True:
        print("\n🏢 Weighted Insurance Company Classification System")
        print("1. Start interactive labeling session")
        print("2. Test specific company")
        print("3. View validation statistics")
        print("4. Adjust similarity weights")
        print("5. Export labeled dataset")
        print("6. Exit")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == '1':
            start_idx = input("Start from index (default 0): ").strip()
            start_idx = int(start_idx) if start_idx else 0
            interface.interactive_labeling_session(start_idx)
            
        elif choice == '2':
            idx = input("Enter company index to test: ").strip()
            if idx.isdigit():
                interface.test_company(int(idx))
                
        elif choice == '3':
            stats = interface.classifier.get_validation_stats()
            print(f"\n📊 Validation Statistics:")
            print(f"  Total validations: {stats['total_validations']}")
            print(f"  Labels with positive examples: {stats['labels_with_positive_examples']}")
            print(f"  Labels with negative examples: {stats['labels_with_negative_examples']}")
            print(f"  Approach: {stats['approach']}")
            
            if stats['top_validated_labels']:
                print(f"\nTop validated labels:")
                for item in stats['top_validated_labels']:
                    print(f"  {item['label']}: {item['examples']} examples")
                    
        elif choice == '4':
            interface._weight_adjustment_interface()
            
        elif choice == '5':
            print("\n📤 Export functionality would be implemented here")
            print("   (Combine all labeled_companies_weighted_*.json files)")
            
        elif choice == '6':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid option!")


if __name__ == '__main__':
    main() 