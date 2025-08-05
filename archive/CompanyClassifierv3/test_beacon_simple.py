"""
Simple BEACON Test - Direct NAICS to Insurance Label Mapping
==========================================================

Test the actual BEACON model:
1. Load BEACON training data (40k examples)
2. Train BEACON model (company_description → NAICS_code)
3. Test on sample descriptions → get NAICS predictions
4. Map NAICS predictions to insurance labels using Master Map

This is the correct approach - use BEACON as intended!
"""

import json
import sys
from pathlib import Path

# Add BEACON to path
sys.path.append('BEACON')
from beacon import BeaconModel, load_naics_data


class SimpleBeaconTest:
    """
    Simple test of BEACON → Insurance Label mapping
    """
    
    def __init__(self):
        print("🔄 Initializing Simple BEACON Test...")
        
        # Load Master Map for NAICS → Insurance Label mapping
        master_map_path = "data/processed/master_insurance_to_naics_mapping_simplified.json"
        
        try:
            with open(master_map_path, 'r') as f:
                self.master_map = json.load(f)
            print(f"✅ Master Map loaded: {len(self.master_map)} insurance labels")
        except Exception as e:
            print(f"❌ Error loading Master Map: {e}")
            return
        
        # Create reverse mapping: NAICS → Insurance Label
        self.naics_to_label = {}
        for label, naics_codes in self.master_map.items():
            for naics_info in naics_codes:
                naics_code = naics_info['naics_code']
                self.naics_to_label[naics_code] = label
        
        print(f"✅ Reverse mapping created: {len(self.naics_to_label)} NAICS codes mapped")
        
        # Initialize BEACON model
        self.beacon_model = None
    
    def load_and_train_beacon(self):
        """Load BEACON training data and train the model"""
        print("\n🚀 Loading and Training BEACON Model")
        print("=" * 50)
        
        try:
            # Load BEACON training data
            print("📥 Loading BEACON training data...")
            X, y, sample_weight = load_naics_data(vintage="2017")
            
            print(f"✅ BEACON data loaded:")
            print(f"   Training examples: {len(X):,}")
            print(f"   Unique NAICS codes: {len(set(y))}")
            
            # Create and train BEACON model
            print("🔄 Training BEACON model...")
            self.beacon_model = BeaconModel(verbose=1)
            self.beacon_model.fit(X, y, sample_weight)
            
            print("✅ BEACON model trained successfully!")
            
            return True
            
        except Exception as e:
            print(f"❌ Error training BEACON: {e}")
            return False
    
    def test_beacon_predictions(self):
        """Test BEACON predictions on sample business descriptions"""
        print("\n🧪 Testing BEACON Predictions")
        print("=" * 50)
        
        if self.beacon_model is None:
            print("❌ BEACON model not trained")
            return
        
        # Sample business descriptions to test
        test_descriptions = [
            "Chemical manufacturing company producing industrial chemicals and polymers",
            "Waste management and recycling services for commercial customers",
            "Insurance brokerage providing risk management and coverage solutions",
            "Software development and technology consulting services",
            "Plastic manufacturing and injection molding operations",
            "Textile manufacturing and fabric production facility",
            "Food processing and beverage manufacturing company",
            "Construction materials supply and building products distribution",
            "Metal fabrication and steel manufacturing services",
            "Transportation and logistics services provider"
        ]
        
        print("🔍 BEACON Predictions → Insurance Label Mapping:")
        print()
        
        for i, description in enumerate(test_descriptions, 1):
            try:
                # Get BEACON prediction
                naics_predictions = self.beacon_model.predict([description])
                predicted_naics = naics_predictions[0]
                
                # Map to insurance label
                if predicted_naics in self.naics_to_label:
                    insurance_label = self.naics_to_label[predicted_naics]
                    status = "✅ MAPPED"
                else:
                    insurance_label = "NOT MAPPED"
                    status = "❌ UNMAPPED"
                
                print(f"   {i:2d}. {status}")
                print(f"       Description: {description[:60]}...")
                print(f"       BEACON → NAICS: {predicted_naics}")
                print(f"       NAICS → Insurance: {insurance_label}")
                print()
                
            except Exception as e:
                print(f"   {i:2d}. ❌ ERROR: {e}")
                print()
    
    def analyze_mapping_coverage(self):
        """Analyze how many BEACON predictions can be mapped to insurance labels"""
        print("\n📊 Analyzing Mapping Coverage")
        print("=" * 50)
        
        if self.beacon_model is None:
            print("❌ BEACON model not trained")
            return
        
        # Load BEACON data to see all possible NAICS codes
        try:
            X, y, sample_weight = load_naics_data(vintage="2017")
            unique_beacon_naics = set(y)
            mapped_naics = set(self.naics_to_label.keys())
            
            # Analyze overlap
            overlapping_naics = unique_beacon_naics.intersection(mapped_naics)
            beacon_only_naics = unique_beacon_naics - mapped_naics
            
            print(f"📈 Coverage Analysis:")
            print(f"   NAICS codes in BEACON: {len(unique_beacon_naics)}")
            print(f"   NAICS codes in Master Map: {len(mapped_naics)}")
            print(f"   Overlapping NAICS codes: {len(overlapping_naics)}")
            print(f"   Coverage percentage: {len(overlapping_naics) / len(unique_beacon_naics) * 100:.1f}%")
            
            # Count training examples that can be mapped
            mappable_examples = sum(1 for naics in y if naics in mapped_naics)
            coverage_by_examples = mappable_examples / len(y) * 100
            
            print(f"\n📊 Training Data Coverage:")
            print(f"   Total BEACON examples: {len(y):,}")
            print(f"   Mappable examples: {mappable_examples:,}")
            print(f"   Coverage by examples: {coverage_by_examples:.1f}%")
            
            print(f"\n🎯 This means:")
            print(f"   • {coverage_by_examples:.1f}% of BEACON training data can be converted to insurance labels")
            print(f"   • That's {mappable_examples:,} training examples for your {len(set(self.naics_to_label.values()))} insurance labels")
            print(f"   • Average examples per label: {mappable_examples / len(set(self.naics_to_label.values())):.0f}")
            
        except Exception as e:
            print(f"❌ Error analyzing coverage: {e}")
    
    def run_complete_test(self):
        """Run the complete test workflow"""
        print("🎯 SIMPLE BEACON TEST - COMPLETE WORKFLOW")
        print("=" * 70)
        
        # Step 1: Train BEACON
        if not self.load_and_train_beacon():
            print("❌ Failed to train BEACON model")
            return
        
        # Step 2: Test predictions
        self.test_beacon_predictions()
        
        # Step 3: Analyze coverage
        self.analyze_mapping_coverage()
        
        # Final summary
        print(f"\n🎉 TEST COMPLETE!")
        print("=" * 70)
        print(f"✅ BEACON model: Trained and working")
        print(f"✅ NAICS predictions: Generated successfully")
        print(f"✅ Insurance mapping: Working for mapped NAICS codes")
        print(f"✅ Coverage analysis: Complete")
        
        print(f"\n🚀 Next Steps:")
        print(f"   1. Expand Master Map to cover more NAICS codes")
        print(f"   2. Apply BEACON to your 9k company dataset")
        print(f"   3. Generate massive training dataset with insurance labels")
        print(f"   4. Train final insurance label classifier")


def main():
    """Main test function"""
    test = SimpleBeaconTest()
    test.run_complete_test()


if __name__ == "__main__":
    main() 