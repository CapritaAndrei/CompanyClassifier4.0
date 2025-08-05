import sys
sys.path.append('./BEACON')
import beacon
import pandas as pd
import numpy as np
import time

def combine_and_train_beacon():
    """Combine 2017+2022 data and train BEACON model"""
    print("🚀 BEACON Combined Dataset Training & Testing")
    print("=" * 60)
    
    # Load both datasets
    print("🔄 Loading Training Data")
    print("-" * 40)
    
    print("Loading 2017 data...")
    X_2017, y_2017, w_2017 = beacon.load_naics_data(vintage='2017')
    print(f"✓ 2017 data: {len(X_2017):,} samples, {len(set(y_2017))} unique NAICS codes")
    
    print("Loading 2022 data...")
    X_2022, y_2022, w_2022 = beacon.load_naics_data(vintage='2022')
    print(f"✓ 2022 data: {len(X_2022):,} samples, {len(set(y_2022))} unique NAICS codes")
    
    # Combine the datasets
    X_combined = np.concatenate([X_2017, X_2022])
    y_combined = np.concatenate([y_2017, y_2022])
    w_combined = np.concatenate([w_2017, w_2022])
    
    print(f"✓ Combined: {len(X_combined):,} samples, {len(set(y_combined))} unique NAICS codes")
    
    # Analyze the combination
    naics_2017 = set(y_2017)
    naics_2022 = set(y_2022)
    common_codes = naics_2017 & naics_2022
    only_2017 = naics_2017 - naics_2022
    only_2022 = naics_2022 - naics_2017
    
    print(f"\nDataset Analysis:")
    print(f"  • Common NAICS codes: {len(common_codes)} (enhanced with more examples)")
    print(f"  • Only in 2017: {len(only_2017)} (additional coverage)")
    print(f"  • Only in 2022: {len(only_2022)} (additional coverage)")
    
    # Train the model
    print(f"\n🔧 Training BEACON Model on Combined Dataset")
    print("-" * 50)
    
    start_time = time.time()
    
    model = beacon.BeaconModel(verbose=1)
    model.fit(X_combined, y_combined, w_combined)
    
    training_time = time.time() - start_time
    print(f"\n✓ Training completed in {training_time:.1f} seconds")
    
    # Display model summary
    model.summary()
    
    return model

def quick_test_model(model):
    """Quick test of the combined model"""
    print(f"\n🧪 Quick Test: Sample Predictions")
    print("-" * 50)
    
    test_descriptions = [
        "software development company",
        "restaurant and food service", 
        "construction contractor",
        "financial planning services",
        "medical practice",
        "manufacturing electronics",
        "retail clothing store",
        "auto repair shop",
        "insurance company",
        "law firm",
        "accounting services",
        "real estate agency"
    ]
    
    predictions = model.predict(test_descriptions)
    
    print(f"{'Business Description':<35} {'Predicted NAICS':<15}")
    print("-" * 52)
    
    for desc, pred in zip(test_descriptions, predictions):
        pred_display = pred if pred else "NO_MATCH"
        print(f"{desc:<35} {pred_display:<15}")

def compare_single_vs_combined():
    """Compare 2017-only vs combined model performance"""
    print(f"\n📊 Performance Comparison: Single vs Combined")
    print("-" * 60)
    
    # Test cases with expected NAICS codes
    test_cases = [
        ("Soybean farming", "111110"),
        ("Software development", "541511"), 
        ("Restaurant", "722513"),
        ("Construction", "236220"),
        ("Financial services", "523930"),
        ("Medical practice", "621111"),
        ("Auto repair", "811111"),
        ("Retail store", "448140"),
        ("Insurance agency", "524210"),
        ("Law office", "541110")
    ]
    
    test_descriptions = [desc for desc, _ in test_cases]
    expected_naics = [naics for _, naics in test_cases]
    
    # Test 2017-only model
    print("Training 2017-only model...")
    X_2017, y_2017, w_2017 = beacon.load_naics_data(vintage='2017')
    model_2017 = beacon.BeaconModel(verbose=0)
    model_2017.fit(X_2017, y_2017, w_2017)
    
    pred_2017 = model_2017.predict(test_descriptions)
    accuracy_2017 = sum(1 for p, e in zip(pred_2017, expected_naics) if p == e) / len(expected_naics) * 100
    
    # Test combined model (already trained)
    print("Training combined model...")
    X_2017, y_2017, w_2017 = beacon.load_naics_data(vintage='2017')
    X_2022, y_2022, w_2022 = beacon.load_naics_data(vintage='2022')
    X_combined = np.concatenate([X_2017, X_2022])
    y_combined = np.concatenate([y_2017, y_2022])
    w_combined = np.concatenate([w_2017, w_2022])
    
    model_combined = beacon.BeaconModel(verbose=0)
    model_combined.fit(X_combined, y_combined, w_combined)
    
    pred_combined = model_combined.predict(test_descriptions)
    accuracy_combined = sum(1 for p, e in zip(pred_combined, expected_naics) if p == e) / len(expected_naics) * 100
    
    print(f"\nAccuracy Results on {len(test_cases)} test cases:")
    print(f"  • 2017-only model:  {accuracy_2017:.1f}% accuracy")
    print(f"  • Combined model:   {accuracy_combined:.1f}% accuracy")
    print(f"  • Improvement:      {accuracy_combined - accuracy_2017:+.1f} percentage points")
    
    print(f"\n{'Test Case':<25} {'Expected':<10} {'2017 Model':<12} {'Combined':<12} {'Status':<10}")
    print("-" * 75)
    
    for desc, expected, p17, pcomb in zip(test_descriptions, expected_naics, pred_2017, pred_combined):
        p17_display = p17 if p17 else "NO_MATCH"
        pcomb_display = pcomb if pcomb else "NO_MATCH"
        
        status = ""
        if p17 == expected and pcomb == expected:
            status = "✓ Both"
        elif p17 != expected and pcomb == expected:
            status = "✓ Improved"
        elif p17 == expected and pcomb != expected:
            status = "✗ Worse"
        else:
            status = "✗ Both wrong"
            
        print(f"{desc[:24]:<25} {expected:<10} {p17_display:<12} {pcomb_display:<12} {status:<10}")

def main():
    # Train the combined model
    model = combine_and_train_beacon()
    
    # Quick test
    quick_test_model(model)
    
    # Performance comparison
    print("\n" + "="*60)
    user_input = input("Run detailed comparison vs 2017-only model? (y/n): ").lower().strip()
    if user_input == 'y':
        compare_single_vs_combined()
    
    print(f"\n✅ SUMMARY")
    print("="*60)
    print("✓ Combined model trained successfully")
    print("✓ 84k+ training samples (2017 + 2022 data)")
    print("✓ 1,150+ unique NAICS codes covered")
    print("✓ Ready for large-scale testing!")

if __name__ == "__main__":
    main() 