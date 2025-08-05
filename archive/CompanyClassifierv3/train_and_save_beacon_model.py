import sys
sys.path.append('./BEACON')
import beacon
import pandas as pd
import numpy as np
import pickle
import time
import os
from collections import Counter

def combine_training_data():
    """Combine 2017 and 2022 NAICS training data intelligently"""
    print("🔄 Loading Training Data")
    print("=" * 50)
    
    # Load both datasets
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
    print(f"  • Common NAICS codes: {len(common_codes)} (will have more examples)")
    print(f"  • Only in 2017: {len(only_2017)} (additional coverage)")
    print(f"  • Only in 2022: {len(only_2022)} (additional coverage)")
    
    return X_combined, y_combined, w_combined

def train_beacon_model(X, y, sample_weight, model_name="combined"):
    """Train BEACON model with timing"""
    print(f"\n🔧 Training BEACON Model ({model_name})")
    print("=" * 50)
    
    start_time = time.time()
    
    model = beacon.BeaconModel(verbose=1)
    model.fit(X, y, sample_weight)
    
    training_time = time.time() - start_time
    print(f"✓ Training completed in {training_time:.1f} seconds")
    
    # Display model summary
    model.summary()
    
    return model, training_time

def save_model(model, filename="beacon_model_combined_2017_2022.pkl"):
    """Save the trained BEACON model"""
    print(f"\n💾 Saving Model")
    print("=" * 50)
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    filepath = os.path.join("models", filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
    print(f"✓ Model saved to: {filepath}")
    print(f"✓ File size: {file_size:.1f} MB")
    
    return filepath

def load_model(filepath):
    """Load a saved BEACON model"""
    print(f"📂 Loading saved model from: {filepath}")
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    print("✓ Model loaded successfully")
    return model

def quick_test_model(model, test_descriptions=None):
    """Quick test of the model with sample predictions"""
    print(f"\n🧪 Quick Model Test")
    print("=" * 50)
    
    if test_descriptions is None:
        test_descriptions = [
            "software development company",
            "restaurant and food service",
            "construction contractor",
            "financial planning services",
            "medical practice",
            "manufacturing electronics",
            "retail clothing store",
            "auto repair shop"
        ]
    
    predictions = model.predict(test_descriptions)
    
    print(f"{'Business Description':<35} {'Predicted NAICS':<15}")
    print("-" * 52)
    
    for desc, pred in zip(test_descriptions, predictions):
        pred_display = pred if pred else "NO_MATCH"
        print(f"{desc:<35} {pred_display:<15}")

def compare_model_performance():
    """Compare performance of single vs combined dataset models"""
    print(f"\n📊 Performance Comparison")
    print("=" * 50)
    
    # Quick test dataset
    test_cases = [
        ("Soybean farming", "111110"),
        ("Software development", "541511"),
        ("Restaurant", "722513"),
        ("Construction", "236220"),
        ("Financial services", "523930"),
        ("Medical practice", "621111"),
        ("Auto repair", "811111"),
        ("Retail store", "448140")
    ]
    
    test_descriptions = [desc for desc, _ in test_cases]
    expected_naics = [naics for _, naics in test_cases]
    
    # Test 2017-only model
    print("\nTesting 2017-only model...")
    X_2017, y_2017, w_2017 = beacon.load_naics_data(vintage='2017')
    model_2017 = beacon.BeaconModel(verbose=0)
    model_2017.fit(X_2017, y_2017, w_2017)
    
    pred_2017 = model_2017.predict(test_descriptions)
    accuracy_2017 = sum(1 for p, e in zip(pred_2017, expected_naics) if p == e) / len(expected_naics) * 100
    
    # Test combined model
    print("Testing combined model...")
    X_combined, y_combined, w_combined = combine_training_data()
    model_combined = beacon.BeaconModel(verbose=0)
    model_combined.fit(X_combined, y_combined, w_combined)
    
    pred_combined = model_combined.predict(test_descriptions)
    accuracy_combined = sum(1 for p, e in zip(pred_combined, expected_naics) if p == e) / len(expected_naics) * 100
    
    print(f"\nResults on {len(test_cases)} test cases:")
    print(f"  • 2017-only model: {accuracy_2017:.1f}% accuracy")
    print(f"  • Combined model:  {accuracy_combined:.1f}% accuracy")
    
    print(f"\n{'Test Case':<25} {'Expected':<10} {'2017 Model':<12} {'Combined':<12}")
    print("-" * 65)
    
    for desc, expected, p17, pcomb in zip(test_descriptions, expected_naics, pred_2017, pred_combined):
        p17_display = p17 if p17 else "NO_MATCH"
        pcomb_display = pcomb if pcomb else "NO_MATCH"
        print(f"{desc[:24]:<25} {expected:<10} {p17_display:<12} {pcomb_display:<12}")

def main():
    print("🚀 BEACON Model Training & Saving Pipeline")
    print("=" * 60)
    
    # Step 1: Combine training data
    X_combined, y_combined, w_combined = combine_training_data()
    
    # Step 2: Train the model
    model, training_time = train_beacon_model(X_combined, y_combined, w_combined)
    
    # Step 3: Save the model
    model_path = save_model(model)
    
    # Step 4: Test loading the model
    print(f"\n🔄 Testing Model Loading")
    print("=" * 50)
    loaded_model = load_model(model_path)
    
    # Step 5: Quick test
    quick_test_model(loaded_model)
    
    # Step 6: Performance comparison (optional, time-consuming)
    compare_performance = input("\nRun performance comparison? (y/n): ").lower().strip()
    if compare_performance == 'y':
        compare_model_performance()
    
    # Final summary
    print(f"\n✅ SUMMARY")
    print("=" * 60)
    print(f"✓ Combined dataset: {len(X_combined):,} samples")
    print(f"✓ Training time: {training_time:.1f} seconds")
    print(f"✓ Model saved to: {model_path}")
    print(f"✓ Model tested and working")
    print(f"\n🎯 Usage:")
    print(f"  from train_and_save_beacon_model import load_model")
    print(f"  model = load_model('{model_path}')")
    print(f"  predictions = model.predict(['your business description'])")

if __name__ == "__main__":
    main() 