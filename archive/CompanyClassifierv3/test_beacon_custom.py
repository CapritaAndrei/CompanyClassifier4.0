import sys
sys.path.append('./BEACON')
import beacon
import pandas as pd

def main():
    print("Testing BEACON with Custom Data")
    print("=" * 50)
    
    # Load and train BEACON model on 2017 data
    print("Loading BEACON training data...")
    X_train, y_train, sample_weight = beacon.load_naics_data(vintage="2017")
    print(f"Training data loaded: {len(X_train)} samples")
    
    print("\nTraining BEACON model...")
    model = beacon.BeaconModel(verbose=1)
    model.fit(X_train, y_train, sample_weight)
    
    print("\nModel training completed!")
    model.summary()
    
    # Load our custom test data
    print("\n" + "="*50)
    print("Testing with Custom Business Descriptions")
    print("="*50)
    
    # Read our test data
    custom_data = pd.read_csv('test_beacon_data.txt', delimiter='|')
    
    # Extract business descriptions for prediction
    business_descriptions = custom_data['TEXT'].tolist()
    actual_naics = custom_data['NAICS'].tolist()
    
    # Make predictions
    predicted_naics = model.predict(business_descriptions)
    
    # Show results
    print(f"\n{'Business Description':<50} {'Actual NAICS':<12} {'Predicted NAICS':<15} {'Match':<8}")
    print("-" * 90)
    
    correct_predictions = 0
    for i, desc in enumerate(business_descriptions):
        actual = str(actual_naics[i])
        predicted = predicted_naics[i] if predicted_naics[i] else "NO MATCH"
        match = "✓" if actual == predicted else "✗"
        if actual == predicted:
            correct_predictions += 1
            
        print(f"{desc[:49]:<50} {actual:<12} {predicted:<15} {match:<8}")
    
    accuracy = correct_predictions / len(business_descriptions) * 100
    print(f"\nAccuracy: {correct_predictions}/{len(business_descriptions)} ({accuracy:.1f}%)")
    
    # Test with some new business descriptions
    print("\n" + "="*50)
    print("Testing with New Business Descriptions")
    print("="*50)
    
    new_descriptions = [
        "insurance company",
        "financial services",
        "construction company", 
        "software company",
        "restaurant",
        "medical clinic",
        "auto shop"
    ]
    
    new_predictions = model.predict(new_descriptions)
    
    print(f"\n{'Business Description':<30} {'Predicted NAICS':<15}")
    print("-" * 45)
    
    for desc, pred in zip(new_descriptions, new_predictions):
        pred_display = pred if pred else "NO MATCH"
        print(f"{desc:<30} {pred_display:<15}")

if __name__ == "__main__":
    main() 