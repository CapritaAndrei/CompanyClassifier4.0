import sys
sys.path.append('./BEACON')
import beacon
import pandas as pd
import numpy as np
import time
from collections import Counter, defaultdict

def load_test_data(file_path='data/input/2022_NAICS_Index_File.xlsx'):
    """Load the 2022 NAICS test data"""
    print(f"Loading test data from {file_path}...")
    
    df = pd.read_excel(file_path)
    df = df.dropna()
    df['NAICS22'] = df['NAICS22'].astype(str)
    df['INDEX ITEM DESCRIPTION'] = df['INDEX ITEM DESCRIPTION'].astype(str)
    
    # Clean data - keep only valid 6-digit NAICS codes
    df = df[df['NAICS22'].str.len() == 6]
    df = df[df['NAICS22'].str.isdigit()]
    
    print(f"Test data loaded: {len(df):,} samples")
    return df

def train_single_model():
    """Train BEACON on 2017 data only"""
    print("🔧 Training Single Dataset Model (2017 only)")
    print("-" * 50)
    
    start_time = time.time()
    X_train, y_train, w_train = beacon.load_naics_data(vintage="2017")
    
    model = beacon.BeaconModel(verbose=0)
    model.fit(X_train, y_train, w_train)
    
    training_time = time.time() - start_time
    print(f"✓ Single model trained in {training_time:.1f}s ({len(X_train):,} samples, {len(set(y_train))} NAICS codes)")
    
    return model

def train_combined_model():
    """Train BEACON on combined 2017+2022 data"""
    print("🔧 Training Combined Dataset Model (2017 + 2022)")
    print("-" * 50)
    
    start_time = time.time()
    
    # Load both datasets
    X_2017, y_2017, w_2017 = beacon.load_naics_data(vintage='2017')
    X_2022, y_2022, w_2022 = beacon.load_naics_data(vintage='2022')
    
    # Combine datasets
    X_combined = np.concatenate([X_2017, X_2022])
    y_combined = np.concatenate([y_2017, y_2022])
    w_combined = np.concatenate([w_2017, w_2022])
    
    model = beacon.BeaconModel(verbose=0)
    model.fit(X_combined, y_combined, w_combined)
    
    training_time = time.time() - start_time
    print(f"✓ Combined model trained in {training_time:.1f}s ({len(X_combined):,} samples, {len(set(y_combined))} NAICS codes)")
    
    return model

def evaluate_model(model, test_descriptions, actual_naics, model_name):
    """Evaluate model performance with detailed metrics"""
    print(f"\n📊 Evaluating {model_name} Model")
    print("-" * 50)
    
    # Make predictions
    start_time = time.time()
    predictions = model.predict(test_descriptions)
    prediction_time = time.time() - start_time
    
    # Calculate accuracy metrics
    total = len(actual_naics)
    exact_matches = sum(1 for a, p in zip(actual_naics, predictions) if a == p)
    exact_accuracy = exact_matches / total * 100
    
    # Sector-level accuracy (first 2 digits)
    actual_sectors = [naics[:2] for naics in actual_naics]
    predicted_sectors = [pred[:2] if pred else 'XX' for pred in predictions]
    sector_matches = sum(1 for a, p in zip(actual_sectors, predicted_sectors) if a == p)
    sector_accuracy = sector_matches / total * 100
    
    # Subsector accuracy (first 3 digits)
    actual_subsectors = [naics[:3] for naics in actual_naics]
    predicted_subsectors = [pred[:3] if pred else 'XXX' for pred in predictions]
    subsector_matches = sum(1 for a, p in zip(actual_subsectors, predicted_subsectors) if a == p)
    subsector_accuracy = subsector_matches / total * 100
    
    # No prediction rate
    no_predictions = sum(1 for pred in predictions if not pred)
    no_pred_rate = no_predictions / total * 100
    
    # Results
    results = {
        'model_name': model_name,
        'exact_accuracy': exact_accuracy,
        'sector_accuracy': sector_accuracy,
        'subsector_accuracy': subsector_accuracy,
        'no_prediction_rate': no_pred_rate,
        'prediction_time': prediction_time,
        'total_samples': total
    }
    
    print(f"Exact Match Accuracy:    {exact_accuracy:.2f}% ({exact_matches:,}/{total:,})")
    print(f"Sector-level Accuracy:   {sector_accuracy:.2f}% ({sector_matches:,}/{total:,})")
    print(f"Subsector-level Accuracy: {subsector_accuracy:.2f}% ({subsector_matches:,}/{total:,})")
    print(f"No Prediction Rate:      {no_pred_rate:.2f}% ({no_predictions:,}/{total:,})")
    print(f"Prediction Time:         {prediction_time:.1f}s ({prediction_time/total*1000:.2f}ms per prediction)")
    
    return results, predictions

def compare_results(single_results, combined_results):
    """Compare the results between single and combined models"""
    print(f"\n🏆 MODEL COMPARISON RESULTS")
    print("=" * 60)
    
    metrics = ['exact_accuracy', 'sector_accuracy', 'subsector_accuracy', 'no_prediction_rate']
    
    print(f"{'Metric':<25} {'Single (2017)':<15} {'Combined':<15} {'Improvement':<15}")
    print("-" * 70)
    
    for metric in metrics:
        single_val = single_results[metric]
        combined_val = combined_results[metric]
        
        if metric == 'no_prediction_rate':
            improvement = single_val - combined_val  # Lower is better for no predictions
            improvement_text = f"{improvement:+.2f}pp"
        else:
            improvement = combined_val - single_val  # Higher is better for accuracy
            improvement_text = f"{improvement:+.2f}pp"
        
        metric_display = metric.replace('_', ' ').title()
        print(f"{metric_display:<25} {single_val:.2f}%{'':<8} {combined_val:.2f}%{'':<8} {improvement_text:<15}")
    
    # Overall assessment
    print(f"\n📈 IMPROVEMENT SUMMARY:")
    exact_improvement = combined_results['exact_accuracy'] - single_results['exact_accuracy']
    sector_improvement = combined_results['sector_accuracy'] - single_results['sector_accuracy']
    
    if exact_improvement > 0:
        print(f"✅ Combined model is BETTER: +{exact_improvement:.2f}pp exact accuracy")
    elif exact_improvement < -0.1:
        print(f"❌ Combined model is WORSE: {exact_improvement:.2f}pp exact accuracy")
    else:
        print(f"➡️  Models perform SIMILARLY: {exact_improvement:+.2f}pp difference")
    
    if sector_improvement > 0:
        print(f"✅ Combined model has better sector accuracy: +{sector_improvement:.2f}pp")
    elif sector_improvement < -0.1:
        print(f"❌ Combined model has worse sector accuracy: {sector_improvement:.2f}pp")
    
    # Time comparison
    single_time = single_results['prediction_time']
    combined_time = combined_results['prediction_time']
    time_diff = combined_time - single_time
    print(f"⏱️  Prediction time difference: {time_diff:+.1f}s ({time_diff/single_time*100:+.1f}%)")

def main():
    print("🧪 BEACON Model Accuracy Comparison")
    print("=" * 60)
    print("Testing Single Dataset (2017) vs Combined Dataset (2017+2022)")
    print("=" * 60)
    
    # Load test data
    test_df = load_test_data()
    test_descriptions = test_df['INDEX ITEM DESCRIPTION'].tolist()
    actual_naics = test_df['NAICS22'].tolist()
    
    print(f"\n📋 Test Setup:")
    print(f"  • Test samples: {len(test_descriptions):,}")
    print(f"  • Unique NAICS codes in test: {len(set(actual_naics))}")
    
    # Train both models
    print(f"\n🔄 Training Models...")
    single_model = train_single_model()
    combined_model = train_combined_model()
    
    # Evaluate both models
    single_results, single_predictions = evaluate_model(
        single_model, test_descriptions, actual_naics, "Single Dataset"
    )
    
    combined_results, combined_predictions = evaluate_model(
        combined_model, test_descriptions, actual_naics, "Combined Dataset"
    )
    
    # Compare results
    compare_results(single_results, combined_results)
    
    # Show some example differences
    print(f"\n🔍 EXAMPLE PREDICTION DIFFERENCES:")
    print("=" * 80)
    print(f"{'Description':<40} {'Actual':<8} {'Single':<8} {'Combined':<8} {'Status':<12}")
    print("-" * 80)
    
    differences_shown = 0
    for i, (desc, actual, single_pred, combined_pred) in enumerate(zip(
        test_descriptions, actual_naics, single_predictions, combined_predictions
    )):
        if single_pred != combined_pred and differences_shown < 10:
            status = ""
            if single_pred == actual and combined_pred != actual:
                status = "Single Better"
            elif single_pred != actual and combined_pred == actual:
                status = "Combined Better"  
            elif single_pred != actual and combined_pred != actual:
                status = "Both Wrong"
            else:
                status = "Both Right"
            
            single_display = single_pred if single_pred else "NO_PRED"
            combined_display = combined_pred if combined_pred else "NO_PRED"
            
            print(f"{desc[:39]:<40} {actual:<8} {single_display:<8} {combined_display:<8} {status:<12}")
            differences_shown += 1
    
    print(f"\n✅ CONCLUSION:")
    print("=" * 60)
    exact_improvement = combined_results['exact_accuracy'] - single_results['exact_accuracy']
    
    if exact_improvement > 1.0:
        print("🎉 SIGNIFICANT IMPROVEMENT: Combined model is substantially better!")
        print("   ✅ Recommendation: Use combined 2017+2022 training data")
    elif exact_improvement > 0.1:
        print("✅ MODEST IMPROVEMENT: Combined model is slightly better")
        print("   ✅ Recommendation: Use combined 2017+2022 training data")
    elif exact_improvement > -0.1:
        print("➡️  SIMILAR PERFORMANCE: No significant difference")
        print("   💭 Recommendation: Either model works, combined has more coverage")
    else:
        print("❌ WORSE PERFORMANCE: Single model is better")
        print("   ⚠️  Recommendation: Investigate why combined model performs worse")

if __name__ == "__main__":
    main() 