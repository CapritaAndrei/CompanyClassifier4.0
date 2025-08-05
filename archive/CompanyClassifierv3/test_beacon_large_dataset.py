import sys
sys.path.append('./BEACON')
import beacon
import pandas as pd
import numpy as np
import time
from collections import Counter, defaultdict
import json

def load_2022_naics_data(file_path='data/input/2022_NAICS_Index_File.xlsx', sample_size=None):
    """Load and prepare the 2022 NAICS data for BEACON testing"""
    print(f"Loading 2022 NAICS data from {file_path}...")
    
    df = pd.read_excel(file_path)
    print(f"Total records loaded: {len(df)}")
    
    # Clean the data
    df = df.dropna()
    df['NAICS22'] = df['NAICS22'].astype(str)
    df['INDEX ITEM DESCRIPTION'] = df['INDEX ITEM DESCRIPTION'].astype(str)
    
    # Remove any invalid NAICS codes (should be 6 digits)
    df = df[df['NAICS22'].str.len() == 6]
    df = df[df['NAICS22'].str.isdigit()]
    
    print(f"Records after cleaning: {len(df)}")
    
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} records for testing")
    
    return df

def create_beacon_format_file(df, output_file='large_test_beacon_data.txt'):
    """Convert DataFrame to BEACON's pipe-delimited format"""
    print(f"Creating BEACON format file: {output_file}")
    
    with open(output_file, 'w') as f:
        f.write("TEXT|NAICS\n")
        for _, row in df.iterrows():
            text = str(row['INDEX ITEM DESCRIPTION']).replace('|', ' ')  # Remove pipes
            naics = str(row['NAICS22'])
            f.write(f"{text}|{naics}\n")
    
    print(f"Created BEACON format file with {len(df)} records")
    return output_file

def analyze_sector_distribution(naics_codes):
    """Analyze the distribution of NAICS sectors"""
    sectors = [naics[:2] for naics in naics_codes]
    sector_counts = Counter(sectors)
    
    sector_names = {
        '11': 'Agriculture, Forestry, Fishing',
        '21': 'Mining, Quarrying, Oil & Gas',
        '22': 'Utilities',
        '23': 'Construction',
        '31': 'Manufacturing (31-33)',
        '32': 'Manufacturing (31-33)',
        '33': 'Manufacturing (31-33)',
        '42': 'Wholesale Trade',
        '44': 'Retail Trade (44-45)',
        '45': 'Retail Trade (44-45)',
        '48': 'Transportation & Warehousing (48-49)',
        '49': 'Transportation & Warehousing (48-49)',
        '51': 'Information',
        '52': 'Finance and Insurance',
        '53': 'Real Estate',
        '54': 'Professional Services',
        '55': 'Management of Companies',
        '56': 'Administrative Services',
        '61': 'Educational Services',
        '62': 'Health Care',
        '71': 'Arts, Entertainment',
        '72': 'Accommodation & Food',
        '81': 'Other Services',
        '92': 'Public Administration'
    }
    
    return [(sector, sector_names.get(sector, f'Sector {sector}'), count) 
            for sector, count in sector_counts.most_common()]

def evaluate_predictions(actual_naics, predicted_naics, business_descriptions):
    """Comprehensive evaluation of BEACON predictions"""
    print("\n" + "="*80)
    print("DETAILED EVALUATION RESULTS")
    print("="*80)
    
    # Basic accuracy metrics
    exact_matches = sum(1 for a, p in zip(actual_naics, predicted_naics) if a == p)
    total = len(actual_naics)
    exact_accuracy = exact_matches / total * 100
    
    print(f"Exact Match Accuracy: {exact_matches}/{total} ({exact_accuracy:.2f}%)")
    
    # Sector-level accuracy (first 2 digits)
    actual_sectors = [naics[:2] for naics in actual_naics]
    predicted_sectors = [pred[:2] if pred else 'XX' for pred in predicted_naics]
    sector_matches = sum(1 for a, p in zip(actual_sectors, predicted_sectors) if a == p)
    sector_accuracy = sector_matches / total * 100
    
    print(f"Sector-level Accuracy: {sector_matches}/{total} ({sector_accuracy:.2f}%)")
    
    # 3-digit subsector accuracy
    actual_subsectors = [naics[:3] for naics in actual_naics]
    predicted_subsectors = [pred[:3] if pred else 'XXX' for pred in predicted_naics]
    subsector_matches = sum(1 for a, p in zip(actual_subsectors, predicted_subsectors) if a == p)
    subsector_accuracy = subsector_matches / total * 100
    
    print(f"Subsector-level Accuracy: {subsector_matches}/{total} ({subsector_accuracy:.2f}%)")
    
    # No prediction rate
    no_predictions = sum(1 for pred in predicted_naics if not pred)
    no_pred_rate = no_predictions / total * 100
    
    print(f"No Prediction Rate: {no_predictions}/{total} ({no_pred_rate:.2f}%)")
    
    # Sector-wise accuracy analysis
    print(f"\n{'Sector':<8} {'Name':<30} {'Total':<8} {'Correct':<8} {'Accuracy':<10}")
    print("-" * 80)
    
    sector_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    for actual, predicted in zip(actual_naics, predicted_naics):
        sector = actual[:2]
        sector_stats[sector]['total'] += 1
        if actual == predicted:
            sector_stats[sector]['correct'] += 1
    
    # Get sector names
    sector_names = {
        '11': 'Agriculture, Forestry, Fishing',
        '21': 'Mining, Quarrying, Oil & Gas',
        '22': 'Utilities',
        '23': 'Construction',
        '31': 'Manufacturing',
        '32': 'Manufacturing',
        '33': 'Manufacturing',
        '42': 'Wholesale Trade',
        '44': 'Retail Trade',
        '45': 'Retail Trade',
        '48': 'Transportation & Warehousing',
        '49': 'Transportation & Warehousing',
        '51': 'Information',
        '52': 'Finance and Insurance',
        '53': 'Real Estate',
        '54': 'Professional Services',
        '55': 'Management of Companies',
        '56': 'Administrative Services',
        '61': 'Educational Services',
        '62': 'Health Care',
        '71': 'Arts, Entertainment',
        '72': 'Accommodation & Food',
        '81': 'Other Services',
        '92': 'Public Administration'
    }
    
    for sector in sorted(sector_stats.keys()):
        stats = sector_stats[sector]
        accuracy = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        name = sector_names.get(sector, f'Sector {sector}')[:29]
        print(f"{sector:<8} {name:<30} {stats['total']:<8} {stats['correct']:<8} {accuracy:.1f}%")
    
    # Show some examples of correct and incorrect predictions
    print(f"\n{'EXAMPLE PREDICTIONS':<80}")
    print("="*80)
    
    correct_examples = []
    incorrect_examples = []
    
    for i, (actual, predicted, desc) in enumerate(zip(actual_naics, predicted_naics, business_descriptions)):
        if actual == predicted and len(correct_examples) < 10:
            correct_examples.append((desc[:50], actual, predicted))
        elif actual != predicted and len(incorrect_examples) < 10:
            incorrect_examples.append((desc[:50], actual, predicted))
    
    print("\n✅ CORRECT PREDICTIONS:")
    print(f"{'Description':<52} {'Actual':<8} {'Predicted':<8}")
    print("-" * 70)
    for desc, actual, predicted in correct_examples:
        print(f"{desc:<52} {actual:<8} {predicted:<8}")
    
    print("\n❌ INCORRECT PREDICTIONS:")
    print(f"{'Description':<52} {'Actual':<8} {'Predicted':<8}")
    print("-" * 70)
    for desc, actual, predicted in incorrect_examples:
        pred_display = predicted if predicted else 'NO_PRED'
        print(f"{desc:<52} {actual:<8} {pred_display:<8}")
    
    return {
        'exact_accuracy': exact_accuracy,
        'sector_accuracy': sector_accuracy,
        'subsector_accuracy': subsector_accuracy,
        'no_prediction_rate': no_pred_rate,
        'total_samples': total,
        'sector_stats': dict(sector_stats)
    }

def main():
    print("🚀 BEACON Large-Scale Testing")
    print("="*60)
    
    # Configuration
    SAMPLE_SIZE = None  # Set to None for full dataset, or e.g., 5000 for testing
    
    start_time = time.time()
    
    # Load the data
    df = load_2022_naics_data(sample_size=SAMPLE_SIZE)
    
    # Analyze data distribution
    print(f"\nData Distribution Analysis:")
    print("-" * 40)
    sector_dist = analyze_sector_distribution(df['NAICS22'].tolist())
    
    print(f"{'Sector':<8} {'Name':<30} {'Count':<8}")
    print("-" * 50)
    for sector, name, count in sector_dist[:15]:  # Show top 15 sectors
        print(f"{sector:<8} {name[:29]:<30} {count:<8}")
    
    if len(sector_dist) > 15:
        print(f"... and {len(sector_dist) - 15} more sectors")
    
    # Create BEACON format file
    beacon_file = create_beacon_format_file(df)
    
    # Load and train BEACON model
    print(f"\n🔧 Training BEACON Model")
    print("-" * 40)
    
    model_start = time.time()
    X_train, y_train, sample_weight = beacon.load_naics_data(vintage="2017")
    print(f"Training data loaded: {len(X_train)} samples")
    
    model = beacon.BeaconModel(verbose=1)
    model.fit(X_train, y_train, sample_weight)
    
    model_time = time.time() - model_start
    print(f"Model training completed in {model_time:.1f} seconds")
    
    # Make predictions on our test data
    print(f"\n🎯 Making Predictions on {len(df)} Test Cases")
    print("-" * 50)
    
    prediction_start = time.time()
    business_descriptions = df['INDEX ITEM DESCRIPTION'].tolist()
    actual_naics = df['NAICS22'].tolist()
    
    # Process in batches for memory efficiency
    batch_size = 1000
    all_predictions = []
    
    for i in range(0, len(business_descriptions), batch_size):
        batch_end = min(i + batch_size, len(business_descriptions))
        batch_descriptions = business_descriptions[i:batch_end]
        
        batch_predictions = model.predict(batch_descriptions)
        all_predictions.extend(batch_predictions)
        
        print(f"Processed {batch_end}/{len(business_descriptions)} predictions...")
    
    prediction_time = time.time() - prediction_start
    print(f"Predictions completed in {prediction_time:.1f} seconds")
    print(f"Average time per prediction: {prediction_time/len(df)*1000:.2f} ms")
    
    # Evaluate results
    results = evaluate_predictions(actual_naics, all_predictions, business_descriptions)
    
    # Save detailed results
    output_data = {
        'test_info': {
            'dataset_size': len(df),
            'sample_size': SAMPLE_SIZE,
            'training_time_seconds': model_time,
            'prediction_time_seconds': prediction_time,
            'total_time_seconds': time.time() - start_time
        },
        'accuracy_metrics': results,
        'predictions': [
            {
                'description': desc,
                'actual_naics': actual,
                'predicted_naics': pred,
                'correct': actual == pred
            }
            for desc, actual, pred in zip(business_descriptions, actual_naics, all_predictions)
        ]
    }
    
    output_file = f"beacon_large_test_results_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n📊 FINAL SUMMARY")
    print("="*60)
    print(f"Dataset Size:           {len(df):,} records")
    print(f"Exact Match Accuracy:   {results['exact_accuracy']:.2f}%")
    print(f"Sector-level Accuracy:  {results['sector_accuracy']:.2f}%")
    print(f"No Prediction Rate:     {results['no_prediction_rate']:.2f}%")
    print(f"Total Processing Time:  {total_time:.1f} seconds")
    print(f"Throughput:            {len(df)/total_time:.1f} predictions/second")

if __name__ == "__main__":
    main() 