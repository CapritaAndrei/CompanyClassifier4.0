import sys
sys.path.append('./BEACON')
import beacon
import pandas as pd
import numpy as np
import time

def train_combined_beacon_model():
    """Train BEACON on combined 2017+2022 data"""
    print("🔧 Training BEACON on Combined Dataset (2017 + 2022)")
    print("=" * 60)
    
    start_time = time.time()
    
    # Load both datasets
    print("Loading 2017 data...")
    X_2017, y_2017, w_2017 = beacon.load_naics_data(vintage='2017')
    print(f"✓ 2017: {len(X_2017):,} samples, {len(set(y_2017))} NAICS codes")
    
    print("Loading 2022 data...")
    X_2022, y_2022, w_2022 = beacon.load_naics_data(vintage='2022')
    print(f"✓ 2022: {len(X_2022):,} samples, {len(set(y_2022))} NAICS codes")
    
    # Combine datasets
    X_combined = np.concatenate([X_2017, X_2022])
    y_combined = np.concatenate([y_2017, y_2022])
    w_combined = np.concatenate([w_2017, w_2022])
    
    print(f"✓ Combined: {len(X_combined):,} samples, {len(set(y_combined))} NAICS codes")
    
    # Train the model
    print("\nTraining BEACON model...")
    model = beacon.BeaconModel(verbose=1)
    model.fit(X_combined, y_combined, w_combined)
    
    training_time = time.time() - start_time
    print(f"\n✓ Training completed in {training_time:.1f} seconds")
    
    return model

def load_insurance_challenge_data(file_path='data/input/ml_insurance_challenge.csv', num_rows=5):
    """Load the first few rows from the insurance challenge CSV"""
    print(f"\n📋 Loading Insurance Challenge Data")
    print("=" * 60)
    
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Loaded insurance challenge data: {len(df):,} total rows")
        
        # Take first num_rows
        test_df = df.head(num_rows).copy()
        print(f"✓ Using first {num_rows} rows for testing")
        
        # Show the data we're testing on
        print(f"\n📄 Test Data Preview:")
        print("-" * 60)
        
        for i, row in test_df.iterrows():
            business_desc = row.get('description', row.get('business_description', 'N/A'))
            company_name = row.get('company_name', 'N/A')
            
            print(f"\nRow {i+1}:")
            print(f"  Company: {company_name}")
            print(f"  Description: {business_desc}")
            
            # Show any other relevant columns  
            for col in df.columns:
                if col not in ['description', 'business_description', 'company_name'] and not pd.isna(row[col]):
                    print(f"  {col}: {row[col]}")
        
        return test_df
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find file {file_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def analyze_predictions(model, test_df):
    """Analyze BEACON predictions on insurance challenge data"""
    print(f"\n🎯 BEACON Predictions on Insurance Data")
    print("=" * 60)
    
    # Extract business descriptions - try different column names
    description_col = None
    if 'description' in test_df.columns:
        description_col = 'description'
    elif 'business_description' in test_df.columns:
        description_col = 'business_description'
    else:
        print("❌ No description column found. Available columns:", list(test_df.columns))
        return
    
    descriptions = test_df[description_col].fillna('').astype(str).tolist()
    print(f"✓ Using column '{description_col}' for business descriptions")
    
    # Make predictions
    print("Making predictions...")
    start_time = time.time()
    predictions = model.predict(descriptions)
    prediction_time = time.time() - start_time
    
    print(f"✓ Predictions completed in {prediction_time:.2f} seconds")
    
    # Show detailed results
    print(f"\n📊 Detailed Results:")
    print("=" * 80)
    
    for i, (desc, pred) in enumerate(zip(descriptions, predictions)):
        company_name = test_df.iloc[i].get('company_name', 'Unknown')
        
        print(f"\n🏢 Company {i+1}: {company_name}")
        print(f"📝 Description: {desc[:100]}{'...' if len(desc) > 100 else ''}")
        print(f"🎯 BEACON Prediction: {pred if pred else 'NO MATCH'}")
        
        if pred:
            # Try to get additional info about the NAICS code
            sector = pred[:2] if len(pred) >= 2 else 'Unknown'
            subsector = pred[:3] if len(pred) >= 3 else 'Unknown'
            
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
            
            sector_name = sector_names.get(sector, f'Sector {sector}')
            print(f"📂 Sector: {sector} - {sector_name}")
            print(f"📁 Subsector: {subsector}")
        
        # Show text cleaning
        cleaned_text = model.clean_text(desc)
        print(f"🧹 Cleaned Text: {cleaned_text[:80]}{'...' if len(cleaned_text) > 80 else ''}")
        
        print("-" * 80)
    
    # Get top predictions for first description
    if len(descriptions) > 0 and descriptions[0]:
        print(f"\n🔍 Top 10 Predictions for First Company:")
        print("-" * 50)
        
        top_predictions = model.predict_top10([descriptions[0]])
        if top_predictions is not None and len(top_predictions) > 0 and len(top_predictions[0]) > 0:
            for rank, naics_code in enumerate(top_predictions[0], 1):
                print(f"{rank:2d}. {naics_code}")
        
        # Get probability scores
        print(f"\n📈 Probability Scores for First Company:")
        print("-" * 50)
        
        probs = model.predict_proba([descriptions[0]])
        if probs is not None and len(probs) > 0 and len(probs[0]) > 0:
            # Sort by probability and show top scores
            sorted_probs = sorted(probs[0].items(), key=lambda x: x[1], reverse=True)
            for naics_code, prob in sorted_probs[:10]:
                print(f"{naics_code}: {prob:.4f}")

def main():
    print("🧪 Testing BEACON on Insurance Challenge Data")
    print("=" * 60)
    
    # Load insurance challenge data
    test_df = load_insurance_challenge_data()
    if test_df is None:
        return
    
    # Train the combined BEACON model
    model = train_combined_beacon_model()
    
    # Analyze predictions
    analyze_predictions(model, test_df)
    
    print(f"\n✅ ANALYSIS COMPLETE")
    print("=" * 60)
    print("✓ BEACON trained on combined 2017+2022 data")
    print("✓ Predictions made on real insurance business descriptions")
    print("✓ Results show how BEACON handles verbose, real-world text")
    print("\n💡 Key Insights:")
    print("  • BEACON's text cleaning simplifies verbose descriptions")
    print("  • Model provides confidence scores and alternative predictions")
    print("  • Performance on real-world data vs structured NAICS data")

if __name__ == "__main__":
    main() 