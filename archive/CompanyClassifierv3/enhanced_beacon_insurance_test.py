import sys
sys.path.append('./BEACON')
import beacon
import pandas as pd
import numpy as np
import time
import json

def load_naics_descriptions():
    """Load both official NAICS titles and specific activity descriptions"""
    print("Loading NAICS descriptions...")
    
    # Load official NAICS titles (cleaner, broader)
    print("  Loading official NAICS titles...")
    codes_df = pd.read_excel('data/input/6-digit_2022_Codes.xlsx')
    
    naics_titles = {}
    for _, row in codes_df.iterrows():
        if pd.notna(row.iloc[0]) and pd.notna(row.iloc[1]):  # Skip NaN rows
            naics_code = str(int(row.iloc[0]))  # Convert to string, remove .0
            title = str(row.iloc[1]).strip()
            naics_titles[naics_code] = title
    
    # Load specific activity descriptions (more detailed examples)
    print("  Loading specific activity descriptions...")
    index_df = pd.read_excel('data/input/2022_NAICS_Index_File.xlsx')
    
    naics_activities = {}
    for _, row in index_df.iterrows():
        naics_code = str(row['NAICS22'])
        description = str(row['INDEX ITEM DESCRIPTION'])
        if naics_code not in naics_activities:
            naics_activities[naics_code] = []
        naics_activities[naics_code].append(description)
    
    print(f"✓ Loaded official titles for {len(naics_titles)} NAICS codes")
    print(f"✓ Loaded activity descriptions for {len(naics_activities)} NAICS codes")
    
    return naics_titles, naics_activities

def load_insurance_mappings():
    """Load the insurance label mappings"""
    print("Loading insurance label mappings...")
    
    try:
        with open('data/processed/exact_text_matches.json', 'r') as f:
            mappings = json.load(f)
        
        # Create reverse mapping: NAICS code -> insurance labels
        naics_to_insurance = {}
        for insurance_label, data in mappings.items():
            if 'hierarchy_breakdown' in data and '6-digit' in data['hierarchy_breakdown']:
                for match in data['hierarchy_breakdown']['6-digit']:
                    naics_code = match['naics_code']
                    if naics_code not in naics_to_insurance:
                        naics_to_insurance[naics_code] = []
                    if insurance_label not in naics_to_insurance[naics_code]:
                        naics_to_insurance[naics_code].append(insurance_label)
        
        print(f"✓ Loaded mappings for {len(naics_to_insurance)} NAICS codes to insurance labels")
        return naics_to_insurance
        
    except FileNotFoundError:
        print("⚠️  Insurance mapping file not found - predictions will show without mappings")
        return {}

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

def load_insurance_challenge_data(file_path='data/input/ml_insurance_challenge.csv', num_rows=10):
    """Load the first few rows from the insurance challenge CSV"""
    print(f"\n📋 Loading Insurance Challenge Data")
    print("=" * 60)
    
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Loaded insurance challenge data: {len(df):,} total rows")
        
        # Take first num_rows
        test_df = df.head(num_rows).copy()
        print(f"✓ Using first {num_rows} rows for testing")
        
        return test_df
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find file {file_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def get_sector_name(naics_code):
    """Get sector name from NAICS code"""
    if not naics_code or len(naics_code) < 2:
        return "Unknown"
    
    sector = naics_code[:2]
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
    return sector_names.get(sector, f'Sector {sector}')

def analyze_predictions_enhanced(model, test_df, naics_titles, naics_activities, insurance_mappings):
    """Enhanced analysis with detailed predictions for all companies"""
    print(f"\n🎯 ENHANCED BEACON Predictions Analysis")
    print("=" * 80)
    
    # Extract business descriptions
    description_col = 'description'
    if description_col not in test_df.columns:
        description_col = 'business_description'
    
    descriptions = test_df[description_col].fillna('').astype(str).tolist()
    print(f"✓ Using column '{description_col}' for business descriptions")
    
    # Make predictions
    print("Making predictions...")
    start_time = time.time()
    predictions = model.predict(descriptions)
    prediction_time = time.time() - start_time
    print(f"✓ Predictions completed in {prediction_time:.2f} seconds")
    
    # Get top 10 and probabilities for all companies
    print("Getting detailed predictions...")
    all_top_predictions = model.predict_top10(descriptions)
    all_probabilities = model.predict_proba(descriptions)
    
    print(f"\n📊 DETAILED RESULTS FOR ALL COMPANIES:")
    print("=" * 100)
    
    for i, (desc, pred) in enumerate(zip(descriptions, predictions)):
        company_name = test_df.iloc[i].get('company_name', f'Company {i+1}')
        
        print(f"\n🏢 COMPANY {i+1}: {company_name}")
        print("="*80)
        print(f"📝 FULL BUSINESS DESCRIPTION:")
        print(f"    {desc}")
        print(f"🧹 CLEANED TEXT:")
        print(f"    {model.clean_text(desc)}")
        
        # Primary prediction
        print(f"\n🎯 PRIMARY PREDICTION: {pred if pred else 'NO MATCH'}")
        if pred:
            sector_name = get_sector_name(pred)
            title = naics_titles.get(pred, 'No title available')
            activities = naics_activities.get(pred, ['No activity description available'])
            insurance_labels = insurance_mappings.get(pred, [])
            
            print(f"📂 Sector: {pred[:2]} - {sector_name}")
            print(f"📄 Official Title: {title}")
            print(f"🏷️  Activity Descriptions:")
            for activity in activities:
                print(f"    {activity}")
            
            if insurance_labels:
                print(f"🏷️  INSURANCE LABELS: {', '.join(insurance_labels)}")
            else:
                print(f"🏷️  INSURANCE LABELS: No direct mapping found")
        
        # Top 10 predictions
        print(f"\n🔟 TOP 10 PREDICTIONS:")
        print("-" * 70)
        
        if all_top_predictions is not None and len(all_top_predictions) > i:
            top_10 = all_top_predictions[i]
            probs = all_probabilities[i] if all_probabilities is not None and len(all_probabilities) > i else {}
            
            for rank, naics_code in enumerate(top_10[:10], 1):
                prob = probs.get(naics_code, 0.0) if probs else 0.0
                title = naics_titles.get(naics_code, 'No title available')
                sector_name = get_sector_name(naics_code)
                insurance_labels = insurance_mappings.get(naics_code, [])
                insurance_mapping = ', '.join(insurance_labels) if insurance_labels else 'No mapping'
                
                print(f"{rank:2d}. {naics_code} ({prob:.4f}) - {sector_name}")
                print(f"    📄 Official Title: {title}")
                print(f"    🏷️  INSURANCE MAPPING: {insurance_mapping}")
                print()
        
        print("="*80)
    
    # Summary statistics
    print(f"\n📈 SUMMARY STATISTICS:")
    print("="*60)
    
    total_predictions = len([p for p in predictions if p])
    no_predictions = len(predictions) - total_predictions
    
    print(f"Total companies analyzed: {len(predictions)}")
    print(f"Successful predictions: {total_predictions}")
    print(f"No predictions: {no_predictions}")
    print(f"Success rate: {total_predictions/len(predictions)*100:.1f}%")
    
    # Sector distribution
    sectors = [get_sector_name(p) for p in predictions if p]
    sector_counts = {}
    for sector in sectors:
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
    
    print(f"\nSector Distribution:")
    for sector, count in sorted(sector_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {sector}: {count}")
    
    # Insurance mapping success rate
    mapped_predictions = len([p for p in predictions if p and insurance_mappings.get(p)])
    if total_predictions > 0:
        mapping_rate = mapped_predictions / total_predictions * 100
        print(f"\nInsurance mapping success: {mapped_predictions}/{total_predictions} ({mapping_rate:.1f}%)")

def main():
    print("🧪 Enhanced BEACON Testing on Insurance Challenge Data")
    print("=" * 70)
    
    # Load supporting data
    naics_titles, naics_activities = load_naics_descriptions()
    insurance_mappings = load_insurance_mappings()
    
    # Load insurance challenge data
    test_df = load_insurance_challenge_data()
    if test_df is None:
        return
    
    # Train the combined BEACON model
    model = train_combined_beacon_model()
    
    # Enhanced analysis
    analyze_predictions_enhanced(model, test_df, naics_titles, naics_activities, insurance_mappings)
    
    print(f"\n✅ ENHANCED ANALYSIS COMPLETE")
    print("=" * 70)
    print("✓ BEACON trained on combined 2017+2022 data")
    print("✓ Detailed predictions with NAICS descriptions")
    print("✓ Insurance label mappings included")
    print("✓ Top 10 predictions for each company")
    print("✓ Comprehensive analysis provided")

if __name__ == "__main__":
    main() 