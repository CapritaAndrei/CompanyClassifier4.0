"""
Train KNN Multi-Modal Classifier on 9K Real Dataset
Uses BEACON predictions + NAICS mappings to create training labels
"""

import pandas as pd
import numpy as np
import json
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Import our classifiers
sys.path.append('BEACON')
from beacon import BeaconModel, load_naics_data
from knn_multi_modal_classifier import KNNMultiModalClassifier

def load_naics_mappings():
    """Load the NAICS to insurance label mappings"""
    print("📋 Loading NAICS mappings...")
    
    # Load from our existing mapping files
    try:
        with open('data/processed/exact_text_matches.json', 'r') as f:
            exact_matches = json.load(f)
        
        with open('data/processed/filtered_matches_summary.json', 'r') as f:
            filtered_matches = json.load(f)
        
        # Combine mappings
        naics_to_insurance = {}
        
        # Add exact matches
        for label, matches in exact_matches.items():
            for match in matches:
                naics_code = match['naics_code']
                if naics_code not in naics_to_insurance:
                    naics_to_insurance[naics_code] = []
                naics_to_insurance[naics_code].append(label)
        
        # Add filtered matches
        for label, matches in filtered_matches.items():
            for match in matches:
                naics_code = match['naics_code']
                if naics_code not in naics_to_insurance:
                    naics_to_insurance[naics_code] = []
                if label not in naics_to_insurance[naics_code]:
                    naics_to_insurance[naics_code].append(label)
        
        print(f"✅ Loaded mappings for {len(naics_to_insurance)} NAICS codes")
        return naics_to_insurance
        
    except Exception as e:
        print(f"⚠️  Could not load existing mappings: {e}")
        print("Creating basic mappings from NAICS sectors...")
        
        # Fallback: Create basic sector-based mappings
        return create_sector_based_mappings()

def create_sector_based_mappings():
    """Create basic insurance label mappings based on NAICS sectors"""
    sector_mappings = {
        # Construction
        '23': ['Pipeline Construction Services', 'Excavation Services', 'Cable Installation Services'],
        # Manufacturing  
        '31': ['Chemical Manufacturing', 'Plastic Manufacturing', 'Food Processing Services'],
        '32': ['Chemical Manufacturing', 'Plastic Manufacturing', 'Printing Services'],
        '33': ['Chemical Manufacturing', 'Plastic Manufacturing', 'Rubber Manufacturing'],
        # Professional Services
        '54': ['Consulting Services', 'Software Manufacturing', 'Engineering Services'],
        # Healthcare
        '62': ['Veterinary Services', 'Medical Services'],
        # Other Services
        '81': ['Restoration Services', 'Welding Services'],
        # Agriculture
        '11': ['Agricultural Equipment Services', 'Landscaping Services'],
        # Transportation
        '48': ['Logistics Services', 'Travel Services'],
        '49': ['Logistics Services', 'Travel Services'],
        # Retail
        '44': ['Retail Services'],
        '45': ['Retail Services'],
        # Accommodation & Food
        '72': ['Catering Services', 'Restaurant Services'],
    }
    
    naics_to_insurance = {}
    for sector, labels in sector_mappings.items():
        # Create mappings for all codes in this sector
        for i in range(1000):
            naics_code = f"{sector}{i:04d}"
            naics_to_insurance[naics_code] = labels
    
    return naics_to_insurance

def create_training_labels(df, beacon_model, naics_mappings):
    """Use BEACON to predict NAICS and map to insurance labels"""
    print("🔮 Creating training labels using BEACON predictions...")
    
    # Get BEACON predictions
    descriptions = df['description'].fillna('').astype(str)
    naics_predictions = beacon_model.predict(descriptions)
    
    training_labels = []
    label_counts = {}
    
    for naics_code in naics_predictions:
        # Get insurance labels for this NAICS code
        insurance_labels = naics_mappings.get(naics_code, [])
        
        if insurance_labels:
            # Use the first/most common insurance label
            label = insurance_labels[0]
            training_labels.append(label)
            label_counts[label] = label_counts.get(label, 0) + 1
        else:
            # Fallback: use sector-based label
            sector = naics_code[:2] if len(naics_code) >= 2 else '00'
            sector_label = get_sector_fallback_label(sector)
            training_labels.append(sector_label)
            label_counts[sector_label] = label_counts.get(sector_label, 0) + 1
    
    print(f"📊 Created {len(training_labels)} training labels")
    print(f"🏷️  Unique labels: {len(set(training_labels))}")
    print("📈 Top 10 label distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   {label}: {count}")
    
    return training_labels

def get_sector_fallback_label(sector):
    """Get fallback insurance label based on NAICS sector"""
    fallback_map = {
        '23': 'Construction Services',
        '31': 'Manufacturing Services', 
        '32': 'Manufacturing Services',
        '33': 'Manufacturing Services',
        '54': 'Professional Services',
        '62': 'Healthcare Services',
        '81': 'Repair Services',
        '11': 'Agricultural Services',
        '48': 'Transportation Services',
        '49': 'Transportation Services',
        '44': 'Retail Services',
        '45': 'Retail Services',
        '72': 'Food Services',
    }
    return fallback_map.get(sector, 'Other Services')

def train_and_evaluate_knn(X_train, X_test, y_train, y_test):
    """Train and evaluate the KNN classifier"""
    print("🚀 Training KNN Multi-Modal Classifier...")
    
    # Initialize classifier with optimized parameters
    knn_classifier = KNNMultiModalClassifier(
        k_neighbors=7,  # Slightly higher K for larger dataset
        text_weight=0.4,
        categorical_weight=0.3,
        tags_weight=0.3,
        distance_weighting=True,
        verbose=1
    )
    
    # Train the model
    knn_classifier.fit(X_train, y_train)
    
    print("\n🎯 Making predictions on test set...")
    
    # Make predictions
    y_pred = knn_classifier.predict(X_test)
    y_pred_proba = knn_classifier.predict_proba(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Show classification report
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return knn_classifier, y_pred, y_pred_proba

def analyze_sample_predictions(knn_classifier, X_test, y_test, y_pred, n_samples=5):
    """Analyze some sample predictions in detail"""
    print(f"\n🔍 Analyzing {n_samples} sample predictions:")
    print("=" * 80)
    
    for i in range(min(n_samples, len(X_test))):
        print(f"\n📋 SAMPLE {i+1}:")
        print(f"Description: {X_test.iloc[i]['description'][:100]}...")
        print(f"Sector: {X_test.iloc[i]['sector']}, Category: {X_test.iloc[i]['category']}")
        print(f"Business Tags: {X_test.iloc[i]['business_tags'][:100]}...")
        print(f"True Label: {y_test.iloc[i]}")
        print(f"Predicted: {y_pred[i]}")
        print(f"Match: {'✅' if y_test.iloc[i] == y_pred[i] else '❌'}")
        
        # Get nearest neighbors
        neighbors = knn_classifier.get_nearest_neighbors(X_test.iloc[i:i+1], n_neighbors=3)
        print(f"Nearest Neighbors:")
        for j, (neighbor_idx, similarity) in enumerate(neighbors[0]):
            neighbor_label = knn_classifier.y_train_[neighbor_idx]
            print(f"  {j+1}. Similarity: {similarity:.3f} → {neighbor_label}")

def main():
    """Main training pipeline"""
    print("🎯 Training KNN Multi-Modal Classifier on 9K Dataset")
    print("=" * 60)
    
    # Load the 9K dataset
    print("📁 Loading 9K challenge dataset...")
    df = pd.read_csv('data/input/ml_insurance_challenge.csv')
    print(f"✅ Loaded {len(df)} companies")
    
    # Load and train BEACON model
    print("\n🔧 Loading and training BEACON model...")
    X_2017, y_2017, sw_2017 = load_naics_data('2017')
    X_2022, y_2022, sw_2022 = load_naics_data('2022')
    
    # Combine datasets for BEACON training
    X_beacon = np.concatenate([X_2017, X_2022])
    y_beacon = np.concatenate([y_2017, y_2022])
    sw_beacon = np.concatenate([sw_2017, sw_2022])
    
    beacon_model = BeaconModel(verbose=0)
    beacon_model.fit(X_beacon, y_beacon, sw_beacon)
    print("✅ BEACON model trained")
    
    # Load NAICS mappings
    naics_mappings = load_naics_mappings()
    
    # Create training labels using BEACON + mappings
    training_labels = create_training_labels(df, beacon_model, naics_mappings)
    df['insurance_label'] = training_labels
    
    # Remove samples with no valid labels or missing data
    df_clean = df.dropna(subset=['description', 'sector', 'category', 'niche', 'business_tags'])
    print(f"📊 Clean dataset: {len(df_clean)} samples")
    
    # Split into train/test
    print("\n🔀 Splitting into train/test sets...")
    X = df_clean[['description', 'sector', 'category', 'niche', 'business_tags']]
    y = df_clean['insurance_label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Training set: {len(X_train)} samples")
    print(f"📊 Test set: {len(X_test)} samples")
    
    # Train and evaluate KNN classifier
    knn_classifier, y_pred, y_pred_proba = train_and_evaluate_knn(X_train, X_test, y_train, y_test)
    
    # Analyze sample predictions
    analyze_sample_predictions(knn_classifier, X_test, y_test, y_pred)
    
    # Save the trained model
    print("\n💾 Saving trained model...")
    import pickle
    with open('models/knn_multi_modal_trained.pkl', 'wb') as f:
        pickle.dump(knn_classifier, f)
    print("✅ Model saved to models/knn_multi_modal_trained.pkl")
    
    return knn_classifier

if __name__ == "__main__":
    main() 