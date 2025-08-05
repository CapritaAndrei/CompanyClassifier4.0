# 🚀 How to Use Veridion3 Insurance Classification System

A practical, step-by-step guide to using your production-ready insurance classification system.

## 📋 Quick Overview

Your system can do 2 main things:
1. **Train new models** using your 2,378 company examples
2. **Make predictions** on new companies using the trained model

## 🎯 Method 1: Using the Simple Command Line Interface

### 🤖 Train a New Model

```bash
python3 main.py train
```

**What this does:**
- Loads your training data (2,378 companies)
- Extracts features (text, tags, categories)
- Trains 4 different ML models
- Shows performance comparison
- Saves the best model automatically

**Expected output:**
```
🚀 Starting model training...
📊 Loading training data...
✅ Loaded 2378 training examples
🔧 FEATURE ENGINEERING:
✅ TF-IDF features: (2378, 5000)
✅ Business tags features: (2378, 2000)
...
🏆 Best performing model: Logistic Regression
✅ Training completed!
```

### 🔮 Make Predictions on New Companies

```bash
python3 main.py predict input_file.csv
```

**What this does:**
- Loads the production model
- Processes your new companies
- Returns insurance classifications

## 🎯 Method 2: Using Python Code Directly

### Step 1: Train a Custom Model

```python
import sys
sys.path.append('src')

from training.pipeline import InsuranceClassificationTrainingPipeline

# Initialize training pipeline
pipeline = InsuranceClassificationTrainingPipeline(
    training_data_path="data/processed/training_data.csv"
)

# Run complete training
results = pipeline.run_full_pipeline()

# See results
print(results)
```

### Step 2: Make Predictions on Individual Companies

```python
import sys
sys.path.append('src')

from inference.predictor import InsuranceModelPredictor

# Load the production model
predictor = InsuranceModelPredictor("models/production_model.pkl")

# Example company data
company = {
    'description': 'Software development company specializing in web applications and mobile apps for healthcare providers',
    'business_tags': ['Software Development', 'Healthcare Technology', 'Web Development'],
    'sector': 'Services',
    'category': 'Software Development', 
    'niche': 'Healthcare Software'
}

# Get predictions
prediction = predictor.predict(company, top_k=5)

print(f"Primary prediction: {prediction['primary_prediction']}")
print(f"Confidence: {prediction['primary_confidence']:.3f}")
print("\nTop 5 predictions:")
for i, pred in enumerate(prediction['top_predictions'][:5], 1):
    print(f"{i}. {pred['label']} ({pred['confidence']:.3f})")
```

## 📊 Understanding Your Data

### What You Have
- **Training examples**: 2,378 companies
- **Insurance labels**: 220+ unique classifications
- **Features**: Company descriptions, business tags, sector/category info
- **Best model**: Logistic Regression (70.5% accuracy, 78.3% top-3 accuracy)

### Data Format Expected
Your input data should have these columns:
```csv
description,business_tags,sector,category,niche
"Software company...",['Software Development'],Services,Technology,Software
```

## 🔧 Practical Examples

### Example 1: Quick Single Prediction

```python
# test_single_prediction.py
import sys
sys.path.append('src')
from inference.predictor import InsuranceModelPredictor

predictor = InsuranceModelPredictor("models/production_model.pkl")

# Test company
company = {
    'description': 'Restaurant serving Italian cuisine in downtown area',
    'business_tags': ['Restaurant', 'Food Service', 'Italian Cuisine'],
    'sector': 'Services',
    'category': 'Food Service',
    'niche': 'Restaurant'
}

result = predictor.predict(company)
print(f"This company is classified as: {result['primary_prediction']}")
```

### Example 2: Batch Processing Multiple Companies

```python
# test_batch_predictions.py
import sys
sys.path.append('src')
from inference.predictor import InsuranceModelPredictor
import pandas as pd

predictor = InsuranceModelPredictor("models/production_model.pkl")

# Load companies from CSV
companies_df = pd.read_csv('new_companies.csv')

results = []
for idx, company in companies_df.iterrows():
    prediction = predictor.predict(company.to_dict())
    results.append({
        'company_index': idx,
        'company_name': company.get('name', f'Company_{idx}'),
        'prediction': prediction['primary_prediction'],
        'confidence': prediction['primary_confidence']
    })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('predictions.csv', index=False)
print(f"Processed {len(results)} companies, saved to predictions.csv")
```

### Example 3: Evaluate Model Performance

```python
# test_model_performance.py
import sys
sys.path.append('src')
from inference.predictor import InsuranceModelPredictor

predictor = InsuranceModelPredictor("models/production_model.pkl")

# Get model information
info = predictor.get_model_info()
print(f"Model: {info['model_name']}")
print(f"Trained on: {info['training_examples']} examples")
print(f"Features: {info['features_shape']}")
print(f"Labels: {info['unique_labels']}")
```

## 🛠️ Setting Up Your Environment

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
python3 -c "
import pandas as pd
import numpy as np
import sklearn
print('✅ All dependencies installed successfully')
"
```

## 📈 Interpreting Results

### Accuracy Metrics Explained

- **Primary Accuracy (70.5%)**: Exact label matches
- **Top-3 Accuracy (78.3%)**: True label in top 3 predictions  
- **Top-5 Accuracy (78.7%)**: True label in top 5 predictions

### Confidence Scores
- **0.8-1.0**: High confidence (trust the prediction)
- **0.6-0.8**: Medium confidence (consider top 3 predictions)
- **0.0-0.6**: Low confidence (review manually)

## 🚨 Troubleshooting Common Issues

### Issue 1: Import Errors
```bash
# If you get import errors, make sure you're in the right directory
cd /path/to/Veridion3
python3 -c "import sys; sys.path.append('src'); from training.pipeline import InsuranceClassificationTrainingPipeline; print('✅ Imports work')"
```

### Issue 2: Missing Model File
```bash
# Check if production model exists
ls -la models/
# Should show: production_model.pkl and model_info.json
```

### Issue 3: Data Format Issues
```python
# Check your data format
import pandas as pd
df = pd.read_csv('your_data.csv')
print(df.columns)  # Should include: description, business_tags, sector, category
print(df.head())   # Check first few rows
```

## 🎯 Common Use Cases

### Use Case 1: Classify New Insurance Prospects
```python
# For insurance sales - classify potential clients
prospects = [
    {'description': 'Manufacturing plant producing automotive parts', 'sector': 'Manufacturing'},
    {'description': 'Software consulting firm', 'sector': 'Services'},
    {'description': 'Local grocery store chain', 'sector': 'Retail'}
]

for prospect in prospects:
    prediction = predictor.predict(prospect)
    print(f"Prospect: {prospect['description'][:50]}...")
    print(f"Insurance type: {prediction['primary_prediction']}")
    print("---")
```

### Use Case 2: Update Training Data
```python
# Add new labeled examples and retrain
from training.pipeline import InsuranceClassificationTrainingPipeline

# Load pipeline with updated data
pipeline = InsuranceClassificationTrainingPipeline(
    training_data_path="data/processed/updated_training_data.csv"
)

# Retrain with new data
results = pipeline.run_full_pipeline()
```

### Use Case 3: API Integration (Future)
```python
# Framework for future API development
from flask import Flask, request, jsonify
import sys
sys.path.append('src')
from inference.predictor import InsuranceModelPredictor

app = Flask(__name__)
predictor = InsuranceModelPredictor("models/production_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    company_data = request.json
    prediction = predictor.predict(company_data)
    return jsonify(prediction)

# Run with: python api_server.py
```

## 🚀 Next Steps

1. **Test with your data**: Try the examples above with your specific data
2. **Experiment with confidence thresholds**: Find the right balance for your use case
3. **Monitor performance**: Track how well predictions work in practice
4. **Collect feedback**: Use real-world results to improve the model
5. **Scale up**: Consider the API framework for production deployment

## 📞 Quick Reference Commands

```bash
# Train new model
python3 main.py train

# Make predictions
python3 main.py predict input.csv

# Test single prediction (interactive)
python3 -c "
import sys; sys.path.append('src')
from inference.predictor import InsuranceModelPredictor
predictor = InsuranceModelPredictor('models/production_model.pkl')
print('✅ System ready for predictions')
"
```

Your system is production-ready! Start with the simple examples above and gradually work up to more complex use cases. 🎉 