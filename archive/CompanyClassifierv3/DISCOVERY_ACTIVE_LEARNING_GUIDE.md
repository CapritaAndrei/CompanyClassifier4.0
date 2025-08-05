# 🎯 Discovery-Based Active Learning Pipeline Guide

## Overview
The Discovery-Based Active Learning Pipeline is an intelligent system that **starts with your 61 mapped labels** and **gradually discovers new labels** based on real data needs. You don't need to wait for complete coverage to start learning!

## 🚀 Key Benefits

### **Why Start with 61 Labels?**
- ✅ **28% coverage** is a solid foundation
- ✅ **Most important labels** are likely already mapped
- ✅ **Incremental expansion** is more efficient than upfront mapping
- ✅ **Data-driven discovery** finds labels based on actual company needs

### **How It Works**
1. **Bootstrap**: Train initial classifier on companies that map to your 61 known labels
2. **Uncertainty Detection**: Find companies that don't fit well into current labels
3. **Label Discovery**: Present unclear cases to you for new label creation
4. **Iterative Learning**: Retrain model with expanded labels and repeat

## 📋 Prerequisites

### **Required Files**
- `data/processed/master_insurance_to_naics_mapping_simplified.json` (Your 61 mapped labels)
- `data/input/insurance_taxonomy - insurance_taxonomy.csv` (Full 220 label taxonomy)
- `data/input/nine_k_companies_with_naics.csv` (Your 9k companies dataset)

### **Dependencies**
```bash
pip install pandas scikit-learn sentence-transformers numpy
```

## 🎯 Quick Start

### **Option 1: Run Complete Pipeline**
```python
from discovery_active_learning_pipeline import DiscoveryActiveLearningPipeline

# Initialize pipeline
pipeline = DiscoveryActiveLearningPipeline()

# Run complete pipeline (3 iterations)
pipeline.run_complete_pipeline(max_iterations=3)
```

### **Option 2: Step-by-Step Control**
```python
# Initialize
pipeline = DiscoveryActiveLearningPipeline()

# Phase 1: Bootstrap with 61 labels
pipeline.bootstrap_initial_classifier()

# Phase 2: Run one discovery iteration
pipeline.run_discovery_iteration(n_uncertain_cases=10)

# Phase 3: Continue as needed
pipeline.run_discovery_iteration(n_uncertain_cases=10)
```

## 🔍 What Happens During Discovery

### **Uncertainty Detection**
The pipeline identifies companies that:
- Don't fit well into any of your current 61 labels
- Have low confidence predictions
- Represent potential new label categories

### **Interactive Label Discovery**
For each uncertain case, you'll see:
```
🔍 UNCERTAIN CASE 1/10
   Company: "Specialized marine insurance underwriting services"
   Predicted NAICS: 524126
   Current best guess: Insurance Services (confidence: 0.35)
   Uncertainty score: 0.65

🎯 Most Similar Existing Labels:
   1. Insurance Services (similarity: 0.45)
   2. Financial Services (similarity: 0.32)
   3. Risk Management (similarity: 0.28)

🔽 Options:
   1-5: Select similar existing label
   n:   Create new label
   s:   Skip this case
   q:   Quit and save progress
```

### **New Label Creation**
When you choose "n" (new label), you can:
- **Select from unused taxonomy labels** (159 remaining labels)
- **Create completely custom labels** for specialized needs
- **View similarity rankings** to find the best fit

## 📊 Progress Tracking

### **Automatic Progress Saving**
- Progress saved after each iteration
- Resume from where you left off
- Track label expansion over time

### **Progress File Location**
`data/processed/active_learning_progress.json`

### **What's Tracked**
- New labels discovered
- Training examples per label
- Model performance metrics
- Iteration history

## 🎯 Expected Workflow

### **Session 1: Bootstrap (30 minutes)**
- Load 61 known labels
- Train initial classifier
- Review 10 uncertain cases
- **Result**: 65-70 labels

### **Session 2: First Expansion (20 minutes)**
- Retrain with new labels
- Review 10 more uncertain cases
- **Result**: 75-80 labels

### **Session 3: Fine-tuning (15 minutes)**
- Focus on remaining high-uncertainty cases
- **Result**: 85-90 labels

### **Final Result**
- **90+ labels** (40% coverage)
- **3,000-5,000 training examples**
- **Production-ready classifier**

## 💡 Best Practices

### **Label Creation Strategy**
1. **Start conservative**: Use existing taxonomy labels when possible
2. **Be specific**: "Marine Insurance" > "Insurance Services"
3. **Consider frequency**: Prioritize labels that appear often
4. **Think business impact**: Focus on commercially important categories

### **Review Strategy**
1. **Review in batches**: 10-15 cases per session
2. **Take breaks**: Avoid decision fatigue
3. **Be consistent**: Use similar naming conventions
4. **Document reasoning**: Note why certain labels were chosen

## 🔧 Customization Options

### **Adjust Uncertainty Threshold**
```python
pipeline.confidence_threshold = 0.5  # Lower = more uncertain cases
```

### **Change Batch Sizes**
```python
pipeline.run_discovery_iteration(n_uncertain_cases=20)  # More cases per session
```

### **Modify File Paths**
```python
pipeline = DiscoveryActiveLearningPipeline(
    master_map_path="your/custom/path.json",
    companies_data_path="your/companies.csv"
)
```

## 📈 Expected Outcomes

### **Coverage Expansion**
- **Start**: 61 labels (28% coverage)
- **After 3 iterations**: 90+ labels (40%+ coverage)
- **Training data**: 3,000-5,000 examples

### **Quality Metrics**
- **Precision**: 85-90% (high confidence labels)
- **Recall**: 75-85% (good coverage)
- **F1-Score**: 80-87% (balanced performance)

## 🚀 Next Steps After Discovery

### **Production Deployment**
1. **Export final model**: Save trained classifier
2. **Create prediction API**: Deploy for real-time classification
3. **Monitor performance**: Track accuracy on new data
4. **Continuous learning**: Add new labels as business needs evolve

### **Further Expansion**
- **Collect more data**: Expand to 50k+ companies
- **Add new label types**: Discover emerging business categories
- **Improve accuracy**: Fine-tune with more examples per label

## 🆘 Troubleshooting

### **Common Issues**

**"No companies dataset found"**
- Ensure `data/input/nine_k_companies_with_naics.csv` exists
- Check file path in initialization

**"No training data available"**
- Verify Master Map file exists and is formatted correctly
- Check NAICS classifier is working

**"Low number of uncertain cases"**
- Lower confidence threshold: `pipeline.confidence_threshold = 0.4`
- Check if most companies already fit existing labels well

### **Performance Optimization**
- **Reduce batch size** if running out of memory
- **Use GPU** for faster embedding calculations
- **Cache embeddings** for repeated runs

## 📞 Support

### **Debug Information**
Check these logs for troubleshooting:
- Bootstrap success/failure
- Number of companies mapped to known labels
- Classifier training metrics
- Uncertainty detection results

### **Files Generated**
- `data/processed/active_learning_progress.json` - Progress tracking
- `data/processed/taxonomy_embeddings_cache.pkl` - Embedding cache
- Trained model files (saved automatically)

---

## 🎉 Ready to Start?

Run this simple command to begin:

```bash
python discovery_active_learning_pipeline.py
```

The pipeline will guide you through each step, and you'll be expanding your label coverage in no time!

**Remember**: You're starting with a solid foundation of 61 labels, and each iteration will make your system smarter and more comprehensive. 🚀 