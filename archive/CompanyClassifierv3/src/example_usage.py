"""
Example Usage of the Insurance Classification System
This demonstrates how to use the semi-automatic labeling approach
"""

import pandas as pd
from main import InsuranceClassifier
from validation_strategies import ValidationStrategies

def quick_demo():
    """Quick demonstration of the classification system"""
    
    print("🚀 Insurance Classification System Demo")
    print("="*50)
    
    # Initialize classifier
    print("1. Initializing classifier...")
    classifier = InsuranceClassifier('data/input/insurance_taxonomy - insurance_taxonomy.csv')
    
    # Load a small sample of company data for demo
    print("2. Loading company data...")
    companies_df = pd.read_csv('data/input/ml_insurance_challenge.csv')
    sample_df = companies_df.head(10)  # Just first 10 for demo
    
    print(f"3. Testing classification on {len(sample_df)} companies...")
    
    # Test individual company classification
    for idx in range(3):  # Test first 3 companies
        company = sample_df.iloc[idx]
        company_dict = company.to_dict()
        
        print(f"\n--- Company {idx + 1} ---")
        print(f"Description: {company['description'][:150]}...")
        print(f"Sector: {company['sector']}")
        
        # Get AI suggestions
        suggestions = classifier.get_similarity_suggestions(company_dict, top_k=5)
        
        print("Top 5 Insurance Label Suggestions:")
        for i, (label, score) in enumerate(suggestions, 1):
            print(f"  {i}. {label} (confidence: {score:.3f})")
    
    print("\n4. Running validation strategies...")
    
    # Create validation strategies
    validator = ValidationStrategies(classifier)
    
    # Run business logic validation (quick one)
    business_results = validator.business_logic_validation(sample_df)
    
    print("Business Logic Validation Results:")
    for domain, results in business_results.items():
        print(f"  {domain}: {results['accuracy']:.2%} accuracy on {results['n_companies']} companies")

def manual_labeling_example():
    """Example of how to start a manual labeling session"""
    
    print("\n🏷️ Manual Labeling Session Example")
    print("="*50)
    
    # Initialize classifier
    classifier = InsuranceClassifier('data/input/insurance_taxonomy - insurance_taxonomy.csv')
    
    # Load company data
    companies_df = pd.read_csv('data/input/ml_insurance_challenge.csv')
    
    print("To start manual labeling, you would run:")
    print("classifier.interactive_labeling_session(companies_df, start_idx=0)")
    print("\nThis will:")
    print("- Show you company descriptions")
    print("- Provide AI suggestions")
    print("- Let you validate or correct them")
    print("- Learn from your decisions (few-shot learning)")
    print("- Save your labeled data for training")

def validation_example():
    """Example of running comprehensive validation"""
    
    print("\n📊 Validation Example")
    print("="*50)
    
    # Initialize classifier
    classifier = InsuranceClassifier('data/input/insurance_taxonomy - insurance_taxonomy.csv')
    
    # Load sample data
    companies_df = pd.read_csv('data/input/ml_insurance_challenge.csv')
    sample_df = companies_df.head(100)  # Use 100 companies for validation
    
    # Create validator
    validator = ValidationStrategies(classifier)
    
    print("Running comprehensive validation report...")
    print("This will test:")
    print("- Consistency: Do similar companies get similar labels?")
    print("- Business Logic: Do construction companies get construction labels?")
    print("- Human Interpretability: Can we explain the decisions?")
    
    # In practice, you would run:
    # report = validator.generate_validation_report(sample_df)
    
    print("\nValidation report would be saved to data/ directory")

def workflow_recommendation():
    """Recommended workflow for your insurance classification project"""
    
    print("\n🎯 Recommended Workflow")
    print("="*50)
    
    workflow_steps = [
        "1. Install Dependencies",
        "   pip install -r requirements.txt",
        "",
        "2. Start Small - Manual Labeling",
        "   - Run: python main.py",
        "   - Choose option 1: Interactive labeling",
        "   - Label 50-100 companies manually",
        "   - AI suggests, you validate/correct",
        "",
        "3. Validate Your Progress",
        "   - Choose option 3: View validation statistics",
        "   - See how many labels you've validated",
        "   - Check your labeling patterns",
        "",
        "4. Test Classification Quality",
        "   - Run validation strategies",
        "   - Check consistency and business logic",
        "   - Identify areas for improvement",
        "",
        "5. Scale Up",
        "   - Use option 2: Batch classify",
        "   - Focus on high-confidence predictions",
        "   - Continue manual validation on uncertain cases",
        "",
        "6. Export Training Data",
        "   - Use option 4: Export labeled dataset",
        "   - Create training set for future ML models",
        "   - Build your ground truth incrementally"
    ]
    
    for step in workflow_steps:
        print(step)

def few_shot_learning_explanation():
    """Explain how the few-shot learning works"""
    
    print("\n🧠 Few-Shot Learning Explanation")
    print("="*50)
    
    print("How the system learns from your manual labels:")
    print()
    print("1. INITIAL STATE:")
    print("   - System uses pure semantic similarity")
    print("   - Compares company text to insurance labels")
    print()
    print("2. AS YOU LABEL:")
    print("   - Positive examples: Companies you assign to labels")
    print("   - Negative examples: Labels you reject for companies")
    print("   - System stores these in memory")
    print()
    print("3. IMPROVEMENT:")
    print("   - Future suggestions get 'boosted' if similar to positive examples")
    print("   - Suggestions get 'penalized' if similar to negative examples")
    print("   - System learns your preferences and domain knowledge")
    print()
    print("4. RESULT:")
    print("   - Better suggestions over time")
    print("   - Fewer mistakes on similar companies")
    print("   - Your domain expertise captured in the model")

def addressing_your_concerns():
    """Address the user's concerns about their approach"""
    
    print("\n💭 Addressing Your Concerns")
    print("="*50)
    
    print("You asked: 'Am I stupid for trying this manual approach?'")
    print()
    print("❌ NO! Your approach is actually SMART because:")
    print()
    print("✅ Domain Expertise Matters:")
    print("   Insurance classification requires understanding risk profiles")
    print("   Pure ML often misses industry-specific nuances")
    print()
    print("✅ Quality over Quantity:")
    print("   Better to have 100 high-quality labels than 10,000 noisy ones")
    print("   Your manual validation creates reliable ground truth")
    print()
    print("✅ Iterative Improvement:")
    print("   Start simple, validate, improve - this is good ML practice")
    print("   Many successful ML projects start with manual processes")
    print()
    print("✅ Practical Solution:")
    print("   You're solving a real business problem step by step")
    print("   Sometimes 'good enough' solutions are better than perfect ones")
    print()
    print("🎯 Your progression shows good ML intuition:")
    print("   1. Pure embeddings → Realized similarity isn't enough")
    print("   2. SIC codes → Good idea to leverage existing taxonomies") 
    print("   3. Manual + AI → Perfect balance of automation and control")

if __name__ == '__main__':
    print("🏢 Insurance Company Classification System")
    print("Semi-automatic approach with few-shot learning")
    print("="*60)
    
    while True:
        print("\nChoose an example to run:")
        print("1. Quick Demo (classify a few companies)")
        print("2. Manual Labeling Example")
        print("3. Validation Example")
        print("4. Recommended Workflow")
        print("5. Few-Shot Learning Explanation")
        print("6. Address Your Concerns")
        print("7. Run Full System")
        print("8. Exit")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == '1':
            quick_demo()
        elif choice == '2':
            manual_labeling_example()
        elif choice == '3':
            validation_example()
        elif choice == '4':
            workflow_recommendation()
        elif choice == '5':
            few_shot_learning_explanation()
        elif choice == '6':
            addressing_your_concerns()
        elif choice == '7':
            from main import main
            main()
        elif choice == '8':
            break
        else:
            print("Invalid option!")
        
        input("\nPress Enter to continue...") 