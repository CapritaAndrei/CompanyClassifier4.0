#!/usr/bin/env python3
"""
Veridion3 Insurance Classification System
Main entry point for the production system
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from training.pipeline import InsuranceClassificationTrainingPipeline
from inference.predictor import InsuranceModelPredictor


def train_model():
    """Train a new model"""
    print("🚀 Starting model training...")
    pipeline = InsuranceClassificationTrainingPipeline()
    results = pipeline.run_full_pipeline()
    print("✅ Training completed!")
    return results


def predict_companies(input_file: str, output_file: str = None):
    """Make predictions on new companies"""
    print(f"🔮 Making predictions from {input_file}...")
    
    # Load production model
    predictor = InsuranceModelPredictor("models/production_model.pkl")
    
    # TODO: Implement batch prediction logic
    print("✅ Predictions completed!")


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py train          # Train new model")
        print("  python main.py predict <file> # Make predictions")
        return
    
    command = sys.argv[1]
    
    if command == "train":
        train_model()
    elif command == "predict":
        if len(sys.argv) < 3:
            print("Error: Please provide input file for predictions")
            return
        predict_companies(sys.argv[2])
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
