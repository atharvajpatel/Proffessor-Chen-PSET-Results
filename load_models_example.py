#!/usr/bin/env python3
"""
Example script to load and use pre-trained models
Run this after training models to demonstrate model loading
"""

import numpy as np
import pandas as pd
from functions.model_training import load_trained_models
from functions.data_preprocessing import load_diabetes_data, preprocess_data

def main():
    """
    Demonstrate loading and using pre-trained models
    """
    print("=" * 60)
    print("LOADING PRE-TRAINED MODELS")
    print("=" * 60)
    
    try:
        # Load saved models
        model_objects, model_results, metadata = load_trained_models("trained_models")
        
        print("✅ Successfully loaded models!")
        print(f"Models trained on: {metadata['timestamp']}")
        print(f"Seed used: {metadata['seed']}")
        print(f"GPU used: {metadata['gpu_used']}")
        print(f"Available models: {metadata['models']}")
        
        print("\n" + "=" * 40)
        print("MODEL PERFORMANCE SUMMARY")
        print("=" * 40)
        
        for name, results in model_results.items():
            auc = results['test_auc']
            ci_lower, ci_upper = results['test_auc_ci']
            print(f"{name:20s}: {auc:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        # Demonstrate using a loaded model for predictions
        print("\n" + "=" * 40)
        print("DEMONSTRATION: USING LOADED MODELS")
        print("=" * 40)
        
        # Load some test data
        df, y, ids = load_diabetes_data()
        data_dict = preprocess_data(df, y)
        X_test = data_dict['X_test']
        y_test = data_dict['y_test']
        
        # Use the best model (you can change this)
        best_model_name = max(model_results.keys(), 
                            key=lambda x: model_results[x]['test_auc'])
        best_model = model_objects[best_model_name]
        
        print(f"Using best model: {best_model_name}")
        
        # Make predictions
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        y_pred = best_model.predict(X_test)
        
        # Show some sample predictions
        print(f"Sample predictions (first 10):")
        print(f"Probabilities: {y_pred_proba[:10]}")
        print(f"Predictions:   {y_pred[:10]}")
        print(f"True labels:   {y_test[:10].values if hasattr(y_test, 'values') else y_test[:10]}")
        
        print("\n✅ Model loading and prediction successful!")
        print("You can now use these pre-trained models in downstream analysis.")
        
    except FileNotFoundError:
        print("❌ No trained models found.")
        print("Please run the main analysis first to train and save models.")
    except Exception as e:
        print(f"❌ Error loading models: {e}")

if __name__ == "__main__":
    main()
