"""
Inference script for making predictions with saved models
"""
import os
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from joblib import load

from scripts.config import MODELS_DIR, MODEL_FILENAME, PREPROCESSOR_FILENAME


def load_model_pipeline(model_name: str = None) -> object:
    """Load the trained model pipeline"""
    if model_name:
        model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    else:
        model_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    return load(model_path)


def load_preprocessor(preprocessor_name: str = None) -> object:
    """Load the separate preprocessor pipeline"""
    if preprocessor_name:
        preprocessor_path = os.path.join(MODELS_DIR, preprocessor_name)
    else:
        preprocessor_path = os.path.join(MODELS_DIR, PREPROCESSOR_FILENAME)
    
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")
    
    return load(preprocessor_path)


def predict_single_instance(data: Dict, model_name: str = None) -> Dict:
    """
    Predict for a single instance
    
    Args:
        data: Dictionary of feature values
        model_name: Optional specific model to use
    
    Returns:
        Dictionary with predictions and probabilities
    """
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    # Load model
    model_pipeline = load_model_pipeline(model_name)
    
    # Make prediction
    prediction = model_pipeline.predict(df)[0]
    
    # Get prediction probabilities (for classification)
    if hasattr(model_pipeline.named_steps['model'], 'predict_proba'):
        probabilities = model_pipeline.predict_proba(df)[0]
        prob_dict = {
            f"prob_class_{i}": prob 
            for i, prob in enumerate(probabilities)
        }
    else:
        prob_dict = {}
    
    # For binary classification, also add probability of positive class
    if len(prob_dict) == 2:
        prob_dict["prob_positive"] = prob_dict.get("prob_class_1", 0)
        prob_dict["prob_negative"] = prob_dict.get("prob_class_0", 0)
    
    return {
        "prediction": prediction,
        **prob_dict
    }


def predict_batch(data: pd.DataFrame, model_name: str = None) -> pd.DataFrame:
    """
    Predict for a DataFrame of instances
    
    Args:
        data: DataFrame of feature values
        model_name: Optional specific model to use
    
    Returns:
        DataFrame with predictions added
    """
    # Load model
    model_pipeline = load_model_pipeline(model_name)
    
    # Make predictions
    predictions = model_pipeline.predict(data)
    
    # Get prediction probabilities (for classification)
    if hasattr(model_pipeline.named_steps['model'], 'predict_proba'):
        probabilities = model_pipeline.predict_proba(data)[:, 1]  # Probability of positive class
    else:
        probabilities = np.zeros(len(predictions))
    
    # Add to DataFrame
    result_df = data.copy()
    result_df["prediction"] = predictions
    result_df["probability"] = probabilities
    
    return result_df


def main():
    parser = argparse.ArgumentParser(description="Make predictions with a trained model")
    parser.add_argument(
        "--input", 
        required=True,
        help="Input data (CSV file path or 'single' for single instance input)"
    )
    parser.add_argument(
        "--model", 
        default=None,
        help="Model file name (uses default if not specified)"
    )
    parser.add_argument(
        "--output", 
        default=None,
        help="Output CSV file path for batch predictions"
    )
    
    args = parser.parse_args()
    
    if args.input == "single":
        # Interactive mode for single prediction
        print("Enter feature values (press Enter to use default if any):")
        
        # Define the features based on the dataset
        feature_names = [
            "age", "income", "employment_length", "home_ownership", 
            "purpose", "credit_amount", "interest_rate", "percent_income",
            "has_defaults", "years_at_residence"
        ]
        
        # Default values
        default_values = {
            "age": 35,
            "income": 50000,
            "employment_length": 5,
            "home_ownership": "MORTGAGE",
            "purpose": "debt_consolidation",
            "credit_amount": 10000,
            "interest_rate": 10.5,
            "percent_income": 20,
            "has_defaults": 0,
            "years_at_residence": 3
        }
        
        # Collect user input
        data = {}
        for feature in feature_names:
            user_input = input(f"{feature} [{default_values[feature]}]: ")
            if user_input.strip():
                # Convert to appropriate type
                if feature in ["has_defaults"]:
                    data[feature] = int(user_input)
                elif feature in ["age", "income", "employment_length", "credit_amount", 
                              "interest_rate", "percent_income", "years_at_residence"]:
                    data[feature] = float(user_input)
                else:
                    data[feature] = user_input
            else:
                data[feature] = default_values[feature]
        
        # Make prediction
        result = predict_single_instance(data, args.model)
        
        print("\nPrediction Results:")
        print(f"Credit Score Approval: {'Approved' if result['prediction'] == 1 else 'Rejected'}")
        if 'prob_positive' in result:
            print(f"Approval Probability: {result['prob_positive']:.4f}")
            
    else:
        # Batch prediction
        data_path = args.input
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Input file not found: {data_path}")
        
        # Load data
        data = pd.read_csv(data_path)
        
        # Make predictions
        result_df = predict_batch(data, args.model)
        
        # Save results if output path provided
        if args.output:
            result_df.to_csv(args.output, index=False)
            print(f"Results saved to: {args.output}")
        else:
            # Print preview
            print("\nPrediction Results (preview):")
            if len(result_df) <= 10:
                print(result_df)
            else:
                print(result_df.head(10))
                print(f"\nShowing 10 of {len(result_df)} rows")


if __name__ == "__main__":
    main()