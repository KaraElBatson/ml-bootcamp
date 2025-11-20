"""
Complete ML pipeline
"""
import os
import argparse
import logging
import warnings
from typing import Dict, Tuple
import pandas as pd
from sklearn.pipeline import Pipeline
# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ML Pipeline")

# Import all components
from scripts.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_CONFIG,
    LGBM_PARAMS, RF_PARAMS, XGB_PARAMS,
    MODELS_DIR, MODEL_FILENAME, PREPROCESSOR_FILENAME
)
from scripts.data_loader import load_csv, split_data, build_preprocessor
from scripts.feature_engineering import DropHighMissing, DropHighCorrelation
from scripts.model_utils import get_models, evaluate_model, save_model, load_model

def run_pipeline(data_file: str, target_column: str = None) -> Dict:
    """
    Run the complete ML pipeline
    Returns a dictionary with model performance metrics
    """
    # Ensure directories exist
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {data_file}")
    df = load_csv(data_file)
    logger.info(f"Data shape: {df.shape}")
    
    # Set target column
    target = target_column or MODEL_CONFIG["target_column"]
    MODEL_CONFIG["target_column"] = target
    
    # Split train/test
    X_train, X_test, y_train, y_test = split_data(df, target)
    logger.info(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # Build preprocessor
    logger.info("Building preprocessor")
    preprocessor = build_preprocessor(X_train)
    
    # Apply preprocessing
    logger.info("Applying preprocessing")
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    
    # Get models
    logger.info("Preparing models")
    models = get_models()
    
    # Evaluate models
    results = {}
    best_model_name = None
    best_score = 0
    metrics = []
    
    for name, model in models.items():
        logger.info(f"Evaluating {name}")
        eval_scores = evaluate_model(
            model, X_train, X_test, y_train, y_test, 
            preprocessor=preprocessor, model_name=name
        )
        results[name] = eval_scores
        
        # Track best model based on ROC AUC
        if eval_scores["roc_auc"] > best_score:
            best_score = eval_scores["roc_auc"]
            best_model_name = name
            
        logger.info(f"{name} ROC AUC: {eval_scores['roc_auc']:.4f}")
    
    # Create full pipeline with best model
    if best_model_name:
        logger.info(f"Creating pipeline with best model: {best_model_name}")
        best_model = models[best_model_name]
        
        final_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", best_model)
        ])
        
        # Train on full training data
        final_pipeline.fit(X_train, y_train)
        
        # Save model
        logger.info("Saving model and preprocessor")
        save_model(final_pipeline, MODEL_FILENAME)
        save_model(preprocessor, PREPROCESSOR_FILENAME)
    
    results["best_model"] = best_model_name
    results["model_params"] = {
        "lgbm": LGBM_PARAMS,
        "rf": RF_PARAMS,
        "xgb": XGB_PARAMS
    }
    
    logger.info(f"Pipeline completed. Best model: {best_model_name}")
    return results


if __name__ == "__main__":
    import argparse
    from scripts.generate_sample_data import generate_sample_dataset
    
    # Check if data file exists
    data_file = "credit_scoring_data.csv"
    data_path = os.path.join(RAW_DATA_DIR, data_file)
    
    parser = argparse.ArgumentParser(description="Run ML pipeline")
    parser.add_argument("--data", default=data_file, help="Data file name")
    parser.add_argument("--target", default="credit_score", help="Target column name")
    parser.add_argument("--generate", action="store_true", help="Generate sample data if not available")
    
    args = parser.parse_args()
    
    # Generate sample data if requested or if file doesn't exist
    if args.generate or not os.path.exists(data_path):
        generate_sample_dataset(n_samples=1000)
    
    # Run pipeline
    results = run_pipeline(args.data, args.target)
    
    # Print results
    print("\n=== Model Evaluation Results ===")
    for name, scores in results.items():
        if name in ["best_model", "model_params"]:
            continue
        print(f"\n{name}:")
        for metric, value in scores.items():
            print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
    
    print(f"\nBest Model: {results['best_model']}")