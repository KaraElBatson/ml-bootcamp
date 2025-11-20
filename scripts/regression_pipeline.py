"""
Regression pipeline for age prediction using SVR
"""
import os
import argparse
import logging
import sys
import warnings
import numpy as np
from typing import Dict, Tuple
from sklearn.pipeline import Pipeline
from scripts.age_prediction_svr import main as run_svr_pipeline

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Age Prediction Pipeline")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run age prediction pipeline with SVR")
    parser.add_argument("--train", default="train.csv", help="Training data file name")
    parser.add_argument("--test", default="test.csv", help="Test data file name")
    parser.add_argument("--output", default="submission_svr.csv", help="Output file name")
    return parser.parse_args()

def run_age_prediction_pipeline(train_file, test_file, output_file):
    """
    Run the age prediction pipeline
    
    Args:
        train_file (str): Name of training data file
        test_file (str): Name of test data file
        output_file (str): Name of output file for predictions
    
    Returns:
        Results dictionary with performance metrics
    """
    logger.info("Starting age prediction pipeline")
    logger.info(f"Training data: {train_file}")
    logger.info(f"Test data: {test_file}")
    logger.info(f"Output file: {output_file}")
    
    # Create directory for outputs if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run the SVR pipeline
    try:
        run_svr_pipeline(train_file, test_file, output_file)
        logger.info("Pipeline completed successfully")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    args = parse_arguments()
    results = run_age_prediction_pipeline(args.train, args.test, args.output)
    
    if results["status"] == "success":
        print("Age prediction pipeline completed successfully!")
    else:
        print("Pipeline failed:")
        print(results.get("message", "Unknown error"))
        sys.exit(1)