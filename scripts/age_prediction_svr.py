"""
Age prediction script using Support Vector Regression
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def load_data(train_path, test_path):
    """Load training and test datasets"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def preprocess_data(train_df, test_df):
    """Preprocess the data for SVR model"""
    # Define features and target
    y_train = train_df["Age"]
    X_train = train_df.drop(columns=["id", "Age"])
    X_test = test_df.drop(columns=["id"])
    
    # Find common features between train and test
    common_features = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_features]
    X_test = X_test[common_features]
    
    # Identify categorical and numerical columns
    categorical_cols = [col for col in common_features if train_df[col].dtype == 'object']
    numerical_cols = [col for col in common_features if train_df[col].dtype != 'object']
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    # Fit and transform the data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    return X_train_processed, X_test_processed, y_train, preprocessor

def train_and_evaluate_svr(X_train, y_train, validation_size=0.2, random_state=42):
    """Train and evaluate SVR model with validation set"""
    # Split training data into train and validation sets
    X_train_svr, X_val_svr, y_train_svr, y_val_svr = train_test_split(
        X_train, y_train, test_size=validation_size, random_state=random_state
    )
    
    # Create SVR model
    svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    
    # Train model
    svr_model.fit(X_train_svr, y_train_svr)
    
    # Evaluate on validation set
    y_pred_val = svr_model.predict(X_val_svr)
    mae_val = mean_absolute_error(y_val_svr, y_pred_val)
    
    print("========== Support Vector Regression Results ==========")
    print(f"Validation set MAE (Mean Absolute Error): {mae_val:.4f}")
    print("========================================================")
    
    return svr_model

def train_final_svr(X_train, y_train):
    """Train final SVR model on all training data"""
    final_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    final_svr_model.fit(X_train, y_train)
    return final_svr_model

def make_predictions(model, X_test, test_ids, output_file="submission_svr.csv"):
    """Make predictions on test data and save to file"""
    y_pred_test = model.predict(X_test)
    
    # Round predictions to nearest integer and convert to int
    y_pred_test_rounded = y_pred_test.round().astype(int)
    
    # Create submission dataframe
    submission_df = pd.DataFrame({'id': test_ids, 'Age': y_pred_test_rounded})
    submission_df.to_csv(output_file, index=False)
    print(f"Test predictions saved to '{output_file}'")

def main(train_file, test_file, output_file="submission_svr.csv"):
    """Main function to run the SVR age prediction pipeline"""
    # Define file paths (assumes files are in data/raw directory)
    train_path = os.path.join("data", "raw", train_file)
    test_path = os.path.join("data", "raw", test_file)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load data
    print("Loading data...")
    train_df, test_df = load_data(train_path, test_path)
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Preprocess data
    print("Preprocessing data...")
    X_train_processed, X_test_processed, y_train, preprocessor = preprocess_data(train_df, test_df)
    
    # Train and evaluate with validation
    print("Training and evaluating SVR model...")
    svr_model = train_and_evaluate_svr(X_train_processed, y_train)
    
    # Train final model on all data
    print("Training final model on all training data...")
    final_svr_model = train_final_svr(X_train_processed, y_train)
    
    # Make predictions
    print("Making predictions on test data...")
    make_predictions(final_svr_model, X_test_processed, test_df['id'], output_file)
    
    print("Age prediction pipeline completed successfully!")

if __name__ == "__main__":
    # Example usage:
    # main("train.csv", "test.csv")
    pass