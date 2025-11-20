"""
Script to generate synthetic credit scoring data
"""
import os
import pandas as pd
import numpy as np
from typing import Optional

from scripts.config import RAW_DATA_DIR


def generate_dataset(
    n_samples: int = 1000,
    output_file: str = "credit_scoring_data.csv",
    random_state: int = 42
) -> None:
    """Generate a synthetic credit scoring dataset"""
    np.random.seed(random_state)
    
    # Define features
    home_ownership = ["RENT", "MORTGAGE", "OWN", "OTHER"]
    purpose = ["debt_consolidation", "home_improvement", "major_purchase", 
               "other", "medical", "small_business", "car", "vacation", 
               "moving", "renewable_energy"]
    
    # Generate features
    data = {
        "age": np.random.randint(18, 85, n_samples),
        "income": np.random.gamma(2, 20000, n_samples),  # Income distribution
        "employment_length": np.random.exponential(8, n_samples),
        "home_ownership": np.random.choice(home_ownership, n_samples, p=[0.4, 0.45, 0.1, 0.05]),
        "purpose": np.random.choice(purpose, n_samples),
        "credit_amount": np.random.gamma(2, 5000, n_samples),
        "interest_rate": np.random.normal(12, 3, n_samples),
        "percent_income": np.random.uniform(5, 40, n_samples),
        "has_defaults": np.random.binomial(1, 0.2, n_samples),  # 20% have defaults
        "years_at_residence": np.random.exponential(8, n_samples)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure reasonable ranges
    df.loc[df["age"] < 18, "age"] = 18
    df.loc[df["age"] > 85, "age"] = 85
    df.loc[df["income"] < 15000, "income"] = 15000
    
    # Create target variable (credit_score = 1 if approved, 0 if rejected)
    # This creates a target based on reasonable credit risk factors
    score_factors = (
        df["income"] / 100000 -  # Higher income increases chance
        df["interest_rate"] / 20 +  # Higher interest rate decreases chance
        (df["percent_income"] > 25) * 0.3 -  # High credit % of income decreases chance
        df["has_defaults"] * 0.5 -  # Previous defaults decrease chance
        (df["employment_length"] < 2) * 0.2 +  # Short employment decreases chance
        (df["credit_amount"] > 20000) * 0.2  # High credit amount decreases chance
    )
    
    # Convert score to probability and then to binary outcome
    prob = 1 / (1 + np.exp(-score_factors))
    df["credit_score"] = np.random.binomial(1, prob)
    
    print(f"Generated {n_samples} samples")
    print(f"Approval rate: {df['credit_score'].mean():.2%}")
    
    # Save to CSV
    output_path = os.path.join(RAW_DATA_DIR, output_file)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to: {output_path}")
    
    return df


def generate_sample_dataset(n_samples: int = 1000):
    """Wrapper function for notebooks to import"""
    return generate_dataset(n_samples)


if __name__ == "__main__":
    # Generate sample data if run directly
    n_samples = 1000
    output_file = "credit_scoring_data.csv"
    
    generate_dataset(n_samples, output_file)