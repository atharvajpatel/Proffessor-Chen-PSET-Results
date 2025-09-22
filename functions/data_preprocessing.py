"""
Data Preprocessing Module for Diabetes Readmission Analysis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def load_diabetes_data(data_path="diabetic_data.csv", ids_path="IDS_mapping.csv"):
    """
    Load and perform initial inspection of diabetes dataset
    
    Returns:
        df: Raw dataframe
        y: Binary outcome (30-day readmission)
        ids: ID mapping dataframe
    """
    # Load data
    df = pd.read_csv(data_path)
    ids = pd.read_csv(ids_path) if ids_path else None
    
    print("Dataset shape:", df.shape)
    print("\nTarget variable distribution:")
    print(df['readmitted'].value_counts())
    
    # Create binary outcome: 30-day readmission
    y = (df["readmitted"] == "<30").astype(int)
    print(f"\n30-day readmission rate: {y.mean():.3f}")
    
    return df, y, ids


def preprocess_data(df, y, test_size=0.2, valid_size=0.25, random_state=42):
    """
    Clean and preprocess the diabetes dataset
    
    Args:
        df: Raw dataframe
        y: Target variable
        test_size: Proportion for test set
        valid_size: Proportion of remaining data for validation
        random_state: Random seed
    
    Returns:
        dict with processed data and metadata
    """
    print("Starting data preprocessing...")
    
    # Check missing values
    print("Missing values per column:")
    missing_counts = df.isnull().sum()
    print(missing_counts[missing_counts > 0])
    
    # Drop identifier columns
    cols_to_drop = ['encounter_id', 'patient_nbr', 'readmitted']
    df_clean = df.drop(columns=cols_to_drop)
    
    # Handle missing values - replace '?' with NaN
    df_clean = df_clean.replace('?', np.nan)
    
    # Separate categorical and numeric columns
    categorical_cols = []
    numeric_cols = []
    
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            categorical_cols.append(col)
        else:
            # Convert to numeric if possible
            try:
                df_clean[col] = pd.to_numeric(df_clean[col])
                numeric_cols.append(col)
            except:
                categorical_cols.append(col)
    
    print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols[:10]}...")
    print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
    
    # Fill missing values
    for col in categorical_cols:
        df_clean[col] = df_clean[col].fillna('Unknown')
    
    for col in numeric_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    print(f"\nFinal dataset shape: {df_clean.shape}")
    print("No more missing values:", df_clean.isnull().sum().sum() == 0)
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
    
    # Train/validation/test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        df_clean, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp, test_size=valid_size, stratify=y_temp, random_state=random_state
    )
    
    print(f"\nData splits:")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_valid.shape[0]} samples") 
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Readmission rates - Train: {y_train.mean():.3f}, Valid: {y_valid.mean():.3f}, Test: {y_test.mean():.3f}")
    
    return {
        'X_train': X_train,
        'X_valid': X_valid,
        'X_test': X_test,
        'y_train': y_train,
        'y_valid': y_valid,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'categorical_cols': categorical_cols,
        'numeric_cols': numeric_cols,
        'df_clean': df_clean
    }


def get_processed_features(preprocessor, X_train):
    """
    Apply preprocessing and return processed feature matrix
    
    Returns:
        X_processed: Preprocessed feature matrix
        feature_names: List of feature names after preprocessing
    """
    # Fit and transform
    X_processed = preprocessor.fit_transform(X_train)
    
    # Get feature names
    numeric_features = preprocessor.transformers_[0][2]
    categorical_features = preprocessor.transformers_[1][1].get_feature_names_out(
        preprocessor.transformers_[1][2]
    )
    
    feature_names = list(numeric_features) + list(categorical_features)
    
    print(f"Processed feature matrix shape: {X_processed.shape}")
    print(f"Total features after preprocessing: {len(feature_names)}")
    
    return X_processed, feature_names
