"""
Utility functions for data loading and preprocessing
Optimized for memory efficiency and carbon reduction
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_and_prepare_data(data_path, test_size=0.2, random_state=42, optimize_memory=True):
    """
    Load and prepare data with optional memory optimization
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV file
    test_size : float
        Proportion of data for testing
    random_state : int
        Random seed for reproducibility
    optimize_memory : bool
        Whether to apply memory optimization techniques
    
    Returns:
    --------
    X_train, X_test, y_train, y_test : arrays
        Split and preprocessed data
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Memory optimization: downcast numeric types
    if optimize_memory:
        df = optimize_dataframe_memory(df)
    
    # Separate features and target
    X = df.drop('Malnutrition Status', axis=1)
    y = df['Malnutrition Status']
    
    # Encode categorical variables
    label_encoders = {}
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column].astype(str))
        label_encoders[column] = le
    
    # Encode target variable
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)
    
    # Convert to numpy arrays with optimized dtype
    if optimize_memory:
        X = X.astype(np.float32)
        y = y.astype(np.int32)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def optimize_dataframe_memory(df):
    """
    Optimize DataFrame memory usage by downcasting numeric types
    This reduces memory footprint and computational overhead
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    
    Returns:
    --------
    df : pandas.DataFrame
        Memory-optimized dataframe
    """
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    return df
