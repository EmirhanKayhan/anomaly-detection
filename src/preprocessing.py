import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_csv(filepath):
    """
    Load and preprocess the IOT dataset
    """
    # Read the CSV file with comma delimiter
    df = pd.read_csv(filepath)
    
    # Remove 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    # Define categorical and numeric columns
    categorical_columns = ['proto', 'service', 'Attack_type']
    
    # Convert categorical columns
    for col in categorical_columns:
        df[col] = df[col].astype('category')
    
    # Convert all other columns to numeric
    numeric_columns = [col for col in df.columns if col not in categorical_columns]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def prepare_train_test_data(df):
    """
    Prepare training and testing datasets
    """
    # Separate features and target
    X = df.drop('Attack_type', axis=1)
    y = df['Attack_type']
    
    # Convert categorical variables to dummy variables
    X = pd.get_dummies(X, columns=['proto', 'service'])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the numeric features
    scaler = StandardScaler()
    
    # Get numeric column names
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
    
    # Scale numeric columns
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    try:
        # Load and preprocess the data
        print("Loading and preprocessing data...")
        df = load_and_preprocess_csv('/home/emirhan/anomaly-detection/data/RT_IOT2022.csv')
        
        # Save cleaned dataset
        output_path = '/home/emirhan/anomaly-detection/data/RT_IOT2022_cleaned.csv'
        df.to_csv(output_path, index=False)
        print(f"Cleaned dataset saved to: {output_path}")
        
        # Prepare train and test datasets
        print("\nPreparing train and test datasets...")
        X_train, X_test, y_train, y_test = prepare_train_test_data(df)
        
        # Save train and test datasets
        X_train.to_csv('/home/emirhan/anomaly-detection/data/X_train.csv', index=False)
        X_test.to_csv('/home/emirhan/anomaly-detection/data/X_test.csv', index=False)
        y_train.to_csv('/home/emirhan/anomaly-detection/data/y_train.csv', index=False)
        y_test.to_csv('/home/emirhan/anomaly-detection/data/y_test.csv', index=False)
        print("Train and test datasets saved successfully!")
        
        # Print information about the dataset
        print("\nDataset Info:")
        print(df.info())
        
        print("\nDataset shape:", df.shape)
        print("Training set shape:", X_train.shape)
        print("Testing set shape:", X_test.shape)
        
        # Print unique values in categorical columns
        print("\nUnique values in categorical columns:")
        for col in ['proto', 'service', 'Attack_type']:
            print(f"\n{col}:", df[col].unique())
        
        # Print basic statistics
        print("\nBasic statistics for numeric columns:")
        print(df.describe())
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise
