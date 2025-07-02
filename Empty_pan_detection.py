import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

def train_models(merged_csv_path):
    """Train models using the merged CSV file without timestamp dependency."""
    
  
    df = pd.read_csv(merged_csv_path)
    
    # Drop unnecessary columns
    df.drop(columns=['Time(ms)','Induction_Power:','Unnamed: 6','Ambient_Temp:'], inplace=True, errors='ignore')

    # Print column names for debugging
    print("Columns after cleaning:", df.columns)
    
    # Select sensor features
    features = ['PAN_Inside:', 'PAN_Outside:', 'Glass_Temp:', 'Ind_Current:', 'Mag_Current:']
    
    # Convert all feature columns to numeric (force errors to NaN)
    df[features] = df[features].apply(pd.to_numeric, errors='coerce')

    # Drop rows where any feature column is NaN
    df.dropna(subset=features, inplace=True)

    # Print sample of cleaned data
    print("Sample of cleaned data:\n", df.head())

    # Define X and y
    X = df[features]
    y_class = df['empty_pan']  # Classification target

    # Check sample consistency
    print(f"X shape: {X.shape}, y_class length: {len(y_class)}")

    # Initialize models
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    scaler = StandardScaler()

    # Scale features
    X_scaled = scaler.fit_transform(X)

    # Train models
    clf.fit(X_scaled, y_class)

    os.makedirs("models", exist_ok=True)

  
    joblib.dump(clf, "models/pan_classifier.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    print("✅ Models trained successfully and saved in the 'models' directory!")

if __name__ == "__main__":
    train_models("newdata.csv")
    