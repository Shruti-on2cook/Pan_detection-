import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_models(merged_csv_path):
    """Train models using the merged CSV file with proper train-test split."""
    
    df = pd.read_csv(merged_csv_path)
    
    # Drop unnecessary columns
    df.drop(columns=['Time(ms)', 'Induction_Power:', 'Unnamed: 6', 'Ambient_Temp:'], inplace=True, errors='ignore')

    print("Columns after cleaning:", df.columns)
    
    # Select sensor features
    features = ['PAN_Inside:', 'PAN_Outside:', 'Glass_Temp:', 'Ind_Current:', 'Mag_Current:']
    
    # Convert all feature columns to numeric (force errors to NaN)
    df[features] = df[features].apply(pd.to_numeric, errors='coerce')

    # Drop rows where any feature column is NaN
    df.dropna(subset=features, inplace=True)

    print("Sample of cleaned data:\n", df.head())

    # Define X and y
    X = df[features]
    y_class = df['empty_pan']

    print(f"X shape: {X.shape}, y_class length: {len(y_class)}")

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize Random Forest model
    clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )

    # Train the model
    clf.fit(X_train_scaled, y_train)

    # Predict on test set
    y_pred = clf.predict(X_test_scaled)

    # Evaluate metrics on test data
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save model and scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/pan_classifier.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    print("✅ Models trained successfully and saved in the 'models' directory!")

if __name__ == "__main__":
    train_models("newdata.csv")
