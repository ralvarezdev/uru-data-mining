import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np

from project import RANDOM_FOREST_DIR, to_snake_case
from project.load import load_transformed_dataset

if __name__ == '__main__':
    print("--- RANDOM FOREST ---\n")

    # Load the transformed dataset CSV file
    df = load_transformed_dataset()

    # Verify that the random forest directory exists, if not create it
    if not os.path.exists(RANDOM_FOREST_DIR):
        os.makedirs(RANDOM_FOREST_DIR)

    for column in df.columns:
        # Select features and target
        X = df.drop(columns=[column])
        y = df[column]

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train a Random Forest model
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)

        # Get feature importance scores
        feature_importances = rf.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        # Save the results to a CSV file
        importance_file = to_snake_case(f"{column}.csv")
        importance_file_path = os.path.join(RANDOM_FOREST_DIR, importance_file)
        importance_df.to_csv(importance_file_path, index=False)

        print(f"Feature importance results for {column} saved to {importance_file}")