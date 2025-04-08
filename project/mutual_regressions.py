import os
from sklearn.feature_selection import mutual_info_regression
import pandas as pd
from project import MUTUAL_REGRESSIONS_DIR, to_snake_case
from project.load import load_transformed_dataset

if __name__ == '__main__':
    print("--- MUTUAL REGRESSIONS ---\n")

    # Load the transformed dataset CSV file
    df = load_transformed_dataset()

    # Verify that the mutual regressions directory exists, if not create it
    if not os.path.exists(MUTUAL_REGRESSIONS_DIR):
        os.makedirs(MUTUAL_REGRESSIONS_DIR)

    for column in df.columns:
        # Select features and target
        X = df.drop(columns=[column])
        y = df[column]

        # Compute mutual information
        mi = mutual_info_regression(X, y)
        mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)

        # Save the results to a CSV file
        mi_file = to_snake_case(f"{column}.csv")
        mi_file_path = os.path.join(MUTUAL_REGRESSIONS_DIR, mi_file)
        mi_series.to_csv(mi_file_path)

        print(f"Mutual information results for {column} saved to {mi_file_path}")
