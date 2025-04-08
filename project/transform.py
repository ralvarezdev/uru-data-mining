from project import TRANSFORMED_ROAD_ACCIDENT_DATASET_FILE
from project.load import load_dataset
from sklearn.preprocessing import LabelEncoder
import os

if __name__ == '__main__':
    print("--- DATA TRANSFORM ---\n")

    # Load the dataset CSV file
    df = load_dataset()

    # Drop duplicate rows
    df.drop_duplicates(inplace=True)

    # Fill NaN values with the median of numeric columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

    # Fill NaN values with the most frequent value of non-numeric columns
    non_numeric_columns = df.select_dtypes(exclude=['float64', 'int64']).columns
    df[non_numeric_columns] = df[non_numeric_columns].apply(lambda x: x.fillna(x.mode()[0]))

    # Encode categorical variables
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype != 'object':
            continue

        # Encode categorical variables
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Save the cleaned dataset to a new CSV file
    transformed_dataset_path = os.path.join(TRANSFORMED_ROAD_ACCIDENT_DATASET_FILE)
    df.to_csv(transformed_dataset_path, index=False)

    print(f"Transformed dataset saved to {transformed_dataset_path}")
