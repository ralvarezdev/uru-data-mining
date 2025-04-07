from project.load import load_transformed_dataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Can we predict Accident Severity based on Weather Conditions, Visibility Level, and Time of Day?

if __name__ == '__main__':
    print("--- QUESTION 1 ---\n")

    # Load the transformed dataset CSV file
    df = load_transformed_dataset()

    # Define features and target
    X = df[['Weather Conditions', 'Visibility Level', 'Time of Day']]
    y = df['Accident Severity']

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))