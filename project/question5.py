from project.load import load_transformed_dataset

# Is there a correlation between Driver Alcohol Level and Number of Injuries or Number of Fatalities?

if __name__ == '__main__':
    print("--- QUESTION 5 ---\n")

    # Load the transformed dataset CSV file
    df = load_transformed_dataset()

    # Calculate correlation
    correlation_injuries = df['Driver Alcohol Level'].corr(df['Number of Injuries'])
    correlation_fatalities = df['Driver Alcohol Level'].corr(df['Number of Fatalities'])

    print(f'Correlation between Driver Alcohol Level and Number of Injuries: {correlation_injuries}')
    print(f'Correlation between Driver Alcohol Level and Number of Fatalities: {correlation_fatalities}')