from project.load import load_transformed_dataset
import seaborn as sns
import matplotlib.pyplot as plt

# Which Road Type has the highest Accident Severity and how does it vary between Urban/Rural areas?

if __name__ == '__main__':
    print("--- QUESTION 4 ---\n")

    # Load the transformed dataset CSV file
    df = load_transformed_dataset()

    # Group by Road Type and Urban/Rural
    road_type_severity = df.groupby(['Road Type', 'Urban/Rural'])['Accident Severity'].mean().reset_index()

    # Plot the results
    sns.barplot(x='Road Type', y='Accident Severity', hue='Urban/Rural', data=road_type_severity)
    plt.title('Accident Severity by Road Type and Urban/Rural Areas')
    plt.show()