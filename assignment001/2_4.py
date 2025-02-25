import numpy as np
import matplotlib.pyplot as plt

# Initial data (age, %fat)
hospital_result = [
    (23, 9.5),
    (23, 26.5),
    (27, 7.8),
    (27, 17.8),
    (39, 31.4),
    (41, 25.9),
    (47, 27.4),
    (49, 27.2),
    (50, 31.2),
    (52, 34.6),
    (54, 42.5),
    (54, 28.8),
    (56, 33.4),
    (57, 30.2),
    (58, 34.1),
    (58, 32.9),
    (60, 41.2),
    (61, 35.7),
]

if __name__ == '__main__':
    # Calculate the mean, median, and standard deviation
    ages = [result[0] for result in hospital_result]
    fats = [result[1] for result in hospital_result]
    mean_age = np.mean(ages)
    mean_fat = np.mean(fats)
    median_age = np.median(ages)
    median_fat = np.median(fats)
    std_age = np.std(ages)
    std_fat = np.std(fats)

    print("(A)")
    print("Mean age: ", mean_age)
    print("Mean %fat: ", mean_fat)
    print("Median age: ", median_age)
    print("Median %fat: ", median_fat)
    print("Standard deviation age: ", std_age)
    print("Standard deviation %fat: ", std_fat)

    # Draw box plots for age and %fat
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.boxplot(ages, vert=False)
    plt.title('Box plot of Ages')
    plt.xlabel('Age')

    plt.subplot(1, 2, 2)
    plt.boxplot(fats, vert=False)
    plt.title('Box plot of %Fat')
    plt.xlabel('%Fat')

    plt.tight_layout()
    plt.show()