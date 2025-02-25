import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Initial data
ages = [13,15,16,16,19,20,20,21,22,22,25,25,25,25,30,33,33,33,35,35,35,35,36,40,45,46,52,70]

if __name__ == '__main__':
    print("EXERCISE 2.2\n")

    # Calculate the mean and get the median
    mean = np.mean(ages)
    median = np.median(ages)
    print("(A)")
    print("Mean: ", mean)
    print("Median: ", median)

    # Calculate the mode
    mode_result = stats.mode(ages)
    print("\n(B)")
    print("Mode: ", mode_result.mode)

    # Determine the modality
    num_modes = mode_result.count
    modality = None
    if num_modes == 1:
        modality = "unimodal"
    elif num_modes == 2:
        modality = "bimodal"
    elif num_modes == 3:
        modality = "trimodal"
    else:
        modality = f"{num_modes}-modal"
    print("Modality: ", modality)

    # Calculate the midrange
    min_age = min(ages)
    max_age = max(ages)
    midrange = (max_age + min_age) / 2
    print("\n(C)")
    print("Midrange: ", midrange)

    # Calculate roughly the first and third quartiles
    q1 = np.percentile(ages, 25)
    q3 = np.percentile(ages, 75)
    print("\n(D)")
    print("Q1: ", q1)
    print("Q3: ", q3)

    # Five-number summary
    print("\n(E)")
    print("Min: ", min_age)
    print("Q1: ", q1)
    print("Median: ", median)
    print("Q3: ", q3)
    print("Max: ", max_age)

    # Draw box plot, normal Q-Q plot, and quantile plot
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.boxplot(ages, vert=False)
    plt.title('Box plot')
    plt.xlabel('Age')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    stats.probplot(ages, dist="norm", plot=plt)
    plt.title("Normal Q-Q plot")
    plt.grid(True)

    sorted_data = np.sort(ages)
    quantiles = np.linspace(0, 1, len(sorted_data))
    plt.subplot(1, 3, 3)
    plt.plot(quantiles, sorted_data)
    plt.title("Quantile Plot")
    plt.xlabel("Quantiles")
    plt.ylabel("Ordered Values")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


