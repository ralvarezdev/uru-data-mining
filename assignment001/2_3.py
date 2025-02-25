# Initial data
intervals = [
    (1, 5, 200),
    (6, 15, 450),
    (16, 20, 300),
    (21, 50, 1500),
    (51, 80, 700),
    (81, 110, 44)
]

if __name__ == '__main__':
    # Calculate an approximate median
    total_frequency = 0
    for interval in intervals:
        total_frequency += interval[2]
    half_frequency_total = total_frequency / 2

    median = 0
    accumulated_frequency = 0
    for interval in intervals:
        frequency = interval[2]
        accumulated_frequency += frequency
        if accumulated_frequency >= half_frequency_total:
            lower_limit = interval[0]
            upper_limit = interval[1]
            amplitude = upper_limit - lower_limit

            median = lower_limit + amplitude * (half_frequency_total - (accumulated_frequency-frequency)) / frequency
            break
    print("(A)")
    print("Median: ", median)
