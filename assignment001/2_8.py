import numpy as np
from stats.distance import euclidean_distance, manhattan_distance, supremum_distance, cosine_similarity

# Initial data
dataset = [
    (1.5, 1.7),
    (2.0, 1.9),
    (1.6, 1.8),
    (1.2, 1.5),
    (1.5, 1.0)
]

query_point = (1.4, 1.6)

# Rank points based on a similarity measure
def rank_points(dataset, query_point, distance_func):
    distances = [(point, distance_func(point, query_point)) for point in dataset]
    distances.sort(key=lambda x: x[1])
    return distances

# Print the ranking based on different similarity measures
def print_ranking(ranking):
    for point, dist in ranking:
        print(f"Point: {point}, Distance: {dist}, Similarity: {1 / (1 + dist)}")

if __name__ == '__main__':
    print("\nEXERCISE 2.8\n")

    # Rank based on different similarity measures
    print("(A) Ranking based on different similarity measures:\n")

    # Euclidean distance
    euclidean_ranking = rank_points(dataset, query_point, euclidean_distance)
    print("Euclidean distance ranking:")
    print_ranking(euclidean_ranking)

    # Manhattan distance
    manhattan_ranking = rank_points(dataset, query_point, manhattan_distance)
    print("\nManhattan distance ranking:")
    print_ranking(manhattan_ranking)

    # Supremum distance
    supremum_ranking = rank_points(dataset, query_point, supremum_distance)
    print("\nSupremum distance ranking:")
    print_ranking(supremum_ranking)

    # Cosine similarity (note: higher similarity means closer, so we sort in descending order)
    cosine_ranking = rank_points(dataset, query_point, lambda p1, p2: -cosine_similarity(p1, p2))
    print("\nCosine similarity ranking:")
    for point, sim in cosine_ranking:
        print(f"Point: {point}, Similarity: {-sim}")

    # (b) Normalize the dataset and rank using Euclidean distance
    print("\n(B) Ranking based on Euclidean distance after normalization:\n")

    def normalize(point):
        norm = np.linalg.norm(point)
        return tuple(np.array(point) / norm)

    normalized_dataset = [normalize(point) for point in dataset]
    normalized_query_point = normalize(query_point)

    normalized_euclidean_ranking = rank_points(normalized_dataset, normalized_query_point, euclidean_distance)
    print("Normalized Euclidean distance ranking:")
    for point, dist in normalized_euclidean_ranking:
        print(f"Point: {point}, Distance: {dist}")