import numpy as np

# Calculate the Euclidean distance between two points
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))

# Calculate the Manhattan distance between two points
def manhattan_distance(p1, p2):
    return np.sum(np.abs(np.array(p1) - np.array(p2)))

# Calculate the Minkowski distance between two points
def minowski_distance(p1, p2, q):
    return np.sum(np.abs(np.array(p1) - np.array(p2)) ** q) ** (1/q)

# Calculate the supremum distance between two points
def supremum_distance(p1, p2):
    return np.max(np.abs(np.array(p1) - np.array(p2)))

# Calculate the cosine similarity between two points
def cosine_similarity(p1, p2):
    return np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))
