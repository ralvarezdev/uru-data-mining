import stats.distance as distance

# Initial data
objects = [
    (22, 1, 42, 10),
    (20, 0, 36, 8)
]

if __name__ == '__main__':
    print("\nEXERCISE 2.6\n")

    object1 = objects[0]
    object2 = objects[1]

    # Calculate the Euclidean distance
    euclidian_distance = distance.euclidean_distance(object1, object2)
    print("(A)")
    print("Euclidean distance: ", euclidian_distance)

    # Calculate the Manhattan distance
    manhattan_distance = distance.manhattan_distance(object1, object2)
    print("\n(B)")
    print("Manhattan distance: ", manhattan_distance)

    # Calculate the Minkowski distance
    minkowski_distance = distance.minowski_distance(object1, object2, 3)

    print("\n(C)")
    print("Minkowski distance: ", minkowski_distance)

    # Calculate the supremum distance
    supremum_distance = distance.supremum_distance(object1, object2)
    print("\n(D)")
    print("Supremum distance: ", supremum_distance)

