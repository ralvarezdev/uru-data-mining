import itertools
from project import COLUMNS

# Generate combinations of different columns from the dataset
def generate_combinations(n:int):
    return list(itertools.combinations(COLUMNS, n))