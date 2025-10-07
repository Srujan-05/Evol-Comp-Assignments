import numpy as np
import matplotlib.pyplot as plt


class FuzzyMap:
    def __init__(self, bins, centroids: np.array):
        self.bins = len(centroids) if bins is None else bins
        self.centroids = centroids if centroids is not None else np.linspace(-1, 1, bins)


class Normalization:
    def __init__(self, input: np.array):
        self.input = input


class FuzzyLogic:
    def __init__(self, input_data: np.array, input_fuzzy_maps: list, output_fuzzy_map: FuzzyMap, fam: np.array):
        self.input_data = input_data
        self.input_fuzzy_maps = input_fuzzy_maps
        self.output_fuzzy_maps = output_fuzzy_map
        self.fam = fam


if __name__ == "__main__":
    pass
