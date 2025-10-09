import numpy as np
import matplotlib.pyplot as plt


class FuzzyMap:
    def __init__(self, bins, centroids: np.array):
        self.bins = len(centroids) if bins is None else bins
        self.centroids = centroids if centroids is not None else np.linspace(-1, 1, bins)

    def mu(self, x: float) -> np.array:
        belonging = [0 for i in range(self.bins)]
        if x <= self.centroids[0]:
            belonging[0] = 1/(self.centroids[0] + 1) * (x + 1)
        elif x >= self.centroids[-1]:
            belonging[-1] = 1 / (self.centroids[-1] - 1) * (x - 1)
        else:
            for i in range(1, self.bins):
                if self.centroids[i - 1] <= x <= self.centroids[i]:
                    belonging[i] = 1 / (self.centroids[i] - self.centroids[i-1]) * (x - self.centroids[i-1])
                    belonging[i-1] = 1 - belonging[i]
                    break

        return np.array(belonging)

    def defuzzify(self, belong: np.array):
        return np.dot(belong, self.centroids)/np.sum(belong)


class Normalization:
    def __init__(self, minLim: int, maxLim: int):
        self.minVal = minLim
        self.maxVal = maxLim

    def normalize(self, x):
        return (2*x - (self.maxVal + self.minVal))/(self.maxVal-self.minVal)

    def denormalize(self, d):
        return (d * (self.maxVal - self.minVal) + (self.maxVal + self.minVal)) / 2


class FuzzyLogic:
    def __init__(self, input_data: np.array, input_ranges: list, input_fuzzy_maps: list, output_fuzzy_map: FuzzyMap, fam: np.array):
        self.input_data = input_data
        self.input_fuzzy_maps = input_fuzzy_maps
        self.output_fuzzy_map = output_fuzzy_map
        self.fam = fam
        self.input_ranges = input_ranges
        self.normalizers = [Normalization(range[0], range[1]) for range in input_ranges]

    def solve(self):
        normalized_outputs = []
        input_list = self.input_data

        for input_data in input_list:
            normalized_inputs = []
            for i, data in enumerate(input_data):
                norm_data = self.normalizers[i].normalize(data)
                normalized_inputs.append(norm_data)
            
            input_memberships = []
            for i, norm_input in enumerate(normalized_inputs):
                fuzzy_val = self.input_fuzzy_maps[i].mu(norm_input)
                input_memberships.append(fuzzy_val)

            firing_intensities = np.zeros_like(self.fam, dtype = float)

            for i in range(len(input_memberships[0])):
                for j in range(len(input_memberships[1])):
                    firing_intensities[j, i] = input_memberships[0][i] * input_memberships[1][j]

            numerator, denominator = 0.0, 0.0

            for i in range(firing_intensities.shape[0]):
                for j in range(firing_intensities.shape[1]):
                    firing_strength = firing_intensities[i, j]
                    output_set_index = self.fam[i, j] - 1

                    if output_set_index < len(self.output_fuzzy_map.centroids) and firing_strength > 0:
                        centroid = self.output_fuzzy_map.centroids[output_set_index]
                        numerator += firing_strength * centroid
                        denominator += firing_strength

            normalized_output = numerator / denominator if denominator != 0 else 0.0
            normalized_outputs.append(normalized_output)

        return normalized_outputs



if __name__ == "__main__":
    fm = FuzzyMap(None, np.array([-0.66, -0.33, 0, 0.15, 0.33, 0.45, 0.75]))
    belong = fm.mu(0.9)
    defuz = fm.defuzzify(belong)
    print(belong, defuz)

    nm = Normalization(-30, 30)
    normVal = nm.normalize(0.1)
    print("normalized: ", normVal)
    denormVal = nm.denormalize(normVal)
    print("denormalized: ", denormVal)

