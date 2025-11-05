import numpy as np


class FuzzyCMeans:
    def __init__(self, input_data:np.array, c, m=2, A=None):
        """
        :param c: Number of clusters in the C-Means
        :param m: fuzzy weightage parameter. Default is 2
        :param A: Weighted distance matrix.
        """
        if A is None:
            A = np.identity(input_data.shape[1])

        self.A = A
        self.m = m
        self.input_data = input_data
        self.train()

    def train(self):
        pass

    def test(self):
        pass

    def optimize_c_value(self):
        pass



