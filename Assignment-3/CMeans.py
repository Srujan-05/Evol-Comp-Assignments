import numpy as np


class FuzzyCMeans:
    def __init__(self, input_data:np.array, c=None, m=2, A=None, epsilon=1e-3):
        """
        :param input_data: training data of size Nxn where N is number of samples and n is dimension
                            of each sample vector.
        :param c: Number of clusters in the C-Means
        :param m: fuzzy weightage parameter. Default is 2
        :param A: Weighted distance matrix.
        :param epsilon: Threshold for training.
        """
        if A is None:
            A = np.identity(input_data.shape[1])

        self.c = c
        self.A = A
        self.m = m
        self.training_data = input_data
        self.epsilon = epsilon
        if c is None:
            optimal_c = self.optimize_c_value()
            self.cluster_centroids, self.obj_func, self.centroid_belongingness = self.train(optimal_c)
        else:
            self.cluster_centroids, self.obj_func, self.centroid_belongingness = self.train(self.c)

    def train(self, c):
        U_prev, U_curr = None, None
        U_curr = self.initialize_fuzzy_partition_matrix(c)
        centroids = self.compute_centroids(U_curr)
        squared_distances = self.calculate_distances(self.training_data, centroids)
        while U_prev is None or np.linalg.norm(U_prev - U_curr) >= self.epsilon:
            U_prev = U_curr
            U_curr = self.compute_fuzzy_partition_matrix(squared_distances)
            centroids = self.compute_centroids(U_curr)
            squared_distances = self.calculate_distances(self.training_data, centroids)

        objective_val = np.sum(np.multiply(U_curr**self.m, squared_distances))
        centroid_belongingness = [np.argmax(U_curr[:, i]) for i in range(self.training_data.shape[0])]
        return centroids, objective_val, centroid_belongingness

    def test(self, test_data):
        squared_distances = self.calculate_distances(test_data, self.cluster_centroids)
        U = self.compute_fuzzy_partition_matrix(squared_distances)
        objective_val = np.sum(np.multiply(U ** self.m, squared_distances))
        centroid_belongingness = [np.argmax(U[:, i]) for i in range(test_data.shape[0])]
        return centroid_belongingness, objective_val

    def calculate_distances(self, data, centroids):
        squared_distances = []
        for centroid in centroids:
            squared_distances.append(np.sum(np.multiply((data-centroid) @ self.A, data-centroid), axis=1))
        return np.array(squared_distances)

    def compute_centroids(self, U):
        return np.divide(np.dot(U**self.m, self.training_data), np.sum(U**self.m, axis=1)[:, None])

    def initialize_fuzzy_partition_matrix(self, c):
        N = self.training_data.shape[0]
        U = np.zeros((c, N))
        for i in range(c):
            U[i][int((i/c)*N):int(((i+1)/c)*N)] += 1
        return U

    def compute_fuzzy_partition_matrix(self, squared_distances):
        U = np.zeros((squared_distances.shape[0], squared_distances.shape[1]))
        for i in range(squared_distances.shape[1]):
            flag = 0
            for j in range(squared_distances.shape[0]):
                if squared_distances[j][i] == 0:
                    U[:, i] = 0
                    U[j][i] = 1
                    flag = 1
                    break
            if flag != 1:
                for j in range(squared_distances.shape[0]):
                    U[j][i] = 1/((np.sum(squared_distances[j][i]/squared_distances[:, i]))**(1/(self.m-1)))
        return U


    def optimize_c_value(self, c_min=2, c_max=10) -> int:
        J = {}
        for k in range(c_min, c_max + 1):
            centroids, obj_val, _ = self.train(k)
            J[k] = obj_val

        ratios = {}
        for k in range(c_min + 1, c_max):
            num = abs(J[k] - J[k + 1])
            den = abs(J[k - 1] - J[k])
            ratios[k] = num / den if den != 0 else np.inf

        optimal_c = min(ratios, key=ratios.get)
        return optimal_c


if __name__ == "__main__":
    cmeans = FuzzyCMeans(np.array([[1, 2], [2, 3], [2, 4], [5, 6], [6, 7]]), 3)
    cb, _ = cmeans.test(np.array([[1, 0], [4, 5], [5, 4], [6, 10]]))
    c = cmeans.optimize_c_value()
    print(c)
