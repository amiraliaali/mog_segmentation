import numpy as np
from scipy.stats import multivariate_normal

class MoG:
    def __init__(self, table_of_data, num_components=2, tol=1e-4, max_iter=100):
        self.table_of_data = table_of_data
        self.num_components = num_components
        self.tol = tol
        self.max_iter = max_iter

        self.means = {}
        self.covs = {}
        self.weights = {}
        
        for k in range(num_components):
            m, cov, prob = self.compute_gaussian_components(k)
            self.means[k] = np.array(m)
            self.covs[k] = np.array(cov)
            self.weights[k] = prob

    def compute_gaussian_components(self, component_num):
        filtered_table = self.table_of_data[self.table_of_data["assigned_component"] == component_num]
        mean_of_gaussian = np.mean(filtered_table[["r", "g", "b"]].values, axis=0)
        cov_of_gaussian = np.cov(filtered_table[["r", "g", "b"]].values, rowvar=False)
        prob_of_gaussian = len(filtered_table) / len(self.table_of_data)
        return mean_of_gaussian, cov_of_gaussian, prob_of_gaussian

    def gaussian_pdf(self, x, mean, cov):
        return multivariate_normal.pdf(x, mean=mean, cov=cov)

    def expectation_step(self):
        data = self.table_of_data[["r", "g", "b"]].values
        responsibilities = np.zeros((len(data), self.num_components))

        for k in range(self.num_components):
            responsibilities[:, k] = self.weights[k] * self.gaussian_pdf(data, self.means[k], self.covs[k])

        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities

    def maximization_step(self, responsibilities):
        data = self.table_of_data[["r", "g", "b"]].values
        for k in range(self.num_components):
            resp_k = responsibilities[:, k]
            total_resp = resp_k.sum()

            self.means[k] = np.sum(resp_k[:, np.newaxis] * data, axis=0) / total_resp
            self.covs[k] = np.cov(data.T, aweights=resp_k, bias=True)
            self.weights[k] = total_resp / len(data)

    def run(self):
        prev_means = np.copy(list(self.means.values()))
        for i in range(self.max_iter):
            responsibilities = self.expectation_step()
            self.maximization_step(responsibilities)

            mean_shift = np.sum([np.linalg.norm(self.means[k] - prev_means[k]) for k in range(self.num_components)])
            if mean_shift < self.tol:
                break
            prev_means = np.copy(list(self.means.values()))

        return self.means, self.covs, self.weights
