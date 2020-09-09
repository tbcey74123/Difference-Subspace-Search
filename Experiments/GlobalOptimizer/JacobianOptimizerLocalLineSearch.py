import sys, os

from .GlobalOptimizer import GlobalOptimizer

import numpy as np

class JacobianOptimizerLocalLineSearch(GlobalOptimizer):
    def __init__(self, n, m, f, g, search_range, jacobian_func, maximizer=True):
        super(JacobianOptimizerLocalLineSearch, self).__init__(n, m, f, g, search_range, maximizer=maximizer)
        self.name = 'Jacobian Optimizer (Local Line Search)'
        self.jacobian_func = jacobian_func
        self.center_tolerance = 0.05

    def init(self, init_z):
        super(JacobianOptimizerLocalLineSearch, self).init(init_z)

        self.update_jacobian(self.current_z)
        self.sample_direction()

    def get_z(self, t):
        return self.current_z + self.subspace_basis * (t - 0.5) * self.search_range * 2

    def update_jacobian(self, z):
        self.jacobian = self.jacobian_func(z)
        u, s, vh = np.linalg.svd(self.jacobian, full_matrices=True)

        num = s.shape[0]
        self.jacobian_vhs = vh[:num]
        self.jacobian_s = s[:num]
        self.jacobian_mask = np.ones(self.jacobian_vhs.shape[0], dtype=bool)

    def sample_direction(self):
        if self.jacobian_s[self.jacobian_mask].shape[0] <= 0:
            self.jacobian_mask = np.ones(self.jacobian_vhs.shape[0], dtype=bool)

        s = self.jacobian_s[self.jacobian_mask] + 1e-6
        vh = self.jacobian_vhs[self.jacobian_mask]

        # Importance sampling
        choice_p = s / np.sum(s)
        idx = np.random.choice(choice_p.shape[0], p=choice_p)

        self.subspace_basis = vh[idx]

        self.jacobian_mask[np.arange(self.jacobian_vhs.shape[0])[self.jacobian_mask][idx]] = False

    def update(self, t):
        self.current_z = self.get_z(t)
        self.current_x = self.f(self.current_z.reshape(1, -1))[0]
        self.current_score = self.g(self.current_x.reshape(1, -1))[0]

        if np.abs(t - 0.5) > self.center_tolerance:
            self.update_jacobian(self.current_z)
        self.sample_direction()

    def binary_search(self, scores, left_idx, right_idx, left_higher):
        center_idx = (left_idx + right_idx) // 2
        if center_idx == left_idx:
            return left_idx
        left_score = scores[left_idx]
        right_score = scores[right_idx]
        center_score = scores[center_idx]

        if left_higher:
            if center_score > left_score:
                return self.binary_search(scores, left_idx, center_idx, False)
            else:
                return left_idx
        else:
            if center_score > right_score:
                return self.binary_search(scores, center_idx, right_idx, True)
            else:
                return right_idx

    def find_optimal(self, sample_n, batch_size=100):
        if batch_size <= 0:
            batch_size = sample_n

        zs = []
        for i in range(sample_n):
            t = i / float(sample_n - 1)
            z = self.get_z(t)
            zs.append(z)
        zs = np.array(zs)

        batch_n = sample_n // batch_size
        remainder = sample_n - batch_size * batch_n

        xs = np.zeros((sample_n, self.m))
        for i in range(batch_n):
            xs[i * batch_size:(i + 1) * batch_size] = self.f(zs[i * batch_size:(i + 1) * batch_size])
        if remainder != 0:
            xs[batch_n * batch_size:] = self.f(zs[batch_n * batch_size:])

        scores = self.g(xs)
        if self.maximizer:
            diff = scores - self.current_score
            # print(diff)
            left_idx = 0
            neg_idx = np.arange(sample_n // 2)[diff[:sample_n // 2] < 0]
            if neg_idx.shape[0] != 0:
                left_idx = np.max(neg_idx)
            left_idx += 2
            right_idx = sample_n - 1
            neg_idx = np.arange(sample_n // 2, sample_n)[diff[sample_n // 2:] < 0]
            if neg_idx.shape[0] != 0:
                right_idx = np.min(neg_idx)
            right_idx -= 1
            # print(left_idx, diff[left_idx], right_idx, diff[right_idx])

            center_idx = self.binary_search(scores, left_idx, right_idx, scores[left_idx] > scores[right_idx])
            # print(center_idx, diff[center_idx])

            idx = center_idx
        else:
            idx = np.argmin(scores)
        t = idx / float(sample_n - 1)

        z = self.get_z(t)
        x = self.f(z.reshape(1, -1))[0]
        score = self.g(x.reshape(1, -1))[0]

        return z, x, score, t
