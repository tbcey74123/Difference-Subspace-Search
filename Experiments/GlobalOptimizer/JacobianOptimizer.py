import sys, os

from .GlobalOptimizer import GlobalOptimizer

import numpy as np

class JacobianOptimizer(GlobalOptimizer):
    def __init__(self, n, m, f, g, search_range, jacobian_func, maximizer=True):
        super(JacobianOptimizer, self).__init__(n, m, f, g, search_range, maximizer=maximizer)
        self.name = 'Jacobian Optimizer'
        self.jacobian_func = jacobian_func
        self.center_tolerance = 0.05

    def init(self, init_z):
        super(JacobianOptimizer, self).init(init_z)

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
