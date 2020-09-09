import sys, os

from .LocalOptimizer import LocalOptimizer

import numpy as np

class JacobianOptimizer(LocalOptimizer):
    def __init__(self, n, m, f, g, df, dg, boundary_range, maximizer=True):
        super(JacobianOptimizer, self).__init__(n, m, f, g, df, dg, boundary_range, maximizer=maximizer)
        self.name = 'Jacobian Optimizer'

    def init(self, init_z):
        super(JacobianOptimizer, self).init(init_z)

        self.update_jacobian(init_z)

    # def get_direction(self, z, x):
    #     jacobian = self.df(z)
    #     u, s, vh = np.linalg.svd(jacobian, full_matrices=True)
    #
    #     # Importance sampling
    #     choice_p = s / np.sum(s)
    #     idx = np.random.choice(choice_p.shape[0], p=choice_p)
    #     full_s = np.zeros((u.shape[1], vh.shape[0]))
    #     full_s[idx, idx] = s[idx]
    #
    #     reduced_jacobian = np.matmul(np.matmul(u, full_s), vh)
    #
    #     return np.matmul(self.dg(x), reduced_jacobian)

    def update_jacobian(self, z):
        self.jacobian = self.df(z)
        u, s, vh = np.linalg.svd(self.jacobian, full_matrices=True)

        self.jacobian_u = u
        self.jacobian_vhs = vh
        self.jacobian_s = s
        self.jacobian_mask = np.ones(self.jacobian_s.shape[0], dtype=bool)

    def get_direction(self, z, x):
        if self.jacobian_s[self.jacobian_mask].shape[0] <= 0:
            self.jacobian_mask = np.ones(self.jacobian_vhs.shape[0], dtype=bool)

        s = self.jacobian_s[self.jacobian_mask] + 1e-6

        # Importance sampling
        choice_p = s / np.sum(s)
        idx = np.random.choice(choice_p.shape[0], p=choice_p)

        orig_idx = np.arange(self.jacobian_vhs.shape[0])[self.jacobian_mask][idx]

        full_s = np.zeros((self.m, self.n))
        full_s[orig_idx, orig_idx] = s[idx]

        reduced_jacobian = np.matmul(np.matmul(self.jacobian_u, full_s), self.jacobian_vhs)
        gradient = -np.sum(reduced_jacobian, axis=0)

        return gradient

    def increment_one_step(self):
        step_size = super(JacobianOptimizer, self).increment_one_step()
        if step_size != 0.0:
            self.update_jacobian(self.current_z)
