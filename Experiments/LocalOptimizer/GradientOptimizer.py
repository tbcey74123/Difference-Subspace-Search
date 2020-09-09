import sys, os

from .LocalOptimizer import LocalOptimizer

import numpy as np

class GradientOptimizer(LocalOptimizer):
    def __init__(self, n, m, f, g, df, dg, boundary_range, maximizer=True):
        super(GradientOptimizer, self).__init__(n, m, f, g, df, dg, boundary_range, maximizer=maximizer)
        self.name = 'Gradient Optimizer'

    def init(self, init_z):
        super(GradientOptimizer, self).init(init_z)

    def get_direction(self, z, x):
        return self.get_true_gradient(z, x)

    # def sample_direction(self, z):
    #     jacobian = self.jacobian_func(z)
    #     if self.use_svd:
    #
    #         u, s, vh = np.linalg.svd(jacobian, full_matrices=True)
    #
    #         # Importance sampling
    #         choice_p = s / np.sum(s)
    #         idx = np.random.choice(choice_p.shape[0], p=choice_p)
    #         print(np.linalg.norm(vh[idx]), s[idx])
    #
    #         return vh[idx]
    #     else:
    #         grad = np.sum(jacobian, axis=0)
    #         return grad
    #
    # def update(self, t):
    #     self.current_z = self.get_z(t)
    #     self.current_x = self.f(self.current_z.reshape(1, -1))[0]
    #     self.current_score = self.g(self.current_x.reshape(1, -1))[0]
    #
    #     self.direction = self.sample_direction(self.current_z)
    #     # print('Grad length:', np.linalg.norm(self.grad))
