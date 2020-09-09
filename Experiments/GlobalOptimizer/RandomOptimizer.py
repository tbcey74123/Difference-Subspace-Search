import sys, os

from .GlobalOptimizer import GlobalOptimizer

import numpy as np

class RandomOptimizer(GlobalOptimizer):
    def __init__(self, n, m, f, g, search_range, maximizer=True):
        super(RandomOptimizer, self).__init__(n, m, f, g, search_range, maximizer=maximizer)
        self.name = 'Random Optimizer'

    def init(self, init_z):
        super(RandomOptimizer, self).init(init_z)

        self.sample_direction()

    def get_z(self, t):
        return self.current_z + self.subspace_basis * (t - 0.5) * self.search_range * 2

    def sample_direction(self):
        self.subspace_basis = np.random.rand(self.n)
        self.subspace_basis /= np.linalg.norm(self.subspace_basis)

    def update(self, t):
        self.current_z = self.get_z(t)
        self.current_x = self.f(self.current_z.reshape(1, -1))[0]
        self.current_score = self.g(self.current_x.reshape(1, -1))[0]

        self.sample_direction()
