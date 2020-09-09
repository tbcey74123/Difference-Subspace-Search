import sys, os
sys.path.append(os.path.abspath('../utils'))

from .GlobalOptimizer import GlobalOptimizer

import numpy as np
import functools
from utils.utils import generateSliderWithCenter
from utils import pySequentialLineSearch as pySLS

class HybridOptimizer(GlobalOptimizer):
    def __init__(self, n, m, f, g, search_range, jacobian_func, low_n, subspace_update_n, use_MAP=False, maximizer=True):
        super(HybridOptimizer, self).__init__(n, m, f, g, search_range, maximizer=maximizer)
        self.name = 'Hybrid Optimizer'
        self.jacobian_func = jacobian_func
        self.low_n = low_n
        self.subspace_update_n = subspace_update_n
        self.use_MAP = use_MAP
        self.kernel = pySLS.KernelType.ArdSquaredExponentialKernel

    def init(self, init_z):
        super(HybridOptimizer, self).init(init_z)

        self.update_subspace(self.current_z)
        self.counter = 0

    def get_z(self, t):
        local_z = (np.array(self.SLS.calc_point_from_slider_position(t)) - 0.5) * self.search_range * 2
        return self.current_z + np.dot(self.subspace_basis.T, local_z)

    def update_subspace(self, z):
        self.update_jacobian(z)

        init = np.ones(self.low_n) * 0.5
        self.SLS = pySLS.SequentialLineSearchOptimizer(self.low_n, False, self.use_MAP, self.kernel, initial_slider_generator=functools.partial(generateSliderWithCenter, center=init))

    def update_jacobian(self, z):
        self.jacobian = self.jacobian_func(z)
        u, s, vh = np.linalg.svd(self.jacobian, full_matrices=True)

        # Importance sampling
        choice_p = s / np.sum(s)
        idx = np.random.choice(choice_p.shape[0], self.low_n, replace=False, p=choice_p)

        self.subspace_basis = vh[idx]

    def update(self, t):
        self.counter += 1
        if self.counter == self.subspace_update_n:
            self.counter = 0

            self.current_z = self.get_z(t)
            self.current_x = self.f(self.current_z.reshape(1, -1))[0]
            self.current_score = self.g(self.current_x.reshape(1, -1))[0]

            self.update_subspace(self.current_z)
        else:
            self.SLS.submit_line_search_result(t)
