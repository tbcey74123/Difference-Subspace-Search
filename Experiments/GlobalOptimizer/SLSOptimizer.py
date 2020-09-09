import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")

from .GlobalOptimizer import GlobalOptimizer

import numpy as np
import functools
from utils.utils import generateSliderWithCenter
from utils import pySequentialLineSearch as pySLS

class SLSOptimizer(GlobalOptimizer):
    def __init__(self, n, m, f, g, search_range, use_MAP=False, random_A=None, maximizer=True):
        super(SLSOptimizer, self).__init__(n, m, f, g, search_range, maximizer=maximizer)
        self.name = 'SLS Optimizer'
        self.use_MAP = use_MAP
        self.use_REMBO = False
        self.kernel = pySLS.KernelType.ArdSquaredExponentialKernel
        if random_A is not None:
            self.use_REMBO = True
            self.low_n = random_A.shape[1]
            self.random_A =random_A

    def init(self, init_z):
        super(SLSOptimizer, self).init(init_z)

        if self.use_REMBO:
            low_z = np.matmul(np.linalg.pinv(self.random_A), self.current_z)
            init = low_z / (self.search_range * 2) + 0.5
            self.SLS = pySLS.SequentialLineSearchOptimizer(self.low_n, False, self.use_MAP, self.kernel, initial_slider_generator=functools.partial(generateSliderWithCenter, center=init))
        else:
            init = self.current_z / (self.search_range * 2) + 0.5
            self.SLS = pySLS.SequentialLineSearchOptimizer(self.n, False, self.use_MAP, self.kernel, initial_slider_generator=functools.partial(generateSliderWithCenter, center=init))

    def get_z(self, t):
        if self.use_REMBO:
            return np.matmul(self.random_A, (np.array(self.SLS.calc_point_from_slider_position(t)) - 0.5) * self.search_range * 2)
        else:
            return (np.array(self.SLS.calc_point_from_slider_position(t)) - 0.5) * self.search_range * 2

    def update(self, t):
        self.SLS.submit_line_search_result(t)
