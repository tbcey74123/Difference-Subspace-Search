import numpy as np

class LocalOptimizer():
    def __init__(self, n, m, f, g, df, dg, boundary_range, maximizer=True):
        self.n = n
        self.m = m
        self.f = f
        self.g = g
        self.df = df
        self.dg = dg
        self.boundary_range = boundary_range
        self.maximizer = maximizer

    def init(self, init_z):
        self.current_z = init_z
        self.current_x = self.f(self.current_z.reshape(1, -1))[0]
        self.current_score = self.g(self.current_x.reshape(1, -1))[0]

        print('Initialize', self.name, 'with score', self.current_score )

    def get_direction(self, z, x):
        pass

    def get_true_gradient(self, z, x):
        return np.matmul(self.dg(x), self.df(z))

    def distance_to_boundary(self, z, direction):
        norm_direction = direction / np.linalg.norm(direction)
        target_boundary = np.ones(self.n) * self.boundary_range
        target_boundary[norm_direction < 0] = -self.boundary_range
        perpendicular_difference_to_boundary = target_boundary - z
        division = perpendicular_difference_to_boundary / norm_direction
        distance = np.min(division)

        intersection_point = z + distance * norm_direction

        return distance

    def get_step_size(self, z, direction):
        x = self.f(z.reshape(1, -1))[0]
        score = self.g(x.reshape(1, -1))[0]
        t = 0.5 * np.matmul(self.get_true_gradient(z, x).T, direction / np.linalg.norm(direction))

        step_size = self.distance_to_boundary(z, direction)
        while True:
            new_score = self.g(self.f((z + direction * step_size).reshape(1, -1)).reshape(1, -1))[0]
            if np.abs(new_score - score) < 1e-20:
                return 0
            if new_score - score >= t * step_size:
                break
            else:
                if self.maximizer and new_score - score > 0:
                    break
                elif self.maximizer is False and new_score - score < 0:
                    break
                else:
                    step_size = 0.5 * step_size
        return step_size

    def increment_one_step(self):
        direction = self.get_direction(self.current_z, self.current_x) + 1e-10
        step_size = self.get_step_size(self.current_z, direction)

        self.current_z = self.current_z + direction * step_size
        self.current_x = self.f(self.current_z.reshape(1, -1))[0]
        self.current_score = self.g(self.current_x.reshape(1, -1))[0]

        return step_size
