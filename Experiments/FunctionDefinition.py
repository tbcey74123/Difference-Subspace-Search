import numpy as np

def getSliderLength(n, boundary_range, ratio, sample_num=1000):
    samples = np.random.uniform(low=-boundary_range, high=boundary_range, size=(sample_num, 2, n))
    distances = np.linalg.norm(samples[:, 0, :] - samples[:, 1, :], axis=1)
    average_distance = np.average(distances)
    return ratio * average_distance

def generateRandomizedAnisotropicMatrix(n, alphas):
    """
        This function returns A^T A, an n-by-n matrix creating anisotropy
        """
    a = np.zeros(shape=(n, n))
    a[np.arange(n), np.arange(n)] = (np.random.rand(n) * 0.99 + 0.01) * alphas

    random_rotation = np.random.rand(n, n)
    random_rotation, _ = np.linalg.qr(random_rotation)

    a = np.matmul(np.matmul(random_rotation.T, a), random_rotation)

    return a

# Complex Anisotropic Sphere function
# zs: (-1, n)
# As: (m, n ,n)
# return (-1, m)
def myFunc(zs, ATAs, shift):
    shifted_zs = zs - shift
    xs = np.tensordot(shifted_zs, ATAs, axes=([1], [1]))
    xs = (xs.transpose(1, 0, 2) * shifted_zs).transpose(1, 0, 2)
    xs = np.sum(xs, axis=2)
    xs = np.sqrt(xs)
    return xs

def myJacobian(z, ATAs, shift):
    z = z - shift
    jacobian = np.tensordot(z, ATAs, axes=([0], [1])) + np.tensordot(z, ATAs, axes=([0], [2]))
    return jacobian

def myGradient(z, ATAs, shift):
    jacobian = myJacobian(z, ATAs, shift)
    grad = np.sum(jacobian, axis=0)
    grad = grad / np.linalg.norm(grad)
    return grad

# (-1, m) -> (-1)
def myGoodness(xs, alphas, gamma):
    a = gamma * alphas ** 2
    ys = -np.sum(a) - np.sum(xs ** 2 - a * np.cos(2 * np.pi * xs), axis=1)
    return ys