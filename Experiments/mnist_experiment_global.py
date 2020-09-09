# python mnist_experiment_global.py [--test]

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import time
import numpy as np
import cv2

from models.MNISTGenerator import MNISTGenerator

from GlobalOptimizer.SLSOptimizer import SLSOptimizer
from GlobalOptimizer.JacobianOptimizer import JacobianOptimizer
from GlobalOptimizer.JacobianOptimizerLocalLineSearch import JacobianOptimizerLocalLineSearch
from GlobalOptimizer.HybridOptimizer import HybridOptimizer
from GlobalOptimizer.RandomOptimizer import RandomOptimizer

from FunctionDefinition import getSliderLength, generateRandomizedAnisotropicMatrix
from utils.utils import safe_mkdir, getRandomAMatrix

def myFunc(model, zs):
    return model.decode(zs).reshape(zs.shape[0], -1)

def myGoodness(target, xs):
    return np.sum((xs.reshape(xs.shape[0], -1) - target.reshape(1, -1)) ** 2, axis=1) ** 0.5

def myJacobian(model, z):
    return model.calc_model_gradient(z)

n = 64
m = 28 * 28
low_n = 6
range_per_axis = 1
case_n = 10
iteration_ns = [100, 200, 200, 200, 200, 100] # First one for SLS-BO
hybrid_iteration_n = 10
sample_n = 10001

# ------------------------------------------------------------------------------
# If this script runs with the "--test" option, change the numbers
if len(sys.argv) >= 2 and "test" in sys.argv[1]:
    print("Test run")
    iteration_ns = [4, 4, 4, 4, 4, 4]
    case_n = 2
# ------------------------------------------------------------------------------

model = MNISTGenerator()
weights_path = '../pretrained_weights/MNIST/model.ckpt-155421'
model.load_model(weights_path)

slider_length = getSliderLength(n, range_per_axis, 0.2)
output_base_path = 'mnist_experiment_global/0/64_1024'
safe_mkdir(output_base_path)

for case_idx in range(case_n):
    print('------------- Start case #' + str(case_idx) + ' -------------')
    output_case_path = output_base_path + '/' + str(case_idx)
    safe_mkdir(output_case_path)

    target_latent_filepath = output_case_path + '/target_latent.txt'
    if os.path.isfile(target_latent_filepath):
        f = open(target_latent_filepath, 'r')
        _ = f.readline()
        data = f.readline().split(' ')
        target_latent = np.array(list(map(float, data))).reshape(n)
        f.close()
    else:
        target_latent = np.random.uniform(-1, 1, 64)
        f = open(target_latent_filepath, 'w')
        f.write(str(n) + "\n")
        for i in range(n):
            f.write(str(target_latent[i]))
            if i == n - 1:
                f.write("\n")
            else:
                f.write(" ")
        f.close()
    target_data = model.decode(target_latent.reshape(1, -1))[0]
    cv2.imwrite(output_case_path + "/target.png", target_data)

    random_A_filepath = output_case_path + '/random_A.txt'
    if os.path.isfile(random_A_filepath):
        f = open(random_A_filepath, 'r')
        _ = f.readline()
        data = f.readline().split(' ')
        random_A = np.array(list(map(float, data))).reshape(n, low_n)
        f.close()
    else:
        while True:
            random_A = getRandomAMatrix(n, low_n, target_latent.reshape(1, -1), range_per_axis)
            if random_A is not None:
                break
            else:
                print("Failed! Get Random matrix again...")
        f = open(random_A_filepath, 'w')
        f.write(str(low_n) + " " + str(n) + "\n")
        for row in range(n):
            for col in range(low_n):
                f.write(str(random_A[row, col]))
                if row == n - 1 and col == low_n - 1:
                    f.write("\n")
                else:
                    f.write(" ")
        f.close()

    init_low_z_filepath = output_case_path + '/init_low_z.txt'
    if os.path.isfile(init_low_z_filepath):
        f = open(init_low_z_filepath, 'r')
        _ = f.readline()
        data = f.readline().split(' ')
        init_low_z = np.array(list(map(float, data))).reshape(low_n)
        f.close()
    else:
        init_z = np.random.uniform(low=-range_per_axis, high=range_per_axis, size=(n))
        init_low_z = np.matmul(np.linalg.pinv(random_A), init_z.T).T

        f = open(init_low_z_filepath, 'w')
        f.write(str(low_n) + "\n")
        for i in range(low_n):
            f.write(str(init_low_z[i]))
            if i == low_n - 1:
                f.write("\n")
            else:
                f.write(" ")
        f.close()

    init_z = np.matmul(random_A, init_low_z)

    init_data = model.decode(init_z.reshape(1, -1))[0]
    cv2.imwrite(output_case_path + "/init.png", init_data)

    opts = [
        SLSOptimizer(n, m, lambda zs: myFunc(model, zs), lambda xs: myGoodness(target_data, xs), range_per_axis, use_MAP=True, maximizer=False),
        SLSOptimizer(n, m, lambda zs: myFunc(model, zs), lambda xs: myGoodness(target_data, xs), range_per_axis, use_MAP=True, random_A=random_A, maximizer=False),
        JacobianOptimizer(n, m, lambda zs: myFunc(model, zs), lambda xs: myGoodness(target_data, xs), slider_length, lambda z: myJacobian(model, z), maximizer=False),
        HybridOptimizer(n, m, lambda zs: myFunc(model, zs), lambda xs: myGoodness(target_data, xs), slider_length, lambda z: myJacobian(model, z), low_n, hybrid_iteration_n, use_MAP=True, maximizer=False),
        JacobianOptimizerLocalLineSearch(n, m, lambda zs: myFunc(model, zs), lambda xs: myGoodness(target_data, xs), slider_length, lambda z: myJacobian(model, z), maximizer=False),
        RandomOptimizer(n, m, lambda zs: myFunc(model, zs), lambda xs: myGoodness(target_data, xs), slider_length, maximizer=False),
    ]

    output_names = [
        'SLS_w_MAP',
        'SLS_w_MAP_REMBO',
        'Jacobian_1d',
        'Hybrid_6d',
        'Jocobian_1d_local',
        'Random'
    ]

    output_results_path = output_case_path + '/results'
    safe_mkdir(output_results_path)

    for i in range(len(opts)):
        iteration_n = iteration_ns[i]
        opt = opts[i]
        opt.init(init_z)
        best_score = opt.current_score

        filename = output_results_path + '/' + output_names[i] + '.txt'
        f = open(filename, 'w')
        f.write(str(iteration_n) + "\n")
        f.flush()

        line = str(best_score) + " " + str(0.0) + "\n"
        f.write(line)
        f.flush()

        opt_t = -1
        start_time = time.time()
        for j in range(iteration_n):
            opt_z, opt_x, opt_score, opt_t = opt.find_optimal(sample_n, batch_size=sample_n)
            if opt_score < best_score:
                best_score = opt_score

            print('Iteration #' + str(j) + ': ' + str(best_score))

            end_time = time.time()
            line = str(best_score) + " " + str(end_time - start_time) + "\n"
            f.write(line)
            f.flush()

            opt.update(opt_t)
        f.close()

        opt_data = opt_x.reshape(28, 28)
        cv2.imwrite(output_results_path + "/" + output_names[i] + ".png", opt_data)
