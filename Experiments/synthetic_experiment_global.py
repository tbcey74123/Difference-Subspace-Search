# python synthetic_experiment_global.py [--test]

import os, sys

import time
import numpy as np

from GlobalOptimizer.SLSOptimizer import SLSOptimizer
from GlobalOptimizer.JacobianOptimizer import JacobianOptimizer
from GlobalOptimizer.JacobianOptimizerLocalLineSearch import JacobianOptimizerLocalLineSearch
from GlobalOptimizer.HybridOptimizer import HybridOptimizer
from GlobalOptimizer.RandomOptimizer import RandomOptimizer

from FunctionDefinition import myFunc, myJacobian, myGoodness, getSliderLength, generateRandomizedAnisotropicMatrix
from utils.utils import safe_mkdir, getRandomAMatrix

low_n = 6
ns = [8, 32, 128, 512]
m = 1024
alpha_range = (0.01, 10.0)
gammas = [2.0]
range_per_axis = 2.5
case_n = 10
iteration_ns = [100, 200, 200, 200, 200, 200] # First one for SLS-BO
hybrid_iteration_n = 10
sample_n = 10001

# ------------------------------------------------------------------------------
# If this script runs with the "--test" option, change the numbers
if len(sys.argv) >= 2 and "test" in sys.argv[1]:
    print("Test run")
    ns = [8]
    iteration_ns = [4, 4, 4, 4, 4, 4]
    case_n = 2
# ------------------------------------------------------------------------------

for gamma in gammas:
    for n in ns:
        slider_length = getSliderLength(n, range_per_axis, 0.2)
        output_base_path = 'synthetic_experiment_global' + '/' + str(gamma) + '/' + str(n) + '_' + str(m)
        safe_mkdir(output_base_path)

        for case_idx in range(case_n):
            print('------------- Start case #' + str(case_idx) + ' -------------')
            output_case_path = output_base_path + '/' + str(case_idx)
            safe_mkdir(output_case_path)

            alphas_filepath = output_case_path + '/alphas.txt'
            if os.path.isfile(alphas_filepath):
                f = open(alphas_filepath, 'r')
                _ = f.readline()
                data = f.readline().split(' ')
                alphas = np.array(list(map(float, data))).reshape(m)
                f.close()
            else:
                alphas = np.random.rand(m) * (alpha_range[1] - alpha_range[0]) + alpha_range[0]
                f = open(alphas_filepath, 'w')
                f.write(str(m) + "\n")
                for i in range(m):
                    f.write(str(alphas[i]))
                    if i == m - 1:
                        f.write("\n")
                    else:
                        f.write(" ")
                f.close()

            matrix_A_filepath = output_case_path + '/As.txt'
            if os.path.isfile(matrix_A_filepath):
                f = open(matrix_A_filepath, 'r')
                _ = f.readline()
                data = f.readline().split(' ')
                ATAs = np.array(list(map(float, data))).reshape(m, n, n)
                f.close()
            else:
                ATAs = np.zeros((m, n, n))
                for d in range(m):
                    ATAs[d] = generateRandomizedAnisotropicMatrix(n=n, alphas=alphas[d])

                f = open(matrix_A_filepath, 'w')
                f.write(str(m) + " " + str(n) + " " + str(n) + "\n")
                for d in range(m):
                    for row in range(n):
                        for col in range(n):
                            f.write(str(ATAs[d, row, col]))
                            if d == m - 1 and row == n - 1 and col == n - 1:
                                f.write("\n")
                            else:
                                f.write(" ")
                f.close()

            f_shift_filepath = output_case_path + '/f_shift.txt'
            if os.path.isfile(f_shift_filepath):
                f = open(f_shift_filepath, 'r')
                _ = f.readline()
                data = f.readline().split(' ')
                f_shift = np.array(list(map(float, data))).reshape(n)
                f.close()
                optimal_z = f_shift
            else:
                f_shift = np.random.uniform(low=-range_per_axis / 2, high=range_per_axis / 2, size=(n))
                f = open(f_shift_filepath, 'w')
                f.write(str(n) + "\n")
                for i in range(n):
                    f.write(str(f_shift[i]))
                    if i == n - 1:
                        f.write("\n")
                    else:
                        f.write(" ")
                f.close()

            random_A_filepath = output_case_path + '/random_A.txt'
            if os.path.isfile(random_A_filepath):
                f = open(random_A_filepath, 'r')
                _ = f.readline()
                data = f.readline().split(' ')
                random_A = np.array(list(map(float, data))).reshape(n, low_n)
                f.close()
            else:
                while True:
                    random_A = getRandomAMatrix(n, low_n, f_shift.reshape(1, -1), range_per_axis)
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

            opts = [
                SLSOptimizer(n, m, lambda zs: myFunc(zs, ATAs, f_shift), lambda xs: myGoodness(xs, alphas, gamma), range_per_axis, use_MAP=True),
                SLSOptimizer(n, m, lambda zs: myFunc(zs, ATAs, f_shift), lambda xs: myGoodness(xs, alphas, gamma), range_per_axis, use_MAP=True, random_A=random_A),
                JacobianOptimizer(n, m, lambda zs: myFunc(zs, ATAs, f_shift), lambda xs: myGoodness(xs, alphas, gamma), slider_length, lambda z: myJacobian(z, ATAs, f_shift)),
                HybridOptimizer(n, m, lambda zs: myFunc(zs, ATAs, f_shift), lambda xs: myGoodness(xs, alphas, gamma), slider_length, lambda z: myJacobian(z, ATAs, f_shift), low_n, hybrid_iteration_n, use_MAP=True),
                JacobianOptimizerLocalLineSearch(n, m, lambda zs: myFunc(zs, ATAs, f_shift), lambda xs: myGoodness(xs, alphas, gamma),
                                  slider_length, lambda z: myJacobian(z, ATAs, f_shift)),
                RandomOptimizer(n, m, lambda zs: myFunc(zs, ATAs, f_shift), lambda xs: myGoodness(xs, alphas, gamma),
                                  slider_length),
            ]

            output_names = [
                'SLS_w_MAP.txt',
                'SLS_w_MAP_REMBO.txt',
                'Jacobian_1d.txt',
                'Hybrid_6d.txt',
                'Jocobian_1d_local.txt',
                'Random.txt'
            ]

            output_results_path = output_case_path + '/results'
            safe_mkdir(output_results_path)

            for i in range(len(opts)):
                iteration_n = iteration_ns[i]
                opt = opts[i]
                opt.init(init_z)
                best_score = opt.current_score

                filename = output_results_path + '/' + output_names[i]
                f = open(filename, 'w')
                f.write(str(iteration_n) + "\n")
                f.flush()

                line = str(best_score) + " " + str(0.0) + "\n"
                f.write(line)
                f.flush()

                opt_t = -1
                start_time = time.time()
                for j in range(iteration_n):
                    opt_z, opt_x, opt_score, opt_t = opt.find_optimal(sample_n, batch_size=500)
                    if opt_score > best_score:
                        best_score = opt_score

                    print('Iteration #' + str(j) + ': ' + str(best_score))

                    end_time = time.time()
                    line = str(best_score) + " " + str(end_time - start_time) + "\n"
                    f.write(line)
                    f.flush()

                    opt.update(opt_t)
                f.close()
