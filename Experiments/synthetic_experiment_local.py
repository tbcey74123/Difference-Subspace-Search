# python synthetic_experiment_local.py [--test]

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import time
import numpy as np

from LocalOptimizer.GradientOptimizer import GradientOptimizer
from LocalOptimizer.JacobianOptimizer import JacobianOptimizer

from FunctionDefinition import myFunc, myJacobian, myGradient, myGoodness, getSliderLength, generateRandomizedAnisotropicMatrix
from utils.utils import safe_mkdir

ns = [8, 32, 128, 512]
m = 1024
alpha_range = (0.01, 10.0)
gammas = [0]
range_per_axis = 2.5
case_n = 10
iteration_n = 500

# ------------------------------------------------------------------------------
# If this script runs with the "--test" option, change the numbers
if len(sys.argv) >= 2 and "test" in sys.argv[1]:
    print("Test run")
    ns = [8, 32]
    case_n = 2
    iteration_n = 20
# ------------------------------------------------------------------------------

for gamma in gammas:
    for n in ns:
        output_base_path = 'synthetic_experiment_local' + '/' + str(gamma) + '/' + str(n) + '_' + str(m)
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

            init_z_filepath = output_case_path + '/init_z.txt'
            if os.path.isfile(init_z_filepath):
                f = open(init_z_filepath, 'r')
                _ = f.readline()
                data = f.readline().split(' ')
                init_z = np.array(list(map(float, data))).reshape(n)
                f.close()
            else:
                init_z = np.random.uniform(low=-range_per_axis, high=range_per_axis, size=(n))

                f = open(init_z_filepath, 'w')
                f.write(str(n) + "\n")
                for i in range(n):
                    f.write(str(init_z[i]))
                    if i == n - 1:
                        f.write("\n")
                    else:
                        f.write(" ")
                f.close()

            opts = [
                GradientOptimizer(n, m, lambda zs: myFunc(zs, ATAs, f_shift), lambda xs: myGoodness(xs, alphas, gamma), lambda z: myJacobian(z, ATAs, f_shift), lambda xs: -2 * xs, range_per_axis),
                JacobianOptimizer(n, m, lambda zs: myFunc(zs, ATAs, f_shift), lambda xs: myGoodness(xs, alphas, gamma), lambda z: myJacobian(z, ATAs, f_shift), lambda xs: -2 * xs, range_per_axis)
            ]

            output_names = [
                'Gradient.txt',
                'Jacobian_1d.txt'
            ]

            output_results_path = output_case_path + '/results'
            safe_mkdir(output_results_path)

            for i in range(len(opts)):
                opt = opts[i]
                opt.init(init_z)
                current_score = opt.current_score

                opt.get_true_gradient(opt.current_z, opt.current_x)

                filename = output_results_path + '/' + output_names[i]
                f = open(filename, 'w')
                f.write(str(iteration_n) + "\n")
                f.flush()

                line = str(current_score) + " " + str(0.0) + "\n"
                f.write(line)
                f.flush()

                start_time = time.time()
                for j in range(iteration_n):
                    opt.increment_one_step()
                    current_score = opt.current_score

                    print('Iteration #' + str(j) + ': ' + str(current_score))

                    end_time = time.time()
                    line = str(current_score) + " " + str(end_time - start_time) + "\n"
                    f.write(line)
                    f.flush()
                f.close()
