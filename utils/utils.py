import os
from OpenGL.GL import *
import ctypes
import numpy as np

def VecF(args):
    """Simple function to create ctypes arrays of floats"""
    return (GLfloat * len(args))(*args)

def safe_mkdir(path):
    if os.path.isdir(path):
        return
    os.makedirs(path)

def readSearchCases(filename):
    fr = open(filename, 'r')
    case_n = int(fr.readline())
    init_range = float(fr.readline())
    latent_size = int(fr.readline())

    target_latents = []
    init_latents = []
    for i in range(case_n):
        target_latent = []
        line = fr.readline()
        latent = line.split(' ')
        for j in range(latent_size):
            target_latent.append(float(latent[j]))
        target_latents.append(target_latent)

        init_latent = []
        line = fr.readline()
        latent = line.split(' ')
        for j in range(latent_size):
            init_latent.append(float(latent[j]))
        init_latents.append(init_latent)

    target_latents = np.array(target_latents)
    init_latents = np.array(init_latents)

    return init_latents, target_latents, init_range

def readMappingTable(filename):
    fr = open(filename, 'r')
    subject_n = int(fr.readline())
    case_n = int(fr.readline())

    mapping_table = []
    for i in range(subject_n):
        line = fr.readline()
        data = line.split(' ')
        for j in range(case_n):
            mapping_table.append(int(data[j]))
    mapping_table = np.array(mapping_table).reshape(subject_n, case_n)

    return mapping_table

def getRandomAMatrix(high_dim, dim, optimals, range):
    A = np.random.normal(size=(high_dim, dim))
    try:
        invA = np.linalg.pinv(A)
    except:
        print("Inverse failed!")
        return None

    low_optimals = np.matmul(invA, optimals.T).T
    conditions = (low_optimals < range) & (low_optimals > -range)
    conditions = np.all(conditions, axis=1)
    if np.any(conditions):
        return A
    else:
        print("A matrix is not qualified. Resampling......")
        return None

def generateSliderWithCenter(dim, center):
    rand_end = np.random.uniform(low=0, high=1, size=(dim))
    dir = rand_end - center
    end0 = center - dir / 2
    end1 = center + dir / 2
    return end0, end1
