import numpy as np

def load_obj(filename):
    vertices = []
    faces = []
    fr = open(filename, "r")
    while True:
        line = fr.readline()
        if line:
            if line[0] == "v":
                data = line.split(" ")
                vertices.append([float(data[1]), float(data[2]), float(data[3])])
            elif line[0] == "f":
                data = line.split(" ")
                faces.append([int(data[1]) - 1, int(data[2]) - 1, int(data[3]) - 1])
            else:
                continue
        else:
            break
    vertices = np.array(vertices).reshape(-1, 3)
    faces = np.array(faces).reshape(-1, 3)

    return vertices, faces