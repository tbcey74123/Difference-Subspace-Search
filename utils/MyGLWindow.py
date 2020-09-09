import sys, os
sys.path.append(os.path.abspath('../utils'))

from OpenGL.GL import *
from OpenGL.GLU import *
from PyQt5.QtOpenGL import *
from PyQt5.QtGui import QOpenGLShaderProgram, QOpenGLShader, QVector3D, QVector4D, QMatrix4x4

from utils.utils import VecF

import numpy as np

class MyMeshObj():
    def __init__(self, vertices, faces, normals=None):
        self.updateMesh(vertices, faces, normals)

    def updateMesh(self, vertices, faces, normals=None):
        self.vertices = vertices
        self.faces = faces
        if normals is None:
            normals = self.updateNormals()
        self.normals = normals

    def render(self):
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        glDrawArrays(GL_TRIANGLES, 0, self.buffer_length)

        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def updateNormals(self):
        vertices = np.array(self.vertices)
        normals = np.zeros_like(vertices)
        for i in range(len(self.faces)):
            v0 = vertices[self.faces[i][0]]
            v1 = vertices[self.faces[i][1]]
            v2 = vertices[self.faces[i][2]]

            normal = np.cross((v1 - v0), (v2 - v0))
            for j in range(3):
                normals[self.faces[i][j]] += normal
        for i in range(normals.shape[0]):
            normals[i] = normals[i] / np.linalg.norm(normals[i])
        return normals.tolist()

    def buildRenderingBuffer(self, pos_id, normal_id):
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        self.updateRenderingBuffer(pos_id, normal_id)

    def updateRenderingBuffer(self, pos_id, normal_id):
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        vertices_buffer = []
        normals_buffer = []
        for i in range(len(self.faces)):
            for j in range(3):
                vertices_buffer.append(self.vertices[self.faces[i][j]][0])
                vertices_buffer.append(self.vertices[self.faces[i][j]][1])
                vertices_buffer.append(self.vertices[self.faces[i][j]][2])
                normals_buffer.append(self.normals[self.faces[i][j]][0])
                normals_buffer.append(self.normals[self.faces[i][j]][1])
                normals_buffer.append(self.normals[self.faces[i][j]][2])

        gl_vertices_buffer = VecF(vertices_buffer)
        gl_normals_buffer = VecF(normals_buffer)

        glBufferData(GL_ARRAY_BUFFER, sizeof(gl_vertices_buffer) + sizeof(gl_normals_buffer), None, GL_STATIC_DRAW)
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(gl_vertices_buffer), gl_vertices_buffer)
        glBufferSubData(GL_ARRAY_BUFFER, sizeof(gl_vertices_buffer), sizeof(gl_normals_buffer), gl_normals_buffer)

        glVertexAttribPointer(pos_id, 3, GL_FLOAT, GL_FALSE, 0, None)
        glVertexAttribPointer(normal_id, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(sizeof(gl_vertices_buffer)))

        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        self.buffer_length = int(len(vertices_buffer) / 3)

class MyCamera():
    def __init__(self, range):
        self.theta = 0
        self.phi = 0
        self.radius = range

        self.focus_center = QVector3D(0, 0, 0)

    def rotate(self, dTheta, dPhi):
        self.theta += dTheta
        self.phi += dPhi

    def zoom(self, distance):
        self.radius -= distance

    def pan(self, dx, dy):
        look = self.getLook()
        worldUp = np.array([0.0, 1.0, 0.0])

        right = np.cross(look, worldUp)
        up = np.cross(look, right)

        focus_center = np.array([self.focus_center.x(), self.focus_center.y(), self.focus_center.z()])
        focus_center += right * dx + up * dy
        self.focus_center = QVector3D(focus_center[0], focus_center[1], focus_center[2])

    def focus(self, obj):
        center = np.mean(np.array(obj.vertices), axis=0)

        self.focus_center = QVector3D(center[0], center[1], center[2])

    def getLook(self):
        def toCartesian(radius, theta, phi):
            x = radius * np.sin(np.deg2rad(phi)) * np.cos(np.deg2rad(theta))
            y = radius * np.sin(np.deg2rad(phi)) * np.sin(np.deg2rad(theta))
            z = radius * np.cos(np.deg2rad(phi))
            return np.array([x, y, z])
        look = toCartesian(self.radius, self.theta, self.phi)
        look /= np.linalg.norm(look)
        return look

    def getMatrix(self):
        look = self.getLook()
        focus_center = np.array([self.focus_center.x(), self.focus_center.y(), self.focus_center.z()])
        camera_position = focus_center - look * self.radius

        matrix = QMatrix4x4()
        matrix.lookAt(QVector3D(camera_position[0], camera_position[1], camera_position[2]), self.focus_center, QVector3D(0, 1, 0))

        return matrix

class MyGLWindow(QGLWidget):

    def __init__(self, obj_num, camera_dist=60):
        super(MyGLWindow, self).__init__()
        self.obj_num = obj_num
        self.objs = []
        for i in range(obj_num):
            self.objs.append(None)
        self.shaderProgram = None

        self.is_rotate = False
        self.is_pan = False
        self.pan_param = 0.1
        self.rotate_param = 0.5

        self.camera = MyCamera(camera_dist)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.shaderProgram.bind()

        p_matrix = QMatrix4x4()
        p_matrix.perspective(60.0, 1.33, 0.1, 1000)
        self.shaderProgram.setUniformValue(self.p_matrix_id, p_matrix)

        self.shaderProgram.setUniformValue(self.main_light_dir, QVector4D(0, 1.0, -1.0, 0.0))
        self.shaderProgram.setUniformValue(self.sub_light_dir, QVector4D(0, 1.0, 1.0, 0.0))
        self.shaderProgram.setUniformValue(self.back_light_dir, QVector4D(0, -1.0, 0.0, 0.0))
        self.shaderProgram.setUniformValue(self.ambient_id, QVector4D(0.15, 0.15, 0.15, 0.0))
        self.shaderProgram.setUniformValue(self.diffuse_id, QVector4D(1.0, 1.0, 1.0, 1.0))

        mv_matrix = self.camera.getMatrix()
        self.shaderProgram.setUniformValue(self.mv_matrix_id, mv_matrix)

        viewport_width = self.width() / self.obj_num
        for i in range(self.obj_num):
            glViewport(int(i * viewport_width), 0, int(viewport_width), self.height())
            if self.objs[i] is not None:
                self.objs[i].render()

        self.shaderProgram.release()

    def initializeGL(self):
        glClearColor(0.95, 0.95, 0.95, 1.0)
        glClearDepth(1.0)
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60.0, 1.33, 0.1, 1000)

        self.shaderProgram = QOpenGLShaderProgram()
        print("Initialize vertex shader: ", self.shaderProgram.addShaderFromSourceFile(QOpenGLShader.Vertex, "shader.vert"))
        print("Initialize fragment shader: ", self.shaderProgram.addShaderFromSourceFile(QOpenGLShader.Fragment, "shader.frag"))
        self.shaderProgram.link()

        self.p_matrix_id = self.shaderProgram.uniformLocation("um4p")
        self.mv_matrix_id = self.shaderProgram.uniformLocation("um4mv")
        self.main_light_dir = self.shaderProgram.uniformLocation("main_light_dir")
        self.sub_light_dir = self.shaderProgram.uniformLocation("sub_light_dir")
        self.back_light_dir = self.shaderProgram.uniformLocation("back_light_dir")

        self.ambient_id = self.shaderProgram.uniformLocation("ambient")
        self.diffuse_id = self.shaderProgram.uniformLocation("diffuse")
        self.pos_id = self.shaderProgram.attributeLocation("pos")
        self.normal_id = self.shaderProgram.attributeLocation("normal")

    def mousePressEvent(self, event):
        if event.button() == 2:
            self.is_rotate = True
        elif event.button() == 4:
            self.is_pan = True
        self.lastPos = event.pos()

    def mouseMoveEvent(self, event):
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()

        if self.is_rotate:
            self.camera.rotate(dy * self.rotate_param, -dx * self.rotate_param)
        elif self.is_pan:
            self.camera.pan(dx * self.pan_param, dy * self.pan_param)

        self.lastPos = event.pos()
        self.repaint()

    def mouseReleaseEvent(self, event):
        self.is_rotate = False
        self.is_pan = False

    def setObj(self, idx, vertices, faces, normals=None, focus=False):
        self.shaderProgram.bind()

        if self.objs[idx] is None:
            self.objs[idx] = MyMeshObj(vertices, faces, normals)
            self.objs[idx].buildRenderingBuffer(self.pos_id, self.normal_id)
        else:
            self.objs[idx].updateMesh(vertices, faces)
            self.objs[idx].updateRenderingBuffer(self.pos_id, self.normal_id)

        self.shaderProgram.release()

        if focus:
            self.camera.focus(self.objs[idx])