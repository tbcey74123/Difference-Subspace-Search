import sys, os
sys.path.append(os.path.abspath('../models'))

import tensorflow as tf
import numpy as np

from IMGAN_ops.ops import *

from models.GenerativeModel import  GenerativeModel

class IMGAN(GenerativeModel):
    def __init__(self, use_approx=True):
        super(IMGAN, self).__init__(use_approx=use_approx)

        self.latent_size = 128
        self.data_size = None
        self.data_dim = None
        self.expected_distance = 3.19

        self.z_dim = 128
        self.z_vector_dim = 128

        self.df_dim = 2048
        self.gf_dim = 2048

        self.input_size = 64

        self.imae_ef_dim = 32
        self.imae_gf_dim = 128

        self.real_size = 64  # output point-value voxel grid size in testing
        self.test_size = 32  # related to testing batch_size, adjust according to gpu memory size
        self.batch_size = self.test_size * self.test_size * self.test_size  # do not change

        self.graph = tf.Graph()
        self.build()
        self.build_coord()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)


    def build(self):
        with self.graph.as_default():
            self.zGAN_output = self.buildZGAN()
            self.buildIMAE()

            self.random_idx = tf.placeholder(tf.int32, shape=(None, 1), name='random_idx')
            with tf.name_scope('Gradient'):
                slices = tf.gather(tf.reshape(self.zD, [1, -1]), self.random_idx[..., 0], axis=1)
                self.gradient = self.jacobian(slices, self.z, parallel_iterations=1)

    def buildZGAN(self):
        self.z_vector = tf.placeholder(shape=[None, self.z_vector_dim], dtype=tf.float32)
        self.z = tf.placeholder(shape=[None, self.z_dim], dtype=tf.float32)

        self.G = self.generator(self.z, reuse=False)
        self.D = self.discriminator(self.z_vector, reuse=False)
        self.D_ = self.discriminator(self.G, reuse=True)

        sG = self.generator(self.z, reuse=True)

        self.d_loss = tf.reduce_mean(self.D) - tf.reduce_mean(self.D_)
        self.g_loss = tf.reduce_mean(self.D_)

        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * self.z_vector + (1 - epsilon) * self.G
        d_hat = self.discriminator(x_hat, reuse=True)

        ddx = tf.gradients(d_hat, x_hat)[0]

        ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
        ddx = tf.reduce_mean(tf.square(ddx - 1.0) * 10.0)

        self.d_loss = self.d_loss + ddx

        self.zgan_vars = tf.trainable_variables()

        return sG

    def generator(self, z, reuse=False):
        with tf.variable_scope("generator") as scope:
            if reuse: scope.reuse_variables()

            h1 = lrelu(linear(z, self.gf_dim, 'g_1_lin'))
            h2 = lrelu(linear(h1, self.gf_dim, 'g_2_lin'))
            h3 = linear(h2, self.z_vector_dim, 'g_3_lin')
            return tf.nn.sigmoid(h3)

    def discriminator(self, z_vector, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse: scope.reuse_variables()

            h1 = lrelu(linear(z_vector, self.df_dim, 'd_1_lin'))
            h2 = lrelu(linear(h1, self.df_dim, 'd_2_lin'))
            h3 = linear(h2, 1, 'd_3_lin')
            return h3

    def buildIMAE(self):
        self.vox3d = tf.placeholder(shape=[1, self.input_size, self.input_size, self.input_size, 1], dtype=tf.float32)
        self.point_coord = tf.placeholder(shape=[self.batch_size, 3], dtype=tf.float32)
        self.point_value = tf.placeholder(shape=[self.batch_size, 1], dtype=tf.float32)

        self.E = self.encoder(self.vox3d, phase_train=True, reuse=False)
        self.D = self.decoder(self.point_coord, self.E, phase_train=True, reuse=False)
        self.sE = self.encoder(self.vox3d, phase_train=False, reuse=True)
        self.sD = self.decoder(self.point_coord, self.sE, phase_train=False, reuse=True)
        self.zD = self.decoder(self.point_coord, self.zGAN_output, phase_train=False, reuse=True)

        self.loss = tf.reduce_mean(tf.square(self.point_value - self.D))

        vars = tf.trainable_variables()
        self.imae_vars = [var for var in vars if var not in self.zgan_vars]

    def decoder(self, points, z, phase_train=True, reuse=False):
        with tf.variable_scope("simple_net") as scope:
            if reuse:
                scope.reuse_variables()

            zs = tf.tile(z, [self.batch_size, 1])
            pointz = tf.concat([points, zs], 1)

            h1 = lrelu(linear(pointz, self.imae_gf_dim * 16, 'h1_lin'))
            h1 = tf.concat([h1, pointz], 1)

            h2 = lrelu(linear(h1, self.imae_gf_dim * 8, 'h4_lin'))
            h2 = tf.concat([h2, pointz], 1)

            h3 = lrelu(linear(h2, self.imae_gf_dim * 4, 'h5_lin'))
            h3 = tf.concat([h3, pointz], 1)

            h4 = lrelu(linear(h3, self.imae_gf_dim * 2, 'h6_lin'))
            h4 = tf.concat([h4, pointz], 1)

            h5 = lrelu(linear(h4, self.imae_gf_dim, 'h7_lin'))
            h6 = tf.nn.sigmoid(linear(h5, 1, 'h8_lin'))

            return tf.reshape(h6, [self.batch_size, 1])

    def encoder(self, inputs, phase_train=True, reuse=False):
        with tf.variable_scope("encoder") as scope:
            if reuse:
                scope.reuse_variables()

            d_1 = conv3d(inputs, shape=[4, 4, 4, 1, self.imae_ef_dim], strides=[1, 2, 2, 2, 1], scope='conv_1')
            d_1 = lrelu(batch_norm(d_1, phase_train))

            d_2 = conv3d(d_1, shape=[4, 4, 4, self.imae_ef_dim, self.imae_ef_dim * 2], strides=[1, 2, 2, 2, 1], scope='conv_2')
            d_2 = lrelu(batch_norm(d_2, phase_train))

            d_3 = conv3d(d_2, shape=[4, 4, 4, self.imae_ef_dim * 2, self.imae_ef_dim * 4], strides=[1, 2, 2, 2, 1],
                         scope='conv_3')
            d_3 = lrelu(batch_norm(d_3, phase_train))

            d_4 = conv3d(d_3, shape=[4, 4, 4, self.imae_ef_dim * 4, self.imae_ef_dim * 8], strides=[1, 2, 2, 2, 1],
                         scope='conv_4')
            d_4 = lrelu(batch_norm(d_4, phase_train))

            d_5 = conv3d(d_4, shape=[4, 4, 4, self.imae_ef_dim * 8, self.z_dim], strides=[1, 1, 1, 1, 1], scope='conv_5',
                         padding="VALID")
            d_5 = tf.nn.sigmoid(d_5)

            return tf.reshape(d_5, [1, self.z_dim])

    def build_coord(self):
        num_per_axis = 64
        gen_num_per_axis = 32
        gen_batch_size = gen_num_per_axis * gen_num_per_axis * gen_num_per_axis
        time_per_axis = int(num_per_axis / gen_num_per_axis)
        time_per_axis2 = time_per_axis * time_per_axis
        time_per_axis3 = time_per_axis * time_per_axis * time_per_axis

        # get coords 256
        aux_x = np.zeros([gen_num_per_axis, gen_num_per_axis, gen_num_per_axis], np.int32)
        aux_y = np.zeros([gen_num_per_axis, gen_num_per_axis, gen_num_per_axis], np.int32)
        aux_z = np.zeros([gen_num_per_axis, gen_num_per_axis, gen_num_per_axis], np.int32)
        for i in range(gen_num_per_axis):
            for j in range(gen_num_per_axis):
                for k in range(gen_num_per_axis):
                    aux_x[i, j, k] = i * time_per_axis
                    aux_y[i, j, k] = j * time_per_axis
                    aux_z[i, j, k] = k * time_per_axis
        coords = np.zeros([time_per_axis3, gen_num_per_axis, gen_num_per_axis, gen_num_per_axis, 3], np.float32)
        for i in range(time_per_axis):
            for j in range(time_per_axis):
                for k in range(time_per_axis):
                    coords[i * time_per_axis2 + j * time_per_axis + k, :, :, :, 0] = aux_x + i
                    coords[i * time_per_axis2 + j * time_per_axis + k, :, :, :, 1] = aux_y + j
                    coords[i * time_per_axis2 + j * time_per_axis + k, :, :, :, 2] = aux_z + k
        coords = (coords + 0.5) / num_per_axis * 2.0 - 1.0
        self.coords = np.reshape(coords, [time_per_axis3, gen_batch_size, 3])

        self.num_per_axis = num_per_axis
        self.gen_num_per_axis = gen_num_per_axis
        self.time_per_axis = time_per_axis
        self.time_per_axis2 = time_per_axis2
        self.time_per_axis3 = time_per_axis3
        self.aux_x = aux_x
        self.aux_y = aux_y
        self.aux_z = aux_z


    def calc_model_gradient(self, latent_vector):
        if self.use_approx:
            idx = np.random.choice(self.time_per_axis3 * 32768, 50, replace=False)
        else:
            idx = np.arange(self.time_per_axis3 * 32768).reshape(-1, 1)

        counter = 0
        gradient = np.zeros((idx.shape[0], self.z_dim))
        for i in range(self.time_per_axis3):
            mask = (idx >= i * 32768) & (idx < (i + 1) * 32768)
            tmp_idx = idx[mask] - i * 32768
            tmp_gradient = self.sess.run(self.gradient, feed_dict={self.z: latent_vector, self.point_coord: self.coords[i], self.random_idx: tmp_idx.reshape(-1, 1)}).reshape(-1, self.latent_size)
            gradient[counter:counter + tmp_gradient.shape[0], :] = tmp_gradient
            counter += tmp_gradient.shape[0]
        return gradient

    def generate_data(self, n=1, z=None):
        if z is None:
            z = np.random.normal(0, 0.2, size=[n, self.z_dim])

        return self.decode(z)

    def decode(self, latent_vector):
        model_floats = np.zeros([latent_vector.shape[0], self.num_per_axis + 2, self.num_per_axis + 2, self.num_per_axis + 2], np.float32)
        for t in range(latent_vector.shape[0]):
            for i in range(self.time_per_axis):
                for j in range(self.time_per_axis):
                    for k in range(self.time_per_axis):
                        minib = i * self.time_per_axis2 + j * self.time_per_axis + k
                        model_out = self.sess.run(self.zD,
                                                  feed_dict={
                                                      self.z: latent_vector[t].reshape(1, -1),
                                                      self.point_coord: self.coords[minib],
                                                  })
                        model_floats[t, self.aux_x + i + 1, self.aux_y + j + 1, self.aux_z + k + 1] = np.reshape(model_out, [self.gen_num_per_axis, self.gen_num_per_axis, self.gen_num_per_axis])
        return model_floats.reshape(latent_vector.shape[0], self.num_per_axis + 2, self.num_per_axis + 2, self.num_per_axis + 2)

    def load_model(self, zgan_name, imae_name):
        with self.graph.as_default():
            saver = tf.train.Saver(var_list=self.zgan_vars)
            saver.restore(self.sess, zgan_name)

            saver = tf.train.Saver(var_list=self.imae_vars)
            saver.restore(self.sess, imae_name)

    def get_random_latent(self):
        return np.random.normal(0, 0.2, self.z_dim)