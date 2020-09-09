import sys, os
sys.path.append(os.path.abspath('../models'))

from GANSynth import flags as lib_flags
from GANSynth import generate_util as gu
from GANSynth import model as lib_model
from GANSynth import util
from GANSynth import train_util
import tensorflow as tf
import numpy as np
import json
from models.GenerativeModel import GenerativeModel

class GANSynthWrapper(GenerativeModel):
    def __init__(self, ckpt_path, data_size):
        self.latent_size = 256
        self.data_size = data_size
        self.data_dim = 1
        self.expected_distance = 22.6

        self.gen_data_size = 64000
        self.model = lib_model.Model.load_from_path(ckpt_path)
        self.batch_size = self.model.batch_size

        self.build()
        self.sess = tf.Session()

        exp_vars = tf.global_variables()
        exp_vars = [var for var in exp_vars if 'ExponentialMovingAverage' or 'global_step' in var.name]
        init_op = tf.initialize_variables(exp_vars)
        self.sess.run(init_op)

    def build(self):
        self.random_idx = tf.placeholder(tf.int32, shape=(None, 1), name='random_idx')
        self.target_data = tf.placeholder(tf.float32, shape=(None, self.data_size), name='target_data')

        with tf.name_scope('Gradient'):
            slices = tf.gather(self.model.fake_waves_ph[..., 0], self.random_idx[..., 0], axis=1)
            # slices = tf.gather(tf.reshape(self.model.fake_data_ph[..., 0], [-1, 128 * 1024]), self.random_idx[..., 0], axis=1)
            self.gradient = self.jacobian(slices, self.model.noises_ph, parallel_iterations=1)

    def calc_model_gradient(self, latent_vector, n=10, idx=None):
        # if idx is None:
        #     idx = np.random.choice(self.data_size // 2, n, replace=False).reshape(-1, 1)
        #     # idx = np.arange(2).reshape(-1, 1)
        #
        # extend_z = np.zeros((self.batch_size, self.latent_size))
        # min_size = np.minimum(latent_vector.shape[0], 8)
        # extend_z[:min_size] = latent_vector[:min_size]
        #
        # pitches = []
        # for i in range(extend_z.shape[0]):
        #     pitches.append(50)
        # pitches = np.array(pitches)
        # labels = self.model._pitches_to_labels(pitches)
        #
        # gradient = self.sess.run(self.gradient, feed_dict={self.model.labels_ph: labels, self.model.noises_ph: extend_z, self.random_idx: idx})[0]
        # # gradient += (np.random.rand(2, 256) - 0.5)
        # return gradient
        return self.calc_model_gradient2(latent_vector, delta=5e-5)

    def calc_model_gradient2(self, latent_vector, delta=1e-4):
        # sample_latents = np.repeat(latent_vector, repeats=self.latent_size * 2, axis=0)
        # sample_latents[:self.latent_size] -= np.identity(self.latent_size) * delta * 0.5
        # sample_latents[self.latent_size:] += np.identity(self.latent_size) * delta * 0.5
        sample_latents = np.repeat(latent_vector, repeats=self.latent_size + 1, axis=0)
        sample_latents[1:] += np.identity(self.latent_size) * delta

        # pitches = []
        # for i in range(sample_latents.shape[0]):
        #     pitches.append(50)
        # pitches = np.array(pitches)
        # sample_datas = self.model.generate_spectrograms_from_z(sample_latents, pitches).reshape(self.latent_size * 2, -1)
        # jacobian = (sample_datas[self.latent_size:] - sample_datas[:self.latent_size]).T / delta
        # # jacobian = jacobian[np.random.choice(self.data_size, 1024)]
        # jacobian = jacobian[np.random.choice(20 * 1024, 1024)]
        # return jacobian

        sample_datas = self.decode(sample_latents)

        # jacobian = (sample_datas[self.latent_size:] - sample_datas[:self.latent_size]).T / delta
        jacobian = (sample_datas[1:] - sample_datas[0]).T / delta
        idx = np.random.choice(self.data_size, 1024, replace=False)
        jacobian = jacobian[idx]
        # start_idx = np.random.choice(self.data_size // 2)
        # jacobian = jacobian[np.arange(start_idx, start_idx + 1024)]
        return jacobian

    def test_model_gradient(self, latent_vector, idx = None):
        delta = 1e-4
        sample_latents = np.repeat(latent_vector.reshape(1, -1), repeats=self.latent_size + 1, axis=0)
        sample_latents[1:] += np.identity(self.latent_size) * delta

        sample_datas = self.decode(sample_latents)

        full_jacobian = (sample_datas[1:] - sample_datas[0]).T / delta
        if idx is None:
            return full_jacobian, None

        # idx = np.random.choice(self.data_size, sample_n, replace=False)
        # stochastic_jacobian = full_jacobian[idx]
        stochastic_jacobian = full_jacobian[idx]

        return None, stochastic_jacobian

    def generate_data(self, n=1, z=None):
        if z is None:
            z = np.random.normal(size=[n, self.latent_size])
        pitches = []
        for i in range(z.shape[0]):
            pitches.append(50)
        pitches = np.array(pitches)
        waves = self.model.generate_samples_from_z(z, pitches, max_audio_length=self.data_size)

        return waves

    def decode(self, latent_vector):
        pitches = []
        for i in range(latent_vector.shape[0]):
            pitches.append(50)
        pitches = np.array(pitches)
        # print(latent_vector.shape)
        waves = self.model.generate_samples_from_z(latent_vector, pitches, max_audio_length=self.data_size)

        return waves

    def get_random_latent(self):
        return np.random.normal(0, 1, self.latent_size)