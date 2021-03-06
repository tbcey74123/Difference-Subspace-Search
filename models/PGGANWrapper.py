import sys, os
sys.path.append(os.path.abspath('../models/PGGAN'))

import pickle
import tensorflow as tf
import numpy as np
from models.GenerativeModel import GenerativeModel

class PGGANWrapper(GenerativeModel):
    def __init__(self, pkl_path, use_approx=True):
        super(PGGANWrapper, self).__init__(use_approx=use_approx)

        self.latent_size = 512
        self.data_size = (1024, 1024)
        self.data_dim = 3
        self.expected_distance = 31.9585

        self.build(pkl_path)

    def build(self, pkl_path):
        # Initialize TensorFlow session.
        self.sess = tf.InteractiveSession()

        with open(pkl_path, 'rb') as file:
            G, D, Gs = pickle.load(file)
            self.input_latents = Gs.input_templates[0]
            self.input_labels = Gs.input_templates[1]
            self.output_images = Gs.output_templates[0]
            self.random_idx = tf.placeholder(tf.int32, shape=(None, 1), name='random_idx')

            slices = tf.gather(tf.reshape(self.output_images, [-1, 3 * 1024 * 1024]), self.random_idx[..., 0], axis=1)
            self.gradient = self.jacobian(slices, self.input_latents, parallel_iterations=50)

    def generate_data(self, n=1, z=None):
        if z is None:
            z = np.random.normal(0, 1, [n, 512])
        return self.decode(z)

    def decode(self, latent_vectors):
        batch_size = 16
        n = latent_vectors.shape[0] // batch_size
        left = latent_vectors.shape[0] - n * batch_size

        images = np.zeros((latent_vectors.shape[0], 1024, 1024, 3), dtype=np.uint8)
        for i in range(n):
            tmp_latent_vectors = latent_vectors[i * batch_size:(i + 1) * batch_size]
            tmp_images = tf.get_default_session().run(self.output_images, feed_dict={self.input_latents: tmp_latent_vectors, self.input_labels: np.zeros((tmp_latent_vectors.shape[0], 0))})
            tmp_images = np.clip(np.rint((tmp_images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
            tmp_images = tmp_images.transpose(0, 2, 3, 1) # NCHW => NHWC
            images[i * batch_size:(i + 1) * batch_size] = tmp_images
        if left != 0:
            tmp_latent_vectors = latent_vectors[n * batch_size:]
            tmp_images = tf.get_default_session().run(self.output_images, feed_dict={self.input_latents: tmp_latent_vectors, self.input_labels: np.zeros((tmp_latent_vectors.shape[0], 0))})
            tmp_images = np.clip(np.rint((tmp_images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
            tmp_images = tmp_images.transpose(0, 2, 3, 1) # NCHW => NHWC
            images[n * batch_size:] = tmp_images
        return images

    def calc_model_gradient(self, latent_vectors):
        if self.use_approx:
            idx = np.random.choice(3 * 1024 * 1024, 50, replace=False).reshape(-1, 1)
        else:
            idx = np.arange(3 * 1024 * 1024).reshape(-1, 1)
        gradient = tf.get_default_session().run(self.gradient, feed_dict={self.input_latents: latent_vectors, self.input_labels: np.zeros((latent_vectors.shape[0], 0)), self.random_idx: idx})
        gradient = gradient.reshape(-1, 512)
        return gradient

    def get_random_latent(self):
        return np.random.normal(0, 1, self.latent_size)
