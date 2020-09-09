import tensorflow as tf
import numpy as np

class GenerativeModel(object):
    def __init__(self):
        self.latent_size = None
        self.data_size = None
        self.data_dim = None

    def build(self):
        pass

    def generate_data(self, n=1, z=None):
        pass

    def decode(self, latent_vectors):
        pass

    def calc_model_gradient(self, latent_vector, n=100, idx=None):
        pass

    def get_random_latent(self):
        pass

    def jacobian(self, fx, x, parallel_iterations=10):
        dtype = x.dtype

        n = tf.shape(fx)[1]
        loop_vars = [
            tf.constant(0),
            tf.TensorArray(dtype, size=n),
        ]

        _, grads = tf.while_loop(
            lambda j, _: j < n,
            lambda j, result: (j + 1, result.write(j, tf.gradients(tf.gather(fx, j, axis=1), x))),
            loop_vars,
            parallel_iterations=parallel_iterations
        )
        grads = grads.stack()
        grads = tf.transpose(grads, [2, 1, 0, 3])
        grads = tf.reshape(grads, [tf.shape(fx)[0], tf.shape(fx)[1], tf.shape(x)[1]])

        return grads

    def close(self):
        self.sess.close()