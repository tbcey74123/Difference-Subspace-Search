import tensorflow as tf
import numpy as np
from models.GenerativeModel import GenerativeModel

def lrelu(inputs, alpha=0.2):
    return tf.maximum(alpha * inputs, inputs)

class MNISTGenerator(GenerativeModel):
    def __init__(self, use_approx=True):
        super(MNISTGenerator, self).__init__(use_approx=use_approx)

        self.latent_size = 64
        self.data_size = (28, 28)
        self.data_dim = 2
        self.expected_distance = 6.5

        self.graph = tf.Graph()
        self.build()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)
        self.sess.run(self.init_op)

    def build(self):
        with self.graph.as_default():
            self.z = tf.placeholder(tf.float32, shape=(None, 64), name='latent_vectors')
            self.x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
            self.random_idx = tf.placeholder(tf.int32, shape=(None, 1), name='random_idx')
            self.target_data = tf.placeholder(tf.float32, shape=(28, 28), name='target_data')

            with tf.variable_scope('gen_scope'):
                self.gen_x = self.generator_structure(self.z)
            self.gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gen_scope')

            with tf.name_scope('true_disc'), tf.variable_scope('disc_scope'):
                self.disc_true = self.discriminator_structure(self.x)
            self.disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='disc_scope')

            with tf.name_scope('fake_disc'), tf.variable_scope('disc_scope', reuse=True):
                self.disc_fake = self.discriminator_structure(self.gen_x)

            self.gen_loss = -tf.reduce_mean(self.disc_fake)
            self.disc_loss = tf.reduce_mean(self.disc_fake) - tf.reduce_mean(self.disc_true)

            alpha = tf.random_uniform(shape=[tf.shape(self.x)[0], 1, 1, 1], minval=0., maxval=1.)
            differences = self.gen_x - self.x
            interpolates = self.x + (alpha * differences)
            with tf.name_scope('interp'), tf.variable_scope('disc_scope', reuse=True):
                interp = self.discriminator_structure(interpolates)

            LAMBDA = 10
            gradients = tf.gradients(interp, [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2.)
            self.disc_loss += LAMBDA * gradient_penalty

            tf.summary.scalar('G_loss', self.gen_loss)
            tf.summary.scalar('D_loss', self.disc_loss)

            # Optimizer (WGAN-GP)
            self.gen_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.0, beta2=0.9)
            self.disc_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.0, beta2=0.9)

            # Training OPs
            self.gen_train_op = self.gen_optimizer.minimize(self.gen_loss, var_list=self.gen_vars, global_step=tf.train.get_or_create_global_step())
            self.disc_train_op = self.disc_optimizer.minimize(self.disc_loss, var_list=self.disc_vars)

            with tf.name_scope('Gradient'):
                slices = tf.gather(tf.reshape(self.gen_x, [-1, 28 * 28]), self.random_idx[..., 0], axis=1)
                self.gradient = self.jacobian(slices, self.z, parallel_iterations=100)

            self.init_op = tf.global_variables_initializer()

    def generator_structure(self, z):
        gen_x = tf.layers.dense(z, 128, activation=tf.nn.relu, name='g_fc_0')
        gen_x = tf.layers.dense(gen_x, 512, activation=tf.nn.relu, name='g_fc_1')
        gen_x = tf.reshape(gen_x, [-1, 1, 1, 512])
        gen_x = tf.layers.conv2d_transpose(gen_x, 256, kernel_size=3, strides=1, padding='valid', activation=tf.nn.relu, name='g_conv_0')
        gen_x = tf.layers.conv2d_transpose(gen_x, 128, kernel_size=3, strides=2, padding='valid', activation=tf.nn.relu, name='g_conv_1')
        gen_x = tf.layers.conv2d_transpose(gen_x, 64, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu, name='g_conv_2')
        gen_x = tf.layers.conv2d_transpose(gen_x, 1, kernel_size=3, strides=2, padding='same', activation=tf.nn.sigmoid, name='g_conv_3')
        gen_x = tf.reshape(gen_x, [-1, 28, 28, 1])

        return gen_x

    def discriminator_structure(self, x, reuse=False):
        disc = tf.layers.conv2d(x, 64, kernel_size=3, strides=2, padding='same', activation=lrelu, name='d_conv_0', reuse=reuse)
        disc = tf.layers.conv2d(disc, 128, kernel_size=3, strides=2, padding='same', activation=lrelu, name='d_conv_1', reuse=reuse)
        disc = tf.layers.conv2d(disc, 256, kernel_size=3, strides=2, padding='same', activation=lrelu, name='d_conv_2', reuse=reuse)
        disc = tf.layers.conv2d(disc, 512, kernel_size=4, strides=1, padding='valid', activation=lrelu, name='d_conv_3', reuse=reuse)
        disc = tf.reshape(disc, [-1, 512])

        disc = tf.layers.dense(disc, 256, activation=lrelu, name='d_fc_0', reuse=reuse)
        disc = tf.layers.dense(disc, 64, activation=lrelu, name='d_fc_1', reuse=reuse)
        disc = tf.layers.dense(disc, 1, name='d_fc_2', reuse=reuse)

        return disc

    def train(self, data, dir='./train', batch_size=100):
        with self.graph.as_default():
            with tf.train.MonitoredTrainingSession(checkpoint_dir=dir, save_checkpoint_secs=300,
                                                          save_summaries_secs=60) as sess:
                data_mask = np.ones(data.shape[0], dtype=np.bool)
                counter = 0
                while True:
                    counter += 1
                    filtered_idx = np.arange(data.shape[0])[data_mask]
                    if filtered_idx.shape[0] == 0:
                        data_mask = np.ones(data.shape[0], dtype=np.bool)
                        filtered_idx = np.arange(data.shape[0])[data_mask]

                    if filtered_idx.shape[0] < batch_size:
                        target_idx = filtered_idx
                        data_mask = np.ones(data.shape[0], dtype=np.bool)
                    else:
                        tmp_idx = np.random.choice(filtered_idx.shape[0], batch_size, replace=False)
                        target_idx = filtered_idx[tmp_idx]
                        data_mask[target_idx] = False
                    target_data = data[target_idx]

                    z = np.random.uniform(-1, 1, [target_data.shape[0], 64])
                    feed_dict = {self.x: target_data, self.z: z}
                    sess.run(self.disc_train_op, feed_dict=feed_dict)

                    if counter % 5 == 0:
                        z = np.random.uniform(-1, 1, [target_data.shape[0], 64])
                        feed_dict = {self.x: target_data, self.z: z}
                        sess.run(self.gen_train_op, feed_dict=feed_dict)
                        counter = 0

    def get_latent(self, n=1):
        return np.random.uniform(-1, 1, [n, 64])

    def generate_data(self, n=1, z=None):
        if z is None:
            z = np.random.uniform(-1, 1, [n, 64])
        return self.decode(z)

    def get_random_latent(self):
        return np.random.uniform(-1, 1, 64)

    def decode(self, latent_vectors):
        n = latent_vectors.shape[0] // 100
        left = latent_vectors.shape[0] - n * 100
        all_data = np.zeros((latent_vectors.shape[0], 28, 28))
        for i in range(n):
            data = self.sess.run(self.gen_x, {self.z: latent_vectors[i * 100:(i + 1) * 100]})
            all_data[i * 100:(i + 1) * 100] = data[..., 0]
        if left != 0:
            data = self.sess.run(self.gen_x, {self.z: latent_vectors[n * 100:]})
            all_data[n * 100:] = data[..., 0]
        all_data = np.clip(np.rint(all_data * 255.0), 0.0, 255.0).astype(np.uint8)  # [-1,1] => [0,255]
        return all_data

    def calc_model_gradient(self, latent_vector):
        if self.use_approx:
            jacobian = self.calc_model_gradient_FDM(latent_vector, delta=1e-2)
            return jacobian
        else:
            idx = np.arange(784).reshape(-1, 1)
            gradient = self.sess.run(self.gradient, feed_dict={self.z: latent_vector.reshape(1, -1), self.random_idx: idx})
            gradient = gradient.reshape(-1, 64)
            return gradient

    def calc_model_gradient_FDM(self, latent_vector, delta=1e-4):
        sample_latents = np.repeat(latent_vector.reshape(1, -1), repeats=self.latent_size + 1, axis=0)
        sample_latents[1:] += np.identity(self.latent_size) * delta

        sample_datas = self.decode(sample_latents)
        sample_datas = sample_datas.reshape(-1, 784)

        jacobian = (sample_datas[1:] - sample_datas[0]).T / delta
        return jacobian

    def save_model(self, name):
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.sess, name)

    def load_model(self, name):
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.sess, name)
