import tensorflow as tf

from discriminator import *
from generator import *
from image_pipeline import create_pipeline_get_iterator, batch_size
from itertools import chain  # (for flattening python lists)

g_learning_rate = 0.002
d_learning_rate = 0.006

label_noise_stddev = 0.15  # truncated at stddev*2


use_wasserstein = False


def get_graph():
    graph = tf.Graph()
    with graph.as_default():

        iterator = create_pipeline_get_iterator()

        sizes = [4, 8, 16, 32, 64, 128, 256]
        generators = [generator_4x4, generator_8x8, generator_16x16, generator_32x32, generator_64x64, generator_128x128, generator_256x256]
        discriminators = [discriminator_4x4, discriminator_8x8, discriminator_16x16, discriminator_32x32, discriminator_64x64, discriminator_128x128, discriminator_256x256]

        # -- D --

        z_half = tf.truncated_normal((batch_size // 2, 4, 4, 32))

        fake_images = [gen(z_half)[1] for gen in generators]
        real_images_256x256 = iterator.get_next()
        real_images = [tf.image.resize_images(real_images_256x256, (size, size), method=tf.image.ResizeMethod.BILINEAR) for size in sizes[:-1]] + \
                      [real_images_256x256]

        d_predictions_on_fake_data = [disc(fake, None) for disc, fake in zip(discriminators, fake_images)]
        d_predictions_on_real_data = [disc(real, None) for disc, real in zip(discriminators, real_images)]

        d_losses = []
        for pred_real, pred_fake in zip(d_predictions_on_real_data, d_predictions_on_fake_data):
            if use_wasserstein:
                d_wasserstein_loss = tf.contrib.gan.losses.wargs.wasserstein_discriminator_loss(pred_real, pred_fake)
                d_losses.append(d_wasserstein_loss)
            else:
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_real, labels=tf.ones_like(pred_real)+tf.truncated_normal(pred_real.get_shape(), mean=0.0, stddev=label_noise_stddev))
                                      + tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_fake, labels=tf.zeros_like(pred_fake)+tf.truncated_normal(pred_fake.get_shape(), mean=0.0, stddev=label_noise_stddev)))
                d_losses.append(loss)



        # Training consists of two phases:
        # 1) Training:          Train the entire network (up to the current resolution)
        # -                     *Add a higher resolution layer
        # 2) Stabilization:     Freeze all previous resolution layers (that are already trained) and only train the newly added higher-res.
        # -                     *Repeat

        disc_var_lists_stabilize = [tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Discriminator/discriminator_{0}x{0}".format(size)) for size in sizes]
        disc_var_lists_train = [list(chain.from_iterable(disc_var_lists_stabilize[:i+1])) for i, size in enumerate(sizes)]

        #d_optimizer = tf.train.GradientDescentOptimizer(learning_rate=d_learning_rate)

        # 0.002+0.008*0.92^(x/10000)
        global_step = tf.Variable(0, trainable=False)
        decayed_learning_rate = tf.train.exponential_decay(0.004, global_step, 10000, 0.92, staircase=False)
        d_optimizer = tf.train.GradientDescentOptimizer(learning_rate=d_learning_rate+decayed_learning_rate)

        #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #with tf.control_dependencies(update_ops):
        d_train_optimizer_ops = [(d_optimizer.minimize(loss, var_list=var_list, global_step=global_step), loss) for var_list, loss in zip(disc_var_lists_train, d_losses)]
        d_stabilize_optimizer_ops = [(d_optimizer.minimize(loss, var_list=var_list, global_step=global_step), loss) for var_list, loss in zip(disc_var_lists_stabilize, d_losses)]


        """
        # Gradient Penalty
		self.epsilon = tf.random_uniform(
				shape=[self.batch_size, 1, 1, 1],
				minval=0.,
				maxval=1.)
		X_hat = self.X_real + self.epsilon * (self.X_fake - self.X_real)
		D_X_hat = self.discriminator(X_hat, reuse=True)
		grad_D_X_hat = tf.gradients(D_X_hat, [X_hat])[0]
		red_idx = range(1, X_hat.shape.ndims)
		slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_D_X_hat), reduction_indices=red_idx))
		gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
		self.d_loss = self.d_loss + 10.0 * gradient_penalty

		self.d_loss_sum = tf.summary.scalar("Discriminator_loss", self.d_loss)
		self.g_loss_sum = tf.summary.scalar("Generator_loss", self.g_loss)
		self.gp_sum = tf.summary.scalar("Gradient_penalty", gradient_penalty)

		train_vars = tf.trainable_variables()

		for v in train_vars:
			tf.add_to_collection("reg_loss", tf.nn.l2_loss(v))
		"""


        """
        # weight clipping
        t_vars = tf.trainable_variables()
        critic_vars = [var for var in t_vars if 'crit' in var.name]
        self.clip_critic = []
        for var in critic_vars:
        self.clip_critic.append(tf.assign(var, tf.clip_by_value(var, -0.1, 0.1)))
        """



        # -- G --

        z = tf.truncated_normal((batch_size, 4, 4, 32))
        fake_images_gen = [gen(z)[1] for gen in generators]

        g_losses = []
        for disc, fake_img in zip(discriminators, fake_images_gen):
            d_judgement = disc(fake_img, None)

            if use_wasserstein:
                g_wasserstein_loss = tf.contrib.gan.losses.wargs.wasserstein_generator_loss(d_judgement)
                g_losses.append(g_wasserstein_loss)
            else:
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_judgement, labels=tf.ones_like(d_judgement)))
                g_losses.append(loss)



        g_var_lists_stabilize = [tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Generator/generator_{0}x{0}".format(size)) for size in sizes]
        g_var_lists_train = [list(chain.from_iterable(g_var_lists_stabilize[:i + 1])) for i, size in enumerate(sizes)]

        #g_optimizer = tf.train.GradientDescentOptimizer(learning_rate=g_learning_rate)

        g_optimizer = tf.train.AdamOptimizer(learning_rate=g_learning_rate)

        #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #with tf.control_dependencies(update_ops):
        g_train_optimizer_ops = [(g_optimizer.minimize(loss, var_list=var_list), loss) for var_list, loss in zip(g_var_lists_train, g_losses)]
        g_stabilize_optimizer_ops = [(g_optimizer.minimize(loss, var_list=var_list), loss) for var_list, loss in zip(g_var_lists_stabilize, g_losses)]


    return graph, iterator, d_train_optimizer_ops, d_stabilize_optimizer_ops, g_train_optimizer_ops, g_stabilize_optimizer_ops, fake_images, sizes
