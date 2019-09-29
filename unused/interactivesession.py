import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from model import model

graph, iterator, D_optimizer, D_loss, G_optimizer, G_loss, ER_optimizer, ER_loss, G_samples, exp_replay_data, G_one_sample, specific_z, specific_G_samples = model.get_graph()


import collections
ER_sample_acc = collections.deque(maxlen=model.batch_size // 2)

session = tf.InteractiveSession(graph=graph)
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

saver = tf.train.Saver()
saver.restore(session, "./model.ckpt")

z = np.random.normal(size=(1, model.G_input_noise_size))

for i in range(10):
    z[0][25] = -10 + i * 3
    sample = session.run([specific_G_samples], feed_dict={specific_z: z})[0]

    plt.imshow(sample[0])
    plt.show()

session.close()


