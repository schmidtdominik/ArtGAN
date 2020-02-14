import itertools
import os

from comet_ml import Experiment
import tensorflow as tf
import matplotlib as mpl
import numpy as np
import time

mpl.use('Agg')
import matplotlib.pyplot as plt

import model


epochs = 100000
steps_per_epoch = 0
total_images_looked_at = 0
d_steps = 1
g_steps = 1

graph, iterator, d_train_optimizer_ops, d_stabilize_optimizer_ops, g_train_optimizer_ops, g_stabilize_optimizer_ops, samples_for_all_resolutions, sizes = model.get_graph()


experiment = Experiment(api_key='<API_KEY>', project_name='art_pgan', workspace='schmidtdominik', log_code=False)
experiment.log_parameters({'G_learning_rate': model.g_learning_rate, 'D_learning_rate': model.d_learning_rate, 'D_steps': d_steps, 'G_steps': g_steps, 'batch_size': model.batch_size})
experiment.set_model_graph(graph)
experiment.set_code('\n# [code]: train.py\n' + open('train.py', 'r').read() + '\n# [code]: image_pipeline.py\n' + open('image_pipeline.py', 'r').read() + '\n# [code]: model.py\n' + open(
    'model.py', 'r').read() + '\n# [code]: discriminator.py\n' + open('discriminator.py', 'r').read() + '\n# [code]: generator.py\n' + open(
    'generator.py', 'r').read())

try: os.mkdir('./checkpoints/')
except FileExistsError: pass

try: os.mkdir('./progress_images/')
except FileExistsError: pass

current_resolution = sizes[0]
current_mode = 'train'
last_schedule_update = 0
last_schedule_update_time = time.time()
schedule_finalized = False

# T4 --> S8 --> T8 --> S16 --> T16 --> ...

os.system('sh reset_checkpoint.sh')
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    saver = tf.train.Saver(max_to_keep=1)
    # try:
    #     saver.restore(session, "./model.ckpt")
    #     print('Restored model.')
    # except ValueError:
    #     print('Initialized.')

    for epoch in range(epochs):
        session.run(iterator.initializer)
        experiment.log_current_epoch(epoch)

        try:

            for step in itertools.count(start=0, step=1):
                steps_per_epoch = max(steps_per_epoch, step)
                experiment.set_step(steps_per_epoch * epoch + step)
                total_images_looked_at = (steps_per_epoch * epoch + step) * (model.batch_size // 2)

                current_resolution_schedule_period_length = (0.3+current_resolution*0.003)*60*60

                #if abs(last_schedule_update-total_images_looked_at) > 5000 and not schedule_finalized:
                #if abs(time.time()-last_schedule_update_time) > 60*60*1.428 and not schedule_finalized:
                if abs(time.time() - last_schedule_update_time) > current_resolution_schedule_period_length and not schedule_finalized:
                    if current_mode == 'train':
                        current_mode = 'stabilize'
                        try:
                            current_resolution = sizes[sizes.index(current_resolution)+1]
                        except IndexError:
                            current_resolution = sizes[-1]
                            current_mode = 'train'
                            schedule_finalized = True
                    elif current_mode == 'stabilize':
                        current_mode = 'train'
                    print(current_resolution, current_mode, schedule_finalized)

                    last_schedule_update = total_images_looked_at
                    last_schedule_update_time = time.time()

                if current_mode == 'train':
                    d_opt_op, d_current_loss_tensor = d_train_optimizer_ops[sizes.index(current_resolution)]
                    g_opt_op, g_current_loss_tensor = g_train_optimizer_ops[sizes.index(current_resolution)]
                elif current_mode == 'stabilize':
                    d_opt_op, d_current_loss_tensor = d_stabilize_optimizer_ops[sizes.index(current_resolution)]
                    g_opt_op, g_current_loss_tensor = g_stabilize_optimizer_ops[sizes.index(current_resolution)]

                for i in range(d_steps):
                    try:
                        _, D_l = session.run([d_opt_op, d_current_loss_tensor])
                    except tf.errors.InvalidArgumentError as e:
                        print('jpeg error: ' + str(e))

                for i in range(g_steps):
                    _, G_l = session.run([g_opt_op, g_current_loss_tensor])

                if step % 10 == 0:
                    experiment.log_metrics({'d_loss': D_l, 'g_loss': G_l, 'current_resolution': current_resolution,
                                            'current_mode': (0 if current_mode == 'train' else 1), 'time_to_res_schedule_update': current_resolution_schedule_period_length - (time.time() - last_schedule_update_time)})
                    # experiment.log_metric("d_loss", D_l)
                    # experiment.log_metric("g_loss", G_l)
                    # experiment.log_metric("current_resolution", current_resolution)
                    # experiment.log_metric("current_mode", 0 if current_mode == 'train' else 1)

                if np.isnan(D_l) or np.isnan(G_l):
                    print('loss is NaN.')
                    exit()

                if step % 1000 == 0:
                    print('epoch: {} step: {} G_loss: {} D_loss: {}'.format(epoch, step, G_l, D_l))
                    # Save figure

                    sampled_images = session.run([samples_for_all_resolutions[sizes.index(current_resolution)]])[0]

                    plot = plt.figure(figsize=(20, 10))
                    for m in range(3):
                        plt.subplot(1, 3, m + 1)
                        plt.imshow(sampled_images[m])

                    plt.savefig('./progress_images/epoch_{0}_{1}_{2}x{2}'.format(epoch, current_mode, current_resolution))
                    experiment.log_figure(figure_name='epoch_{0}_{1}_{2}x{2}'.format(epoch, current_mode, current_resolution))

                    plt.close()

                if step % 5000 == 0:
                    save_path = saver.save(session, './checkpoints/model.ckpt', global_step=total_images_looked_at)

        except tf.errors.OutOfRangeError:
            pass
