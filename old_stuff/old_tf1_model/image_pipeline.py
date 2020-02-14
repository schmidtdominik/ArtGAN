import os
import tensorflow as tf
import numpy as np

image_size_x = 256
image_size_y = 256
batch_size = 16

def load_image(path):
    image_string = tf.read_file(path)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    image = tf.image.resize_images(image, [image_size_y, image_size_x])
    return image


def train_preprocess(image):
    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image




#files = []

# for person in os.listdir('../RESCALED_labeled_faces_in_the_wild/'):
#     for image_path in os.listdir('../RESCALED_labeled_faces_in_the_wild/' + person):
#         files.append('../RESCALED_labeled_faces_in_the_wild/{}/{}'.format(person, image_path))
"""for person in os.listdir('../RESCALED_karolinska_KDEF_and_AKDEF/KDEF/'):
    for image_path in os.listdir('../RESCALED_karolinska_KDEF_and_AKDEF/KDEF/' + person):
        file = '../RESCALED_karolinska_KDEF_and_AKDEF/KDEF/{}/{}'.format(person, image_path)
        files.append(file)"""

files = ['../PBN_crops/' + file for file in os.listdir('../PBN_crops')]

print('Training with {} image files.'.format(len(files)))


def create_pipeline_get_iterator():
    dataset = tf.data.Dataset.from_tensor_slices(files) \
        .shuffle(len(files)).map(load_image, num_parallel_calls=8) \
        .map(train_preprocess, num_parallel_calls=8) \
        .batch(batch_size // 2, drop_remainder=True) \
        .prefetch(4)
    iterator = dataset.make_initializable_iterator()

    return iterator
