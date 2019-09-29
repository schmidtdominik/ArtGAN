import tensorflow as tf


#def batch_norm(layer):
#    return tf.layers.batch_normalization(layer, training=True)

def discriminator_256x256(image_256x256=None, features_256x256=None):
    with tf.variable_scope("GAN/Discriminator/discriminator_256x256", reuse=tf.AUTO_REUSE):
        if image_256x256 is not None:
            features_256x256 = tf.layers.conv2d(image_256x256, 16, (1, 1), strides=(1, 1), padding='same', activation=tf.nn.leaky_relu, name='fromRGB_256x256')

        conv_256x256_0 = tf.layers.conv2d(features_256x256, 16, (3, 3), strides=(1, 1), padding='same', activation=tf.nn.leaky_relu, name='conv_256x256_0')
        #conv_256x256_0 = batch_norm(conv_256x256_0)
        conv_256x256_1 = tf.layers.conv2d(conv_256x256_0, 32, (3, 3), strides=(1, 1), padding='same', activation=tf.nn.leaky_relu, name='conv_256x256_1')
        #conv_256x256_1 = batch_norm(conv_256x256_1)

        resize_128x128 = tf.layers.average_pooling2d(conv_256x256_1, (2, 2), (2, 2), padding='same')

    return discriminator_128x128(features_128x128=resize_128x128)


def discriminator_128x128(image_128x128=None, features_128x128=None):
    with tf.variable_scope("GAN/Discriminator/discriminator_128x128", reuse=tf.AUTO_REUSE):
        if image_128x128 is not None:
            features_128x128 = tf.layers.conv2d(image_128x128, 32, (1, 1), strides=(1, 1), padding='same', activation=tf.nn.leaky_relu, name='fromRGB_128x128')

        conv_128x128_0 = tf.layers.conv2d(features_128x128, 32, (3, 3), strides=(1, 1), padding='same', activation=tf.nn.leaky_relu, name='conv_128x128_0')
        #conv_128x128_0 = batch_norm(conv_128x128_0)
        conv_128x128_1 = tf.layers.conv2d(conv_128x128_0, 64, (3, 3), strides=(1, 1), padding='same', activation=tf.nn.leaky_relu, name='conv_128x128_1')
        #conv_128x128_1 = batch_norm(conv_128x128_1)

        resize_64x64 = tf.layers.average_pooling2d(conv_128x128_1, (2, 2), (2, 2), padding='same')

    return discriminator_64x64(features_64x64=resize_64x64)


def discriminator_64x64(image_64x64=None, features_64x64=None):
    with tf.variable_scope("GAN/Discriminator/discriminator_64x64", reuse=tf.AUTO_REUSE):
        if image_64x64 is not None:
            features_64x64 = tf.layers.conv2d(image_64x64, 64, (1, 1), strides=(1, 1), padding='same', activation=tf.nn.leaky_relu, name='fromRGB_64x64')

        conv_64x64_0 = tf.layers.conv2d(features_64x64, 64, (3, 3), strides=(1, 1), padding='same', activation=tf.nn.leaky_relu, name='conv_64x64_0')
        #conv_64x64_0 = batch_norm(conv_64x64_0)
        conv_64x64_1 = tf.layers.conv2d(conv_64x64_0, 128, (3, 3), strides=(1, 1), padding='same', activation=tf.nn.leaky_relu, name='conv_64x64_1')
        #conv_64x64_1 = batch_norm(conv_64x64_1)

        resize_32x32 = tf.layers.average_pooling2d(conv_64x64_1, (2, 2), (2, 2), padding='same')

    return discriminator_32x32(features_32x32=resize_32x32)


def discriminator_32x32(image_32x32=None, features_32x32=None):
    with tf.variable_scope("GAN/Discriminator/discriminator_32x32", reuse=tf.AUTO_REUSE):
        if image_32x32 is not None:
            features_32x32 = tf.layers.conv2d(image_32x32, 128, (1, 1), strides=(1, 1), padding='same', activation=tf.nn.leaky_relu, name='fromRGB_32x32')

        conv_32x32_0 = tf.layers.conv2d(features_32x32, 128, (3, 3), strides=(1, 1), padding='same', activation=tf.nn.leaky_relu, name='conv_32x32_0')
        #conv_32x32_0 = batch_norm(conv_32x32_0)
        conv_32x32_1 = tf.layers.conv2d(conv_32x32_0, 256, (3, 3), strides=(1, 1), padding='same', activation=tf.nn.leaky_relu, name='conv_32x32_1')
        #conv_32x32_1 = batch_norm(conv_32x32_1)

        resize_16x16 = tf.layers.average_pooling2d(conv_32x32_1, (2, 2), (2, 2), padding='same')

    return discriminator_16x16(features_16x16=resize_16x16)


def discriminator_16x16(image_16x16=None, features_16x16=None):
    with tf.variable_scope("GAN/Discriminator/discriminator_16x16", reuse=tf.AUTO_REUSE):
        if image_16x16 is not None:
            features_16x16 = tf.layers.conv2d(image_16x16, 256, (1, 1), strides=(1, 1), padding='same', activation=tf.nn.leaky_relu, name='fromRGB_16x16')

        conv_16x16_0 = tf.layers.conv2d(features_16x16, 256, (3, 3), strides=(1, 1), padding='same', activation=tf.nn.leaky_relu, name='conv_16x16_0')
        #conv_16x16_0 = batch_norm(conv_16x16_0)
        conv_16x16_1 = tf.layers.conv2d(conv_16x16_0, 512, (3, 3), strides=(1, 1), padding='same', activation=tf.nn.leaky_relu, name='conv_16x16_1')
        #conv_16x16_1 = batch_norm(conv_16x16_1)

        resize_8x8 = tf.layers.average_pooling2d(conv_16x16_1, (2, 2), (2, 2), padding='same')

    return discriminator_8x8(features_8x8=resize_8x8)


def discriminator_8x8(image_8x8=None, features_8x8=None):
    with tf.variable_scope("GAN/Discriminator/discriminator_8x8", reuse=tf.AUTO_REUSE):
        if image_8x8 is not None:
            features_8x8 = tf.layers.conv2d(image_8x8, 512, (1, 1), strides=(1, 1), padding='same', activation=tf.nn.leaky_relu, name='fromRGB_8x8')

        conv_8x8_0 = tf.layers.conv2d(features_8x8, 512, (3, 3), strides=(1, 1), padding='same', activation=tf.nn.leaky_relu, name='conv_8x8_0')
        #conv_8x8_0 = batch_norm(conv_8x8_0)
        conv_8x8_1 = tf.layers.conv2d(conv_8x8_0, 512, (3, 3), strides=(1, 1), padding='same', activation=tf.nn.leaky_relu, name='conv_8x8_1')
        #conv_8x8_1 = batch_norm(conv_8x8_1)

        resize_4x4 = tf.layers.average_pooling2d(conv_8x8_1, (2, 2), (2, 2), padding='same')

    return discriminator_4x4(features_4x4=resize_4x4)


def discriminator_4x4(image_4x4=None, features_4x4=None):
    with tf.variable_scope("GAN/Discriminator/discriminator_4x4", reuse=tf.AUTO_REUSE):

        if image_4x4 is not None:
            features_4x4 = tf.layers.conv2d(image_4x4, 512, (1, 1), strides=(1, 1), padding='same', activation=tf.nn.leaky_relu, name='fromRGB_4x4')

        # Todo: Minibatch stddev 1 layer --> concat with rest of features
        # features_4x4 = tf.concat ... features_4x4, minibatch_stddev

        conv_4x4_0 = tf.layers.conv2d(features_4x4, 512, (3, 3), strides=(1, 1), padding='same', activation=tf.nn.leaky_relu, name='conv_4x4_0')
        #conv_4x4_0 = batch_norm(conv_4x4_0)
        conv_4x4_1 = tf.layers.conv2d(conv_4x4_0, 512, (4, 4), strides=(1, 1), padding='valid', activation=tf.nn.leaky_relu, name='conv_4x4_1')
        #conv_4x4_1 = batch_norm(conv_4x4_1)

        fc0 = tf.layers.dense(conv_4x4_1, 1, name='fc0')

    return fc0
