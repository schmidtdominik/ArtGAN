import tensorflow as tf


def pixel_norm(x, epsilon=1e-8):
    with tf.variable_scope('PixelNorm'):
        return x# * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)


def batch_norm(layer):
    return tf.layers.batch_normalization(layer, training=True)


def generator_4x4(z):
    with tf.variable_scope("GAN/Generator/generator_4x4", reuse=tf.AUTO_REUSE):
        conv_4x4_0 = tf.layers.conv2d(z, 512, (4, 4), strides=(1, 1), padding='same', activation=tf.nn.leaky_relu, name='conv_4x4_0')
        conv_4x4_0 = batch_norm(conv_4x4_0)
        conv_4x4_1 = tf.layers.conv2d(conv_4x4_0, 512, (3, 3), strides=(1, 1), padding='same', activation=tf.nn.leaky_relu, name='conv_4x4_1')
        conv_4x4_1 = batch_norm(conv_4x4_1)

        toRGB_4x4 = tf.layers.conv2d(conv_4x4_1, 3, (1, 1), strides=(1, 1), padding='same', activation=tf.nn.sigmoid, name='toRGB_4x4')

    return conv_4x4_1, toRGB_4x4


def generator_8x8(z):
    features_4x4, _ = generator_4x4(z)

    with tf.variable_scope("GAN/Generator/generator_8x8", reuse=tf.AUTO_REUSE):
        resize_8x8 = tf.image.resize_images(features_4x4, (8, 8), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        conv_8x8_0 = tf.layers.conv2d(resize_8x8, 512, (3, 3), strides=(1, 1), padding='same', activation=tf.nn.leaky_relu, name='conv_8x8_0')
        conv_8x8_0 = batch_norm(conv_8x8_0)
        conv_8x8_1 = tf.layers.conv2d(conv_8x8_0, 512, (3, 3), strides=(1, 1), padding='same', activation=tf.nn.leaky_relu, name='conv_8x8_1')
        conv_8x8_1 = batch_norm(conv_8x8_1)

        toRGB_8x8 = tf.layers.conv2d(conv_8x8_1, 3, (1, 1), strides=(1, 1), padding='same', activation=tf.nn.sigmoid, name='toRGB_8x8')

    return conv_8x8_1, toRGB_8x8


def generator_16x16(z):
    features_8x8, _ = generator_8x8(z)

    with tf.variable_scope("GAN/Generator/generator_16x16", reuse=tf.AUTO_REUSE):
        resize_16x16 = tf.image.resize_images(features_8x8, (16, 16), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        conv_16x16_0 = tf.layers.conv2d(resize_16x16, 256, (3, 3), strides=(1, 1), padding='same', activation=tf.nn.leaky_relu, name='conv_16x16_0')
        conv_16x16_0 = batch_norm(conv_16x16_0)
        conv_16x16_1 = tf.layers.conv2d(conv_16x16_0, 256, (3, 3), strides=(1, 1), padding='same', activation=tf.nn.leaky_relu, name='conv_16x16_1')
        conv_16x16_1 = batch_norm(conv_16x16_1)

        toRGB_16x16 = tf.layers.conv2d(conv_16x16_1, 3, (1, 1), strides=(1, 1), padding='same', activation=tf.nn.sigmoid, name='toRGB_16x16')

    return conv_16x16_1, toRGB_16x16


def generator_32x32(z):
    features_16x16, _ = generator_16x16(z)

    with tf.variable_scope("GAN/Generator/generator_32x32", reuse=tf.AUTO_REUSE):
        resize_32x32 = tf.image.resize_images(features_16x16, (32, 32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        conv_32x32_0 = tf.layers.conv2d(resize_32x32, 128, (3, 3), strides=(1, 1), padding='same', activation=tf.nn.leaky_relu, name='conv_32x32_0')
        conv_32x32_0 = batch_norm(conv_32x32_0)
        conv_32x32_1 = tf.layers.conv2d(conv_32x32_0, 128, (3, 3), strides=(1, 1), padding='same', activation=tf.nn.leaky_relu, name='conv_32x32_1')
        conv_32x32_1 = batch_norm(conv_32x32_1)

        toRGB_32x32 = tf.layers.conv2d(conv_32x32_1, 3, (1, 1), strides=(1, 1), padding='same', activation=tf.nn.sigmoid, name='toRGB_32x32')

    return conv_32x32_1, toRGB_32x32


def generator_64x64(z):
    features_32x32, _ = generator_32x32(z)

    with tf.variable_scope("GAN/Generator/generator_64x64", reuse=tf.AUTO_REUSE):
        resize_64x64 = tf.image.resize_images(features_32x32, (64, 64), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        conv_64x64_0 = tf.layers.conv2d(resize_64x64, 64, (3, 3), strides=(1, 1), padding='same', activation=tf.nn.leaky_relu, name='conv_64x64_0')
        conv_64x64_0 = batch_norm(conv_64x64_0)
        conv_64x64_1 = tf.layers.conv2d(conv_64x64_0, 64, (3, 3), strides=(1, 1), padding='same', activation=tf.nn.leaky_relu, name='conv_64x64_1')
        conv_64x64_1 = batch_norm(conv_64x64_1)

        toRGB_64x64 = tf.layers.conv2d(conv_64x64_1, 3, (1, 1), strides=(1, 1), padding='same', activation=tf.nn.sigmoid, name='toRGB_64x64')

    return conv_64x64_1, toRGB_64x64


def generator_128x128(z):
    features_64x64, _ = generator_64x64(z)

    with tf.variable_scope("GAN/Generator/generator_128x128", reuse=tf.AUTO_REUSE):
        resize_128x128 = tf.image.resize_images(features_64x64, (128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        conv_128x128_0 = tf.layers.conv2d(resize_128x128, 32, (3, 3), strides=(1, 1), padding='same', activation=tf.nn.leaky_relu, name='conv_128x128_0')
        conv_128x128_0 = batch_norm(conv_128x128_0)
        conv_128x128_1 = tf.layers.conv2d(conv_128x128_0, 32, (3, 3), strides=(1, 1), padding='same', activation=tf.nn.leaky_relu, name='conv_128x128_1')
        conv_128x128_1 = batch_norm(conv_128x128_1)

        toRGB_128x128 = tf.layers.conv2d(conv_128x128_1, 3, (1, 1), strides=(1, 1), padding='same', activation=tf.nn.sigmoid, name='toRGB_128x128')

    return conv_128x128_1, toRGB_128x128


def generator_256x256(z):
    features_128x128, _ = generator_128x128(z)

    with tf.variable_scope("GAN/Generator/generator_256x256", reuse=tf.AUTO_REUSE):
        resize_256x256 = tf.image.resize_images(features_128x128, (256, 256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        conv_256x256_0 = tf.layers.conv2d(resize_256x256, 16, (3, 3), strides=(1, 1), padding='same', activation=tf.nn.leaky_relu, name='conv_256x256_0')
        conv_256x256_0 = batch_norm(conv_256x256_0)
        conv_256x256_1 = tf.layers.conv2d(conv_256x256_0, 16, (3, 3), strides=(1, 1), padding='same', activation=tf.nn.leaky_relu, name='conv_256x256_1')
        conv_256x256_1 = batch_norm(conv_256x256_1)

        toRGB_256x256 = tf.layers.conv2d(conv_256x256_1, 3, (1, 1), strides=(1, 1), padding='same', activation=tf.nn.sigmoid, name='toRGB_256x256')

    return conv_256x256_1, toRGB_256x256
