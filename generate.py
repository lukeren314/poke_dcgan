from IPython import display
import time
import PIL
import os
import imageio
import glob
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
# define the dimensions of thei mage, including the rgb channel
HEIGHT, WIDTH, CHANNEL = 128, 128, 3

noise_dim = 100  # have a noise dimension for the noise vectors
num_examples_to_generate = 16  # number of examples to generate for each batch
num_batches_to_generate = 100
random_vector_for_generation = tf.random.normal(
    [num_examples_to_generate, noise_dim])  # create a random vector to generate a random image

output_dir = "./newPokemon"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


def generator_model():  # create the generator model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(
        8*8*256, use_bias=False, input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((8, 8, 256)))
    # Note: None is the batch size
    assert model.output_shape == (None, 8, 8, 256)

    model.add(tf.keras.layers.Conv2DTranspose(
        128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 128)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(
        64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 64)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(
        32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 64, 64, 32)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(
        2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 128, 128, 3)

    return model


def discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(
        64, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(
        128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))

    return model


def generate_and_save_images(model, epoch, test_input):
    # make sure the training parameter is set to False because we
    # don't want to train the batchnorm layer when doing inference.
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        # plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.imshow(predictions[i, :, :, :]*127.5+127.5)
        plt.axis('off')

    plt.savefig(output_dir+'/epoch_{:04d}.png'.format(epoch))
    # plt.show()


def display_image(epoch_no):
    return PIL.Image.open(output_dir+'/epoch_{:04d}.png'.format(epoch_no))


if __name__ == "__main__":
    # create the models and optimizers
    generator = generator_model()
    discriminator = discriminator_model()

    generator_optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)
    discriminator_optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)

    # set up checkpoints
    checkpoint_dir = './training_checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    # restoring the latest checkpoint in checkpoint_dir
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    for i in range(num_batches_to_generate):
        generate_and_save_images(generator, i, random_vector_for_generation)
