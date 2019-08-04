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
BUFFER_SIZE = 10000  # have a buffer size for the number of training images
BATCH_SIZE = 128  # declare the batch size

EPOCHS = 500  # define the number of epochs
noise_dim = 100  # have a noise dimension for the noise vectors
num_examples_to_generate = 16  # number of examples to generate for each batch
random_vector_for_generation = tf.random.normal(
    [num_examples_to_generate, noise_dim])  # create a random vector to generate a random image
input_dir = "./prepared_data"
output_dir = "./newPokemon"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


def prepare_data():  # get the training dataset ready
    # current_dir = os.getcwd()
    # pokemon_dir = os.path.join(current_dir, input_dir)
    pokemon_dir = input_dir
    image_paths = []
    for each in os.listdir(pokemon_dir):
        image_paths.append(os.path.join(pokemon_dir, each))
    image_tensors = []
    for image_path in image_paths:
        image_raw = tf.io.read_file(image_path)
        image_tensor = tf.image.decode_image(
            image_raw, channels=CHANNEL)
        image_tensor = tf.cast(image_tensor, tf.float32)
        image_tensors.append(image_tensor)
    dataset = tf.data.Dataset.from_tensor_slices(
        image_tensors).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    return dataset


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


def generator_loss(generated_output):
    return tf.losses.sigmoid_cross_entropy(tf.ones_like(generated_output), generated_output)


def discriminator_loss(real_output, generated_output):
    # [1,1,...,1] with real output since it is true and we want our generated examples to look like it
    real_loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.ones_like(real_output), logits=real_output)

    # [0,0,...,0] with generated images since they are fake
    generated_loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.zeros_like(generated_output), logits=generated_output)

    total_loss = real_loss + generated_loss

    return total_loss


def train_step(images):
    # general some noise
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # generate some noise
        generated_images = generator(noise, training=True)
        # test the discriminator on the real images and the fake ones
        real_output = discriminator(images, training=True)
        generated_output = discriminator(generated_images, training=True)
        # calculate the loss on the generated predictions
        gen_loss = generator_loss(generated_output)
        disc_loss = discriminator_loss(real_output, generated_output)
    # calculate gradients for the generator and dscriminator using the loss
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.variables)
    # optimize the model based on the gradients
    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.variables))
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.variables))


train_step = tf.contrib.eager.defun(train_step)


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        for images in dataset:
            train_step(images)
        display.clear_output(wait=True)

        if (epoch + 1) % 50 == 0:
            generate_and_save_images(generator, epoch + 1,
                                     random_vector_for_generation)
        if (epoch + 1) % 100 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print('Epoch: {} Time taken: {}'.format(epoch+1, time.time()-start))
    # one last generation
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epochs,
                             random_vector_for_generation)


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

    # prepare the data
    train_dataset = prepare_data()

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
    train(train_dataset, EPOCHS)
    display_image(EPOCHS)
