import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class Generator(tf.keras.Model):
    def __init__(self, window_size, noise_dim):
        super(Generator, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(window_size, activation='tanh')
        self.reshape = tf.keras.layers.Reshape((window_size, 1))
        self.noise_dim = noise_dim

    def call(self, noise):
        x = self.dense1(noise)
        x = self.dense2(x)
        return self.reshape(x)


class Discriminator(tf.keras.Model):
    def __init__(self, window_size):
        super(Discriminator, self).__init__()
        self.conv1 = tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(window_size, 1))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(50, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, ecg):
        x = self.conv1(ecg)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)


def train_gan(generator, discriminator, real_data, epochs, batch_size, noise_dim):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False )
    
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    for epoch in range(epochs):
        for i in range(0, real_data.shape[0], batch_size):
            noise = tf.random.normal([batch_size, noise_dim])
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_data = generator(noise, training=True)

                real_output = discriminator(real_data[i:i+batch_size], training=True)
                generated_output = discriminator(generated_data, training=True)

                gen_loss = cross_entropy(tf.ones_like(generated_output), generated_output)
                disc_loss = cross_entropy(tf.ones_like(real_output), real_output) + \
                            cross_entropy(tf.zeros_like(generated_output), generated_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    
    return generator, discriminator

def augment_data(generator, data, labels, num_samples, noise_dim):
    
    augmented_data = {label: [] for label in np.unique(labels)}

    for label in np.unique(labels):
        print(f"Augmenting data for class {label}")
        for _ in range(num_samples[label]):
            noise = tf.random.normal([1, noise_dim])
            generated_sample = generator(noise, training=False)
            augmented_data[label].append(generated_sample.numpy())

    # Convert augmented data into numpy arrays and combine with original data
    augmented_data_combined = []
    augmented_labels_combined = []

    for label, samples in augmented_data.items():
        augmented_data_combined.extend(samples)
        augmented_labels_combined.extend([label] * len(samples))

    augmented_data_combined = np.array(augmented_data_combined).reshape(-1, data.shape[1], data.shape[2])
    augmented_labels_combined = np.array(augmented_labels_combined)

    # Combine the original and augmented data
    final_data = np.concatenate((data, augmented_data_combined))
    final_labels = np.concatenate((labels, augmented_labels_combined))

    return final_data, final_labels
