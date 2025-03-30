import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Parameters
IMG_HEIGHT, IMG_WIDTH = 64, 64  # Spectrogram dimensions
NOISE_DIM = 100  # Size of the random noise vector
NUM_EXAMPLES = 16  # Number of spectrograms to generate
BATCH_SIZE = 32
EPOCHS = 50

# Generator Model
def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(8 * 8 * 256, input_dim=NOISE_DIM))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((8, 8, 256)))
    
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', activation='tanh'))
    return model

# Discriminator Model
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[IMG_HEIGHT, IMG_WIDTH, 1]))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# Loss Functions
cross_entropy = tf.keras.losses.BinaryCrossentropy()

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Build models
generator = build_generator()
discriminator = build_discriminator()

# Training step
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# Simulate a small "real" dataset of spectrograms (for training)
def generate_dummy_spectrograms(num_samples):
    # Simulate spectrograms with random patterns (e.g., sine waves + noise)
    spectrograms = np.zeros((num_samples, IMG_HEIGHT, IMG_WIDTH, 1))
    for i in range(num_samples):
        x = np.linspace(0, 10, IMG_WIDTH)
        y = np.linspace(0, 5, IMG_HEIGHT)
        X, Y = np.meshgrid(x, y)
        # Simulate a frequency pattern with noise
        spectrograms[i, :, :, 0] = np.sin(2 * np.pi * (X + Y)) + np.random.normal(0, 0.1, (IMG_HEIGHT, IMG_WIDTH))
    return spectrograms

# Training loop
def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch)
        print(f'Epoch {epoch + 1}/{epochs} completed')

# Generate and save synthetic spectrograms
def generate_spectrograms(generator, num_examples):
    noise = tf.random.normal([num_examples, NOISE_DIM])
    generated_spectrograms = generator(noise, training=False)
    return generated_spectrograms.numpy()

# Main execution
if __name__ == "__main__":
    # Create dummy "real" spectrograms for training
    real_spectrograms = generate_dummy_spectrograms(100)
    dataset = tf.data.Dataset.from_tensor_slices(real_spectrograms).shuffle(100).batch(BATCH_SIZE)
    
    # Train the GAN
    train(dataset, EPOCHS)
    
    # Generate synthetic spectrograms
    synthetic_spectrograms = generate_spectrograms(generator, NUM_EXAMPLES)
    
    # Visualize a few examples
    plt.figure(figsize=(10, 10))
    for i in range(NUM_EXAMPLES):
        plt.subplot(4, 4, i + 1)
        plt.imshow(synthetic_spectrograms[i, :, :, 0], cmap='viridis')
        plt.axis('off')
    plt.show()
    
    # Save the dataset (e.g., as NumPy array)
    np.save('synthetic_spectrograms.npy', synthetic_spectrograms)
    print(f"Generated and saved {NUM_EXAMPLES} synthetic spectrograms.")