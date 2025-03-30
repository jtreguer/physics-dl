import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Parameters
alpha = 0.01  # Diffusion coefficient
n_samples = 1000  # Number of points to sample
n_epochs = 5000
learning_rate = 0.001

# Neural network architecture
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)),  # (x, t)
        tf.keras.layers.Dense(50, activation='tanh'),
        tf.keras.layers.Dense(50, activation='tanh'),
        tf.keras.layers.Dense(50, activation='tanh'),
        tf.keras.layers.Dense(1)  # Output u(x, t)
    ])
    return model

# Compute derivatives using automatic differentiation
def compute_pde_residual(model, x, t):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(t)
        u = model(tf.stack([x, t], axis=1))
        u_t = tape.gradient(u, t)
        with tf.GradientTape() as tape2:
            tape2.watch(x)
            u_x = tape.gradient(u, x)
        u_xx = tape2.gradient(u_x, x)
    del tape
    residual = u_t - alpha * u_xx  # Heat equation: u_t = alpha * u_xx
    return residual

# Loss function
def loss_fn(model, x_pde, t_pde, x_bc, t_bc, u_bc, x_ic, t_ic, u_ic):
    # PDE loss
    pde_residual = compute_pde_residual(model, x_pde, t_pde)
    pde_loss = tf.reduce_mean(tf.square(pde_residual))
    
    # Boundary condition loss (u = 0 at x = 0 and x = 1)
    u_bc_pred = model(tf.stack([x_bc, t_bc], axis=1))
    bc_loss = tf.reduce_mean(tf.square(u_bc_pred - u_bc))
    
    # Initial condition loss (u(x, 0) = sin(pi x))
    u_ic_pred = model(tf.stack([x_ic, t_ic], axis=1))
    ic_loss = tf.reduce_mean(tf.square(u_ic_pred - u_ic))
    
    # Total loss
    return pde_loss + bc_loss + ic_loss

# Generate training data
def generate_data(n_samples):
    # PDE domain points
    x_pde = np.random.uniform(0, 1, n_samples).astype(np.float32)
    t_pde = np.random.uniform(0, 1, n_samples).astype(np.float32)
    
    # Boundary points (x = 0 and x = 1)
    x_bc = np.concatenate([np.zeros(n_samples//2), np.ones(n_samples//2)]).astype(np.float32)
    t_bc = np.random.uniform(0, 1, n_samples).astype(np.float32)
    u_bc = np.zeros(n_samples).astype(np.float32)
    
    # Initial condition points (t = 0)
    x_ic = np.random.uniform(0, 1, n_samples).astype(np.float32)
    t_ic = np.zeros(n_samples).astype(np.float32)
    u_ic = np.sin(np.pi * x_ic).astype(np.float32)
    
    return (x_pde, t_pde), (x_bc, t_bc, u_bc), (x_ic, t_ic, u_ic)

# Training loop
model = build_model()
optimizer = tf.keras.optimizers.Adam(learning_rate)

@tf.function
def train_step(x_pde, t_pde, x_bc, t_bc, u_bc, x_ic, t_ic, u_ic):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, x_pde, t_pde, x_bc, t_bc, u_bc, x_ic, t_ic, u_ic)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# Generate data
(x_pde, t_pde), (x_bc, t_bc, u_bc), (x_ic, t_ic, u_ic) = generate_data(n_samples)

# Convert to tensors
x_pde, t_pde = tf.convert_to_tensor(x_pde), tf.convert_to_tensor(t_pde)
x_bc, t_bc, u_bc = tf.convert_to_tensor(x_bc), tf.convert_to_tensor(t_bc), tf.convert_to_tensor(u_bc)
x_ic, t_ic, u_ic = tf.convert_to_tensor(x_ic), tf.convert_to_tensor(t_ic), tf.convert_to_tensor(u_ic)

# Train the model
for epoch in range(n_epochs):
    loss = train_step(x_pde, t_pde, x_bc, t_bc, u_bc, x_ic, t_ic, u_ic)
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy():.6f}")

# Visualize the result
x_test = np.linspace(0, 1, 100).astype(np.float32)
t_test = np.linspace(0, 1, 100).astype(np.float32)
X, T = np.meshgrid(x_test, t_test)
xt_flat = np.stack([X.flatten(), T.flatten()], axis=1)
u_pred = model(xt_flat).numpy().reshape(100, 100)

plt.contourf(X, T, u_pred, levels=20)
plt.colorbar(label='u(x, t)')
plt.xlabel('x')
plt.ylabel('t')
plt.title('DGM Solution to 1D Heat Equation')
plt.show()