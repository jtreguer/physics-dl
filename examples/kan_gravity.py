"""
A simple example demonstrating how a KAN might uncover a physical relationship like E_p = m g h

"""

import numpy as np
import torch
from kan import KAN
import matplotlib.pyplot as plt

# Generate synthetic data for gravitational potential energy
np.random.seed(42)
m = 2.0  # mass in kg (fixed)
g = 9.81  # gravitational acceleration in m/s^2 (fixed)
h = np.linspace(0, 10, 100).reshape(-1, 1)  # height from 0 to 10 meters
U = m * g * h  # potential energy in joules

# Convert to PyTorch tensors
h_tensor = torch.tensor(h, dtype=torch.float32)
U_tensor = torch.tensor(U, dtype=torch.float32)

# Initialize KAN with a simple architecture: 1 input (h), 2 hidden nodes, 1 output (U)
model = KAN(width=[1, 2, 1], grid=5, k=3, seed=42)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

# Train the model
model.train()
for epoch in range(200):  # Increased epochs for better convergence
    optimizer.zero_grad()
    U_pred = model(h_tensor)
    loss = loss_fn(U_pred, U_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Evaluate the model
model.eval()  # Set to evaluation mode
U_pred = model(h_tensor)

# Plot results
plt.figure(figsize=(8, 6))
plt.scatter(h, U, label="True U = mgh", color="blue", alpha=0.5)
plt.plot(h, U_pred.detach().numpy(), label="KAN Prediction", color="red")
plt.xlabel("Height (m)")
plt.ylabel("Potential Energy (J)")
plt.title("KAN Modeling Gravitational Potential Energy")
plt.legend()
plt.grid(True)
plt.show()

# Symbolic regression to "discover" the relationship
model.auto_symbolic()
print("Symbolic formula discovered by KAN:")
formula, variables = model.symbolic_formula()
print(formula[0])  # Outputs an approximate symbolic expression