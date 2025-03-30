import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

# Define the neural network
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 20),  # Input: (t, x)
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)   # Output: u
        )
        # Learnable parameter for alpha (initialized arbitrarily)
        self.alpha = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

# Physics-informed loss (PDE residual)
def physics_loss(model, x, t):
    x.requires_grad = True
    t.requires_grad = True
    u = model(x, t)
    
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    
    # Heat equation residual: u_t - alpha * u_xx = 0
    f = u_t - model.alpha * u_xx
    return torch.mean(f**2)

# Data loss (fit to synthetic measurements + boundary/initial conditions)
def data_loss(model, x_data, t_data, u_data, x_ic, u_ic, x_bc, t_bc, u_bc):
    u_pred = model(x_data, t_data)
    u_ic_pred = model(x_ic, torch.zeros_like(x_ic))
    u_bc_pred = model(x_bc, t_bc)
    return (torch.mean((u_pred - u_data)**2) + 
            torch.mean((u_ic_pred - u_ic)**2) + 
            torch.mean((u_bc_pred - u_bc)**2))

# Generate synthetic data (simulating the true solution with alpha = 0.1)
def generate_synthetic_data(n_data=20):
    x = torch.linspace(0, 1, n_data).reshape(-1, 1)
    t = torch.linspace(0, 1, n_data).reshape(-1, 1)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
    x_data = X.flatten().reshape(-1, 1)
    t_data = T.flatten().reshape(-1, 1)
    # Analytical solution: u(x,t) = sin(pi*x) * exp(-alpha * pi^2 * t)
    alpha_true = 0.1
    u_data = torch.sin(np.pi * x_data) * torch.exp(-alpha_true * np.pi**2 * t_data)
    return x_data, t_data, u_data

# Training function
def train(model, epochs=10000):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Collocation points for physics loss
    x_col = torch.FloatTensor(1000).uniform_(0, 1).reshape(-1, 1)
    t_col = torch.FloatTensor(1000).uniform_(0, 1).reshape(-1, 1)
    
    # Synthetic data
    x_data, t_data, u_data = generate_synthetic_data()
    
    # Initial condition: u(x, 0) = sin(pi * x)
    x_ic = torch.FloatTensor(100).uniform_(0, 1).reshape(-1, 1)
    u_ic = torch.sin(np.pi * x_ic)
    
    # Boundary condition: u(0, t) = u(1, t) = 0
    t_bc = torch.FloatTensor(100).uniform_(0, 1).reshape(-1, 1)
    x_bc = torch.cat([torch.zeros(50, 1), torch.ones(50, 1)])
    u_bc = torch.zeros(100, 1)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss_p = physics_loss(model, x_col, t_col)
        loss_d = data_loss(model, x_data, t_data, u_data, x_ic, u_ic, x_bc, t_bc, u_bc)
        loss = loss_p + loss_d  # Combine physics and data loss
        loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}, Estimated alpha: {model.alpha.item():.6f}")

    print(f"Final estimated alpha: {model.alpha.item():.6f} (True alpha: 0.1)")

# Initialize and train the model
model = PINN()
train(model)

# Plot the solution and compare with true solution
x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
t_test = torch.ones_like(x_test) * 0.5  # t = 0.5
u_pred = model(x_test, t_test).detach().numpy()
u_true = np.sin(np.pi * x_test.numpy()) * np.exp(-0.1 * np.pi**2 * 0.5)

plt.plot(x_test.numpy(), u_pred, label="PINN Solution at t=0.5")
plt.plot(x_test.numpy(), u_true, "--", label="True Solution at t=0.5")
plt.xlabel("x")
plt.ylabel("u(x, t)")
plt.legend()
plt.show()