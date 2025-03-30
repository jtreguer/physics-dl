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
    
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

# Physics-informed loss
def physics_loss(model, x, t, nu=0.01):
    x.requires_grad = True
    t.requires_grad = True
    u = model(x, t)
    
    # Compute derivatives using autograd
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    
    # Burgers' equation residual
    f = u_t + u * u_x - nu * u_xx
    return torch.mean(f**2)

# Data loss for initial and boundary conditions
def data_loss(model, x_ic, u_ic, x_bc, u_bc, t_bc):
    u_ic_pred = model(x_ic, torch.zeros_like(x_ic))
    u_bc_pred = model(x_bc, t_bc)
    return torch.mean((u_ic_pred - u_ic)**2) + torch.mean((u_bc_pred - u_bc)**2)

# Training
def train(model, epochs=5000):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Collocation points for physics loss
    x_col = torch.FloatTensor(1000).uniform_(-1, 1).reshape(-1, 1)
    t_col = torch.FloatTensor(1000).uniform_(0, 1).reshape(-1, 1)
    
    # Initial condition: u(x, 0) = -sin(pi * x)
    x_ic = torch.FloatTensor(100).uniform_(-1, 1).reshape(-1, 1)
    u_ic = -torch.sin(np.pi * x_ic)
    
    # Boundary condition: u(-1, t) = u(1, t) = 0
    t_bc = torch.FloatTensor(100).uniform_(0, 1).reshape(-1, 1)
    x_bc = torch.cat([torch.ones(50, 1) * -1, torch.ones(50, 1) * 1])
    u_bc = torch.zeros(100, 1)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss_p = physics_loss(model, x_col, t_col)
        loss_d = data_loss(model, x_ic, u_ic, x_bc, u_bc, t_bc)
        loss = loss_p + loss_d
        loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# Initialize and train the model
model = PINN()
train(model)

# Plot the solution
x_test = torch.linspace(-1, 1, 100).reshape(-1, 1)
t_test = torch.ones_like(x_test) * 0.5  # t = 0.5
u_pred = model(x_test, t_test).detach().numpy()

plt.plot(x_test.numpy(), u_pred, label="PINN Solution at t=0.5")
plt.xlabel("x")
plt.ylabel("u(x, t)")
plt.legend()
plt.show()