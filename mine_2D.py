import numpy as np

import torch
from torch import nn
import torch.distributions
import torch.utils.data as data_utils

import torch.optim as optim

import matplotlib.pyplot as plt


class MINE(nn.Module):
    def __init__(self):
        super(MINE, self).__init__()

        self.T = nn.Sequential(
            nn.Linear(4, 10),
            nn.ReLU(),
            nn.Linear(10, 1))

        for module in self.T.modules():
            if hasattr(module, 'weight'):
                torch.nn.init.xavier_uniform_(module.weight)

    def forward(self, x, y):
        y_shuffled = y[torch.randperm(y.size()[0])]
        #print(x.shape, '\n\n\n')
        #print(y.shape, '\n\n\n')
        #print(torch.cat((x, y), dim=1), '\n\n\n')
        #print(self.T(torch.cat((x, y), dim=1)), '\n\n\n')
        T_joint = self.T(torch.cat((x, y), dim=1))
        T_marginal = self.T(torch.cat((x, y_shuffled), dim=1))

        return T_joint, T_marginal

    def lower_bound(self, x, y):
        T_joint, T_marginal = self.forward(x, y)
        mine = torch.mean(T_joint) - torch.log(torch.mean(torch.exp(T_marginal)))

        return -mine

# Create data
N = 10000
data = np.random.randn(4, N)

# Set covariance = A A^T
A = np.eye(4)
A[0, 2] = 3
A[0, 3] = 0
A[1, 2] = 0
A[1, 3] = 9

# Introduce correlation
data = A @ data

# Split into x and y
x = data[:2, :].T
y = data[2:, :].T

# Compute MI using: MI(Y, X) = H(Y) + H(X) - H(X,Y)
cov = A @ A.T
N_x = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), torch.from_numpy(cov[:2, :2]))
N_y = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), torch.from_numpy(cov[2:, 2:]))
N_x_y = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(4), torch.from_numpy(cov))

MI = N_x.entropy().numpy() + N_y.entropy().numpy() - N_x_y.entropy().numpy()
print(MI)

# Model and optimizer
model = MINE()#.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Dataloader
kwargs = {'num_workers': 8, 'pin_memory': True}
train = data_utils.TensorDataset(torch.from_numpy(x).float(),
                                 torch.from_numpy(y).float())
train_loader = data_utils.DataLoader(train, batch_size=1000, shuffle=True, **kwargs) # becomes unstable and biased for batch_size of 100

# Train
loss_list = []
epochs = 100


model.train()

for epoch in range(epochs):
    loss_per_epoch = 0
    for _, (x, y) in enumerate(train_loader):
        #x, y = x.cuda(), y.cuda()
        model.zero_grad()
        loss = model.lower_bound(x, y)
        loss_per_epoch += loss
        loss.backward()
        optimizer.step()
    loss_list.append(-loss_per_epoch / len(train_loader))  # since pytorch can only minimize the return of mine is negative, we have to invert that again
    print(epoch, -loss_per_epoch.detach().cpu().numpy() / len(train_loader))

# Plot
epochs = np.arange(1, epochs + 1)
MI = np.repeat(MI, len(epochs))

plt.figure(1)
plt.plot(epochs, MI)
plt.plot(epochs, loss_list)
plt.legend(['MI', 'MINE'])
plt.ylabel('value')
plt.xlabel('correlation')
plt.show()
