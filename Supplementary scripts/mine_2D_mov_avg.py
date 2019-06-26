import numpy as np

import torch
from torch import nn
import torch.distributions
import torch.utils.data as data_utils

import torch.optim as optim

import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict


class MINE(nn.Module):
    def __init__(self, dim):
        super(MINE, self).__init__()

        self.T = nn.Sequential(
            nn.Linear(2 * dim, 10),
            nn.ReLU(),
            #nn.Linear(1000, 10),
            #nn.ReLU(),
            nn.Linear(10, 1))

        for module in self.T.modules():
            if hasattr(module, 'weight'):
                torch.nn.init.xavier_uniform_(module.weight)

    def forward(self, x, y):
        y_shuffled = y[torch.randperm(y.size()[0])]
        T_joint = self.T(torch.cat((x, y), dim=1))
        T_marginal = self.T(torch.cat((x, y_shuffled), dim=1))
        return T_joint, T_marginal

    def lower_bound(self, x, y, method = 'kl', step = 2, ema = 0):
        T_joint, T_marginal = self.forward(x, y)
        batch_size = x.shape[0]
        if method == 'kl':
            mine = torch.mean(T_joint) - torch.log(torch.mean(torch.exp(T_marginal)))
        elif method == 'f':
            mine = torch.mean(T_joint) - torch.mean(torch.exp(T_marginal - 1))   
        elif method == 'ema':
            alpha = 2 / (batch_size + 1)
            if step == 1:
                ema = torch.mean(torch.exp(T_marginal))
            else:
                ema = (1 - alpha) * ema + alpha * torch.mean(torch.exp(T_marginal)).detach()  
            ema_normalization = (1 / ema) * torch.mean(torch.exp(T_marginal)).detach() #multiplying by detached mean to compensate the denominator after derivation of log
            mine = torch.mean(T_joint) - ema_normalization * torch.log(torch.mean(torch.exp(T_marginal)))
            return -mine, ema 
        return -mine

def train_MINE(data_loader, method, dim, num_runs=1, epochs=200, lr=0.01):
    for run in range(num_runs):
        print('\n\nMethod - %s. Run - %s' % (method, run))
        # Model and optimizer
        model = MINE(dim)#.cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # Train
        loss_list = []

        model.train()
        
        method = 'ema'
        if num_runs > 1:
            iterator = tqdm(range(epochs))
        else: 
            iterator = range(epochs)
        for epoch in iterator:
            loss_per_epoch = 0
            step = 1
            ema = 1.
            for _, (x, y) in enumerate(data_loader):
                #x, y = x.cuda(), y.cuda()
                model.zero_grad()
                if method == 'ema':
                    loss, ema = model.lower_bound(x, y, method, ema, step)
                    step += 1
                else:
                    loss = model.lower_bound(x, y, method)
                loss_per_epoch += loss
                loss.backward()
                optimizer.step()
            loss_list.append(-loss_per_epoch / len(train_loader))  # since pytorch can only minimize the return of mine is negative, we have to invert that again
            if epoch % 10 == 0:
                print(epoch, -loss_per_epoch.detach().cpu().numpy() / len(train_loader))
        
        print('Real MI: ', MI)
        print('Highest estimation: ', np.max(loss_list).item())
        #print('Closest estimation: KL - %s, f - %s\n\n' % (best_result['kl'], best_result['f']))
        return model, loss_list

# Create data
N = 10000
dim = 10
data = np.random.randn(2 * dim, N)

# Set covariance = A A^T
A = np.eye(2 * dim)
A[0, 2] = 3
A[0, 3] = 0
A[1, 2] = 0
A[1, 3] = 9

A = (np.random.randint(1, 10, size=(2 * dim, 2 * dim)) * np.tri(2 * dim)).T
print(A)
# Introduce correlation
data = A @ data

# Split into x and y
data_x = data[:dim, :].T
data_y = data[dim:, :].T

# Compute MI using: MI(Y, X) = H(Y) + H(X) - H(X,Y)
cov = A @ A.T
#cov = (B + B.T)/2
print(cov)
N_x = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(dim), torch.from_numpy(cov[:dim, :dim]))
N_y = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(dim), torch.from_numpy(cov[dim:, dim:]))
N_x_y = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2 * dim), torch.from_numpy(cov))

MI = N_x.entropy().numpy() + N_y.entropy().numpy() - N_x_y.entropy().numpy()
print('True value of Mutual Information - ', MI)

batch_size = 1000
# Dataloader
kwargs = {'num_workers': 8, 'pin_memory': True}
train = data_utils.TensorDataset(torch.from_numpy(data_x).float(),
                                torch.from_numpy(data_y).float())
train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True, **kwargs) # becomes unstable and biased for batch_size of 100

methods = ['kl']#kl', 'f', 'ema']
epochs = 2000
loss_track = defaultdict(list)
models = defaultdict()
for method in methods:
    models[method], loss_track[method] = train_MINE(train_loader, method, dim, epochs=epochs)
# Plot
epochs = np.arange(1, epochs + 1)
MI = np.repeat(MI, len(epochs))
legend = ['MI']
plt.figure(1)
plt.plot(epochs, MI)
for method in methods:
    plt.plot(epochs, loss_track[method])
    legend.append('MI_%s' % method)
plt.legend(legend)
plt.ylabel('value')
plt.xlabel('correlation')
plt.show()