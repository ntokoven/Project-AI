import numpy as np

import torch
from torch import nn
import torch.distributions
import torch.utils.data as data_utils

import torch.optim as optim

import matplotlib.pyplot as plt
from tqdm import tqdm

def EMA(data, window_size=100):
    return [np.mean(data[i:i+window_size]) for i in range(0,len(data)-window_size)]



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

    def lower_bound(self, x, y, method = 'kl', step = 2, ema = 0):
        T_joint, T_marginal = self.forward(x, y)
        batch_size = x.shape[0]
        if method == 'kl':
            mine = torch.mean(T_joint) - torch.log(torch.mean(torch.exp(T_marginal)))
        elif method == 'f':
            mine = torch.mean(T_joint) - torch.mean(torch.exp(T_marginal - 1))   
        elif method == 'ema':
            alpha = 1#2 / (batch_size + 1)
            if step == 1:
                ema = torch.mean(torch.exp(T_marginal))
            else:
                ema = (1 - alpha) * ema + alpha * torch.mean(torch.exp(T_marginal))  
            #print(ema.mean(), ema)#.detach()) 
            print(torch.mean(torch.exp(T_marginal)))
            mine = torch.mean(T_joint) - (1 / ema.mean().detach()) * torch.mean(torch.exp(T_marginal))
            return -mine, ema 
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
data_x = data[:2, :].T
data_y = data[2:, :].T

# Compute MI using: MI(Y, X) = H(Y) + H(X) - H(X,Y)
cov = A @ A.T
N_x = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), torch.from_numpy(cov[:2, :2]))
N_y = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), torch.from_numpy(cov[2:, 2:]))
N_x_y = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(4), torch.from_numpy(cov))

MI = N_x.entropy().numpy() + N_y.entropy().numpy() - N_x_y.entropy().numpy()
print(MI)

batch_size = 1000
# Dataloader
kwargs = {'num_workers': 8, 'pin_memory': True}
train = data_utils.TensorDataset(torch.from_numpy(data_x).float(),
                                torch.from_numpy(data_y).float())
train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True, **kwargs) # becomes unstable and biased for batch_size of 100





num_runs = 1
winners = []
for run in range(num_runs):
    # Model and optimizer
    model = MINE()#.cuda()
    #model_f = MINE()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    #optimizer_f = optim.Adam(model_f.parameters(), lr=0.01)
    
    # Train
    loss_list = []
    epochs = 200


    model.train()
    #model_f.train()
    loss_track = {'kl':[], 'f':[], 'ema':[]}
    print('Run ', run)
    ema = 1.
    method = 'kl'
    for epoch in range(epochs):#tqdm(range(epochs)):
        loss_per_epoch = 0
        step = 1
        for _, (x, y) in enumerate(train_loader):
            #x, y = x.cuda(), y.cuda()
            model.zero_grad()
            #print('step', step)
            if method == 'ema':
                loss, ema = model.lower_bound(x, y, ema, step, method)
            else:
                loss = model.lower_bound(x, y, method)
            loss_per_epoch += loss
            loss.backward()
            optimizer.step()
            step += 1
        #loss_list.append(-loss_per_epoch / len(train_loader))  # since pytorch can only minimize the return of mine is negative, we have to invert that again
        if epoch % 10 == 0:
            print(epoch, -loss_per_epoch.detach().cpu().numpy() / len(train_loader))
        #loss_track['kl'].append(-loss_per_epoch.detach().cpu().numpy() / len(train_loader))
        #loss_track['ema'].append(-loss_per_epoch.detach().cpu().numpy() / len(train_loader))
        #loss_track['f'].append(-loss_per_epoch_f.detach().cpu().numpy() / len(train_loader))
    #print(loss_track['kl'])
    #print(np.max(loss_track['kl']))
    #best_result = {'kl': np.abs(MI - np.max(loss_track['KL'])), 'f': np.abs(MI - np.max(loss_track['f']))}
    #print('Real MI: ', MI)
    #print('Highest estimations: KL - %s, f - %s' % (np.max(loss_track['kl']), np.max(loss_track['f'])))
    #print('Closest estimations: KL - %s, f - %s\n\n' % (best_result['kl'], best_result['f']))
    #winners.append(1 if best_result['kl'] < best_result['f'] else 0)

#print('Winner: ', np.mean(winners))
# Plot
epochs = np.arange(1, epochs + 1)
MI = np.repeat(MI, len(epochs))
'''
plt.figure(1)
plt.plot(epochs, MI)
plt.plot(epochs, loss_track['KL'])
plt.plot(epochs, loss_track['f'])
plt.legend(['MI', 'MINE_KL', 'MINE_f'])
plt.ylabel('value')
plt.xlabel('correlation')
plt.show()
#'''