import numpy as np


import torch.distributions
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

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
        y=y.float()
        x = x.float()
        y_shuffled = y[torch.randperm(y.size()[0])]
        print(x.shape,y.shape)
        T_joint = self.T(torch.cat((x, y), dim=1))
        T_marginal = self.T(torch.cat((x, y_shuffled), dim=1))
        
        return T_joint, T_marginal

    def lower_bound(self, x, y):
        T_joint, T_marginal = self.forward(x, y)
        mine = torch.mean(T_joint) - torch.log(torch.mean(torch.exp(T_marginal)))

        return -mine

    def trainMine(self, trainLoader, epochs, batch_size, plot=False, transF=None, target=False):
        model = MINE()  # .cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        train_loader = trainLoader  # becomes unstable and biased for batch_size of 100
        # Train
        loss_list = []
        for epoch in range(epochs):
            loss_per_epoch = 0
            for _, (x, y) in enumerate(train_loader):
                # x, y = x.cuda(), y.cuda()
                model.zero_grad()
                if target:
                    tX = transF(x).detach()
                    loss = model.lower_bound(y, tX)
                else:
                    tX = transF(x).detach()
                    loss = model.lower_bound(x, tX)
                loss_per_epoch += loss
                loss.backward()
                optimizer.step()
            loss_list.append(-loss_per_epoch / len(
                train_loader))  # since pytorch can only minimize the return of mine is negative, we have to invert that again
            print(epoch, -loss_per_epoch.detach().cpu().numpy() / len(train_loader))
        if plot:
            # Plot
            epochs = np.arange(1, epochs + 1)
            plt.figure(1)
            plt.plot(epochs, loss_list)
            plt.legend(['MINE'])
            plt.ylabel('lower bound')
            plt.xlabel('epochs')
            plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ConvNet():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
        self.parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                                 help='input batch size for training (default: 64)')
        self.parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                                 help='input batch size for testing (default: 1000)')
        self.parser.add_argument('--epochs', type=int, default=10, metavar='N',
                                 help='number of epochs to train (default: 10)')
        self.parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                                 help='learning rate (default: 0.01)')
        self.parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                                 help='SGD momentum (default: 0.5)')
        self.parser.add_argument('--no-cuda', action='store_true', default=False,
                                 help='disables CUDA training')
        self.parser.add_argument('--seed', type=int, default=1, metavar='S',
                                 help='random seed (default: 1)')
        self.parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                                 help='how many batches to wait before logging training status')

        self.parser.add_argument('--save-model', action='store_true', default=False,
                                 help='For Saving the current Model')
        self.args = self.parser.parse_args()
        use_cuda = not self.args.no_cuda and torch.cuda.is_available()

        torch.manual_seed(self.args.seed)

        self.device = torch.device("cuda" if use_cuda else "cpu")

        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        self.model = Net().to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)

    def train(self, train_loader,epoch):
        self.model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

    def test(self,test_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    def run(self):
        for epoch in range(1, self.args.epochs + 1):
            self.train(self.args, self.model, self.device, self.train_loader, self.optimizer, epoch)
            self.test(self.args, self.model, self.device, self.test_loader)

        if (self.args.save_model):
            torch.save(self.model.state_dict(), "mnist_cnn.pt")

class TrackMI():
    def __init__(self):
        self.batch_size = 64
        self.mineEpoch = 100
        self.convN = ConvNet()
        self.mine = MINE()
        # Dataloader
        kwargs = {'num_workers': 8, 'pin_memory': True}
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=self.batch_size, shuffle=True, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=self.batch_size, shuffle=True, **kwargs)

    def run(self):
        for i in range(self.convN.args.epochs):
            if (i+1) % 1 == 0:
                # self.convN.train(self.train_loader,i)
                for layer in self.convN.net.layers:
                    transFunction = layer
                    mIx = self.mine.trainMine(self.train_loader, self.mineEpoch, self.batch_size, plot=False, transF=transFunction, target=False)
                    mIy = self.mine.trainMine(self.train_loader, self.mineEpoch, self.batch_size, plot=False, transF=transFunction, target=True)

            else:
                self.convN.train(self.train_loader,i)

def main():
    trackMI=TrackMI()
    trackMI.run()

if __name__ == '__main__':
    # freeze_support() here if program needs to be frozen
    main()

