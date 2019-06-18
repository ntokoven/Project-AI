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
import math
from tqdm import tqdm
from collections import defaultdict
import pickle
import random


class ChangeShape(nn.Module):
    def _init_(self):
        super(ChangeShape, self)._init_()

    def forward(self, x,layer):
        #print("orig", x.shape, layer.shape)
        if x.shape == layer.shape:
            return x
        elif x.dim() == layer.dim():
            if x.dim() == 4:
                x = nn.Conv2d(1, 1, (x.shape[2] - layer.shape[2] + 1, x.shape[2] - layer.shape[2] + 1))(x)
                if x.shape[1] != layer.shape[1]:
                    noRepeat = layer.shape[1] - x.shape[1] + 1
                    x = x.repeat(1, noRepeat, 1, 1)
            else:
                layer = nn.Linear(layer.shape[1], x.shape[1])(layer)
        else:
            if x.dim() > layer.dim():
                x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
                x = nn.Linear(x.shape[1], layer.shape[1])(x)
            else:
                layer = layer.reshape(layer.shape[0], layer.shape[1] * layer.shape[2] * layer.shape[3])
                layer = nn.Linear(layer.shape[1], x.shape[1])(layer)
        #print("transformed", x.shape, layer.shape)
        return x, layer

class MINE(nn.Module):
    def __init__(self, shape_input, shape_output, args):
        super(MINE, self).__init__()

        self.args = args
        use_cuda = not self.args.no_cuda and torch.cuda.is_available()
        torch.manual_seed(self.args.seed)
        self.device = torch.device("cuda" if use_cuda else "cpu")

        #self.change_shape = ChangeShape().to(self.device)
        a = torch.ones(shape_input).to(self.device)#.item())
        b = torch.ones(shape_output).to(self.device)#.item())
        a, b = self.change_shape(a, b)
        self.n_input = torch.prod(torch.tensor(a.shape[1:])).item()
        self.T = nn.Sequential(
            nn.Linear(self.n_input*2, 10),
            nn.ReLU(),
            nn.Linear(10, 1)).to(self.device)

        for module in self.T.modules():
            if hasattr(module, 'weight'):
                torch.nn.init.xavier_uniform_(module.weight)

    def forward(self, x, y):
        y = y.float()
        x = x.float()
        batch_size = x.shape[0]
        x, y = self.change_shape(x, y)
        input_size = torch.prod(torch.tensor(x.shape[1:])).item()
        x, y = x.reshape((batch_size, input_size)), \
               y.reshape((batch_size, input_size))
        y_shuffled = y[torch.randperm(y.size()[0])]
        T_joint = self.T(torch.cat((x, y), dim=1))
        T_marginal = self.T(torch.cat((x, y_shuffled), dim=1))

        return T_joint, T_marginal

    def lower_bound(self, x, y):
        T_joint, T_marginal = self.forward(x, y)
        mine = torch.mean(T_joint) - torch.log(torch.mean(torch.exp(T_marginal)))

        return -mine

    def lcm(self,a, b):
        return abs(a * b) // math.gcd(a, b)
    #'''
    def change_shape(self, x, layer):
        #x, layer = x.float(), layer.float()
        x, layer = x.to(self.device), layer.to(self.device)
        # print(x.get_device(), layer.get_device())


        #print("orig", x.shape, layer.shape)
        if x.shape == layer.shape:
            x.to(self.device), layer.to(self.device)
        elif x.dim() == layer.dim():
            if x.dim() == 4:
                x = nn.Conv2d(1, 1, (x.shape[2]-layer.shape[2]+1,x.shape[2]-layer.shape[2]+1)).to(self.device)(x)
                if x.shape[1] != layer.shape[1]:
                    noRepeat = layer.shape[1]-x.shape[1]+1
                    x = x.repeat(1,noRepeat,1,1)
            else:
                layer=nn.Linear(layer.shape[1],x.shape[1]).to(self.device)(layer)
        else:
            if x.dim()>layer.dim():
                x=x.reshape(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3])
                x=nn.Linear(x.shape[1],layer.shape[1]).to(self.device)(x)
            else:
                layer=layer.reshape(layer.shape[0],layer.shape[1]*layer.shape[2]*layer.shape[3])
                layer=nn.Linear(layer.shape[1],x.shape[1]).to(self.device)(layer)
        #print("transformed",x.shape,layer.shape)
        return x.to(self.device), layer.to(self.device)
    #'''

    #"brute-force" (every with every) version of shape transition 
    def change_shape_old(self, x, y): 
        #print('Shapes ot T input:')
        #print(x.shape)
        #print(y.shape,'\n\n\n')
        '''
        Change to convolutions
        '''
        n = abs(x.dim() - y.dim())
        if x.dim() > y.dim():
            for i in range(n):
                y = torch.unsqueeze(torch.tensor(y), 1).clone().detach().requires_grad_(True)
        elif x.dim() < y.dim():
            for i in range(n):
                x = torch.unsqueeze(torch.tensor(x), 1).clone().detach().requires_grad_(True)
        desired_shape = []
        for i in range(x.dim()):
            desired_shape.append(self.lcm(x.shape[i], y.shape[i]))
        d = 1
        for shape in desired_shape:
            d *= shape
        xresh=x.reshape(int(torch.prod(torch.tensor(x.shape[0:]))))
        yresh=y.reshape(int(torch.prod(torch.tensor(y.shape[0:]))))
        x = xresh.repeat(d // xresh.shape[0]).reshape(desired_shape)
        y = yresh.repeat(d // yresh.shape[0]).reshape(desired_shape)
        return x, y


class LeNet(nn.Module):
    def __init__(self, args):
        super(LeNet, self).__init__()
        self.args = args

        use_cuda = not self.args.no_cuda and torch.cuda.is_available()
        torch.manual_seed(self.args.seed)
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.moduleList = {}

        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.moduleList["conv1"] = self.conv1

        self.relu1 = nn.ReLU()
        self.moduleList["relu1"]= self.relu1

        self.maxP1 = nn.MaxPool2d(2,2)
        self.moduleList['maxP1'] = self.maxP1

        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.moduleList['conv2'] = self.conv2

        self.relu2= nn.ReLU()
        self.moduleList['relu2'] = self.relu2

        self.maxP2 = nn.MaxPool2d(2,2)
        self.moduleList['maxP2'] = self.maxP2

        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.moduleList['fc1'] = self.fc1

        self.relu3 = nn.ReLU()
        self.moduleList['relu3'] = self.relu3

        self.fc2 = nn.Linear(500, 10)
        self.moduleList['fc2'] = self.fc2

        self.sm1 = nn.LogSoftmax(dim=1)
        self.moduleList['sm1'] = self.sm1

    def forward(self, x, exitLayer=None):
        x = x.to(self.device)
        for layer in self.moduleList.keys():
            if layer == 'fc1':
                x = x.view(-1, 4 * 4 * 50)
            x = self.moduleList[layer](x)
            if type(exitLayer) != 'NoneType':
                if layer == exitLayer:
                    break
            #if layer == 'fc2':
            #    x = self.sm1(x)
            #    break
        return x

    def get_dims(self, x):
        x = x.to(self.device)
        with torch.no_grad():
            dims = {}
            for layer in self.moduleList.keys():
                if layer == 'fc1':
                    x = x.view(-1, 4 * 4 * 50)
                x = self.moduleList[layer](x).to(self.device)
                if layer == 'maxP1':
                    dims['maxP1'] = x.shape#torch.prod(torch.tensor(x.shape))
                elif layer == 'maxP2':
                    dims['maxP2'] = x.shape#torch.prod(torch.tensor(x.shape[1:]))
                elif layer =='relu3':
                    dims['relu3'] = x.shape#torch.prod(torch.tensor(x.shape[1:]))
                elif layer == 'sm1':
                    dims['sm1'] = x.shape
        return dims

#Class for training and evaluating of target network
class ConvNet(nn.Module):
    def __init__(self, args):
        super(ConvNet, self).__init__()
        self.args = args 
        use_cuda = not self.args.no_cuda and torch.cuda.is_available()

        torch.manual_seed(self.args.seed)

        self.device = torch.device("cuda" if use_cuda else "cpu")

        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        self.model = LeNet(args).to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)

    def train(self, train_loader, epoch):
        self.model.train()
        print('Training ConvNet. Epoch: ', epoch)
        batches_nums = random.sample(range(500), 50)
        for batch_idx, (data, target) in enumerate(train_loader):
            '''
            if batch_idx not in batches_nums:
                continue
            else:
            '''
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
            '''
            Constraints the amount of data used per epoch 
            '''
            if batch_idx == 100:
                break

    def test(self, test_loader):
        self.model.eval()
        print('\nTesting ConvNet')
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

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    
class TrackMI(nn.Module):
    def __init__(self, args):
        super(TrackMI, self).__init__()
        self.args = args
        use_cuda = not self.args.no_cuda and torch.cuda.is_available()
        torch.manual_seed(args.seed)
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.batch_size = self.args.batch_size
        self.mine_epochs = self.args.mine_epochs
        self.convN = ConvNet(args).to(self.device)
        self.device = self.convN.device
        
        self.num_classes = 10

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
        
        for _, (data, target) in enumerate(self.train_loader):
            self.dimensions = self.convN.model.get_dims(data)
            self.dimensions['input'] = data.shape
            self.dimensions['target'] = (self.batch_size, self.num_classes) #target.shape but mapped to onehot
            break

        self.mineList = {}
    
        self.mi_values = defaultdict(list)
        for layer in ['maxP1','maxP2','relu3','sm1']:
            self.mineList[layer] = MINE(self.dimensions['input'], self.dimensions[layer], self.args).to(self.device)
            self.mineList[layer+'T'] = MINE(self.dimensions['target'], self.dimensions[layer], self.args).to(self.device)
            


    def trainMine(self, trainLoader, mine_epochs, batch_size, plot=False, convNet=None, target=False,\
                  mineMod=None,layer=None):
        model = mineMod.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        train_loader = trainLoader  # becomes unstable and biased for batch_size of 100
        # Train
        loss_list = []
        #print('\nMINE training, layer: %s, target: %s' % (layer, target))
        for mine_epoch in range(mine_epochs):
            loss_per_epoch = 0
            step = 0
            for batch_idx, (x, y) in tqdm(enumerate(train_loader)):
                x, y = x.to(self.device), y.to(self.device)
                model.zero_grad()
                if target:
                    tX = convNet(x, layer).detach()
                    y_onehot = torch.FloatTensor(y.shape[0], self.num_classes).to(self.device)
                    y_onehot.zero_()
                    y_onehot.scatter_(1, y.view(y.shape[0], 1), 1)
                    loss = model.lower_bound(y_onehot, tX)
                else:
                    tX = convNet(x, layer).detach()
                    loss = model.lower_bound(x, tX)
                loss_per_epoch += loss
                #print("Epoch MINE: %s. Lowerbound: %s" % (epoch, loss.item()))
                loss.backward()
                optimizer.step()
                
                #############
                '''
                NEED DELETE THIS
                Training not on full data to save development time
                '''
                step += 1
                if step == 100:
                    break
            #print('DONE')


            loss_list.append(-loss_per_epoch / len(train_loader))  # since pytorch can only minimize the return of mine is negative, we have to invert that again
            #if layer == 'sm1' and target:
            print('Epoch MINE: %s. Layer: %s. Target = %s. Lowerbound: %s' % (mine_epoch, layer, target, -loss_per_epoch.detach().cpu().numpy() / len(train_loader)))
        if plot:
            # Plot
            mine_epochs = np.arange(1, mine_epochs + 1)
            plt.figure(1)
            plt.plot(mine_epochs, loss_list)
            plt.legend(['MINE'])
            plt.ylabel('lower bound')
            plt.xlabel('epochs')
            plt.show()
        return loss_list[-1].detach().item()
        #return model

    def run(self, mine_path=None, save=False):
        '''
        TODO: make use of trained MINE estimators for visualization of LeNet training process
        '''
        for epoch in range(self.args.epochs):
            self.convN.train(self.train_loader, epoch)
            '''
            NEED TO UNCOMMENT EVERYTHING HERE
            '''
            #if mine_path == None:
            for layer in ['maxP1','maxP2','relu3','sm1']:
                #self.mineList[layer] = 
                self.mi_values[layer].append(self.trainMine(self.train_loader, self.mine_epochs, self.batch_size, plot=False, convNet=self.convN.model, mineMod=self.mineList[layer],target=False, layer=layer))
                #self.mineList[layer+'T'] = 
                self.mi_values[layer+'T'].append(self.trainMine(self.train_loader, self.mine_epochs, self.batch_size, plot=False, convNet=self.convN.model, mineMod=self.mineList[layer+'T'], target=True, layer=layer))
                '''
                TODO: Save trained MINE models
                '''
                if save == True:
                    #torch.save(self.mineList[layer].state_dict(), 'MINE_models/'+layer)
                    torch.save(self.mineList[layer+'T'].state_dict(), 'MINE_models/'+layer+'T')
            self.convN.test(self.test_loader)
            write_results(self.mi_values)
        return self.mineList, self.mi_values

def build_information_plane(MI, epochs):
    plt.figure(2)
    epochs = np.arange(1, epochs + 1)
    layers = ['maxP1','maxP2','relu3','sm1']
    print(MI)
    for layer in ['sm1']:#MI.keys():#layers:    
        plt.scatter(MI[layer], MI[layer+'T'])
    plt.legend(['MINE'])
    plt.ylabel('MI(T, Y)')
    plt.xlabel('MI(X, T)')
    plt.show()
    plt.savefig('information_plane.png')

def write_results(results, filename='mi_values_dict.pickle'):
    with open(filename, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                                help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                                help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=7, metavar='N',
                                help='number of epochs to train (default: 10)')
    parser.add_argument('--mine-epochs', type=int, default=25, metavar='N', #set to default 100
                                help='number of epochs to train MINE (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                                help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                                help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                                help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                                help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                                help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                                help='For Saving the current Model')
    parser.add_argument('--mine-path', type=str, default='MINE_models',
                                help='For Saving the current Model')
    args = parser.parse_args()
    args.mine_epochs

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")



    trackMI = TrackMI(args)#.to(device)
    mineList, mi_values = trackMI.run()
   
    #build_information_plane(mi_values, args.epochs)

if __name__ == '__main__':
    # freeze_support() here if program needs to be frozen
    main()

