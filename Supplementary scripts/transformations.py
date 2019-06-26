from torchvision import datasets, transforms
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from torchvision.transforms import functional as F
import numpy as np
from PIL import ImageOps


def main():
    trainSeperatePartition=False
    unevenLabels=False
    kwargs = {'num_workers': 8, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ])),
                batch_size=1, shuffle=True, **kwargs)
    fig = plt.figure(figsize=(8, 8))
    no=1
    for batch_idx, (x, y) in tqdm(enumerate(train_loader)):
        if trainSeperatePartition:
            if batch_idx%2==0:
                continue
        if unevenLabels:
            y=y%2
        trans=mirror(x)
        fig.add_subplot(4, 5, no).set_title("original")
        no+=1
        plt.imshow(x[0][0])
        fig.add_subplot(4, 5, no).set_title("transformed")
        plt.imshow(trans)
        no+=1

        if batch_idx>5:
            break
    plt.show()
def rotate(img):
    orig = transforms.ToPILImage()(img[0])
    trans=transforms.RandomRotation(degrees=[-180,180])(orig)

    print(transforms.ToTensor()(trans).shape)
    return transforms.ToTensor()(trans)[0]
def transpose(img):
    img=img[0][0]
    print(img.shape)
    img=img.t()
    return img
def random_grey(img):
    orig = transforms.ToPILImage()(img[0])
    trans = transforms.RandomGrayscale(p=0.3)(orig)

    return transforms.ToTensor()(trans)[0]
def mirror(img):
    orig = transforms.ToPILImage()(img[0])
    trans = ImageOps.mirror(orig)
    return transforms.ToTensor()(trans)[0]

if __name__ == '__main__':
# freeze_support() here if program needs to be frozen
    main()