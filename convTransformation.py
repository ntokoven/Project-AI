import torch
import torch.functional as F
import torch.nn as nn

def change_shape(x,layer):
    print("orig",x.shape,layer.shape)
    if x.shape==layer.shape:
        return x
    elif x.dim() == layer.dim():
        if x.dim()==4:
            x=nn.Conv2d(1,1,(x.shape[2]-layer.shape[2]+1,x.shape[2]-layer.shape[2]+1))(x)
            if x.shape[1]!= layer.shape[1]:
                noRepeat=layer.shape[1]-x.shape[1]+1
                x=x.repeat(1,noRepeat,1,1)
        else:
            layer=nn.Linear(layer.shape[1],x.shape[1])(layer)
    else:
        if x.dim()>layer.dim():
            x=x.reshape(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3])
            x=nn.Linear(x.shape[1],layer.shape[1])(x)
        else:
            layer=layer.reshape(layer.shape[0],layer.shape[1]*layer.shape[2]*layer.shape[3])
            layer=nn.Linear(layer.shape[1],x.shape[1])(layer)
    print("transformed",x.shape,layer.shape)
    return x,layer


dims= {'maxP1': torch.Size([64, 20, 12, 12]), 'maxP2': torch.Size([64, 50, 4, 4]),
       'relu3': torch.Size([64, 500]), 'sm1': torch.Size([64, 10]),
       'input': torch.Size([64, 1, 28, 28]), 'target': torch.Size([64,10])}
layer1=torch.Tensor(64, 20, 12, 12)
layer2=torch.Tensor(64, 50, 4, 4)
layer3=torch.Tensor(64, 500)
layer4=torch.Tensor(64, 10)
x=torch.Tensor(64,1,28,28)
y=torch.Tensor(64,10)

a=change_shape(x,layer1)
a=change_shape(x,layer2)
a=change_shape(x,layer3)
a=change_shape(x,layer4)
a=change_shape(y,layer1)
a=change_shape(y,layer2)
a=change_shape(y,layer3)
a=change_shape(y,layer4)