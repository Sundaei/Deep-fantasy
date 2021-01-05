import torch as tc
import torchvision as tvs
import torchvision.transforms as transforms

transform=transforms.Compose(
    [transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

trainset=tvs.datasets.CIFAR10(root='./data',train=True, download=True, transform=transform)
trainloader=tc.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)

testset=tvs.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
testloader=tc.utils.data.DataLoader(testset,batch_size=4,shuffle=False,num_workers=2)

classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img=img/2+0.5
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

dataiter=iter(trainloader)
images,labels=dataiter.next()

imshow(tvs.utils.make_grid(images))
print(''.join('%5s'%classes[labels[j]]for j in range(4)))