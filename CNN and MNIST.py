import torch as tc
import torch.nn as nn
inputs=tc.Tensor(1,1,28,28)
print('텐서의 크기:{}'.format(inputs.shape))

conv1=nn.Conv2d(1,32,3,padding=1)
print(conv1)
conv2=nn.Conv2d(32,64,kernel_size=3,padding=1)
print(conv2)
pool=nn.MaxPool2d(2)
print(pool)

out=conv1(inputs)
print(out.shape)
out=pool(out)
print(out.shape)

out=conv2(out)
print(out.shape)

out=pool(out)
print(out.shape)

out=out.view(out.size(0),-1)
print(out.shape)

fc=nn.Linear(3136,10)
out=fc(out)
print(out.shape)

import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
device='cuda'if torch.cuda.is_available() else 'cpu'

tc.manual_seed(777)
if device=='cuda':
    torch.cuda.manual_seed(777)
learning_rate=0.001
training_epochs=15
batch_size=100

mnist_train=dsets.MNIST(root='MNIST_data/',train=True,transform=transforms.ToTensor(),download=True)
mnist_test=dsets.MNIST(root='MNIST_data/',train=False,transform=transforms.ToTensor(),download=True)
data_loader=tc.utils.data.DataLoader(dataset=mnist_train,batch_size=batch_size,shuffle=True,drop_last=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1),nn.ReLU(),nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer2=nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),nn.ReLU(),nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.fc=nn.Linear(7*7*64,10,bias=True)
        nn.init.xavier_uniform_(self.fc.weight)
    def forward(self,x):
        out=self.layer1(x)
        out=self.layer2(out)
        out=out.view(out.size(0),-1)
        out=self.fc(out)
        return out

model=CNN().to(device)
criterion=nn.CrossEntropyLoss().to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

total_batch=len(data_loader)
print('총 배치의 수:{}'.format(total_batch))

for epoch in range(training_epochs):
    avg_cost=0

    for X,Y in data_loader:
        X=X.to(device)
        Y=Y.to(device)

        optimizer.zero_grad()
        hypothesis=model(X)
        cost=criterion(hypothesis,Y)
        cost.backward()
        optimizer.step()

        avg_cost+=cost/total_batch
    print('[Epoch: {:>4}] cost={:>.9}'.format(epoch+1,avg_cost))