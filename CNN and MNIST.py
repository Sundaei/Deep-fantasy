import torch as tc
import torch.nn as nn
inputs=tc.Tensor(1,1,28,28)
print('텐서의 크기:{}'.format(inputs.shape))

conv1=nn.Conv2d(1,32,3,padding=1)
print(conv1)
conv2=nn.Conv2d(32,63,kernel_size=3,padding=1)
print(conv2)
pool=nn.MaxPool2d(2)
print(pool)

out=conv1(inputs)
print(out.shape)
out=pool(out)
print(out.shape)

out=conv2(out)
print(out.shape)
