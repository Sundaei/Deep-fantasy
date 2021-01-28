import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy

device='cuda'if tc.cuda.is_available() else 'cpu'
tc.manual_seed(777)
if device=='cuda':
    tc.cuda.manual_seed_all(777)

X=tc.FloatTensor([[0,0],[0,1],[1,0],[1,1]]).to(device)
Y=tc.FloatTensor([[0],[1],[1],[0]]).to(device)
linear=nn.Linear(2,1,bias=True)
sigmoid=nn.Sigmoid()
model=nn.Sequential(linear,sigmoid).to(device)

criterion=tc.nn.BCELoss().to(device)
optimizer=tc.optim.SGD(model.parameters(),lr=1)

for step in range(10001):
    optimizer.zero_grad()
    hypothesis=model(X)

    cost=criterion(hypothesis,Y)
    cost.backward()
    optimizer.step()

    if step %100==0:
        print(step,cost.item())
with tc.no_grad():
    hypothesis=model(X)
    predicted=(hypothesis>0.5).float()
    accuracy=(predicted==Y).float().mean()
    print('모델 출력값(hypothesis):',hypothesis.detach().cpu().numpy())
    print('모델 예측값(predicted):',predicted.detach().cpu().numpy())
    print('실제값(Y):',Y.cpu().numpy())
    print('정확도(accuracy):',accuracy.item())
