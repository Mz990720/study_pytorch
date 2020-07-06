import torch.nn as nn
import torch

dtype = torch.float
device = torch.device("cuda:0")
N, D_in, H, D_out = 64,1000,100,10

x = torch.randn(N,D_in,device=device,dtype=dtype)
y = torch.randn(N,D_out,device=device,dtype=dtype)

class TwoLayerNet(torch.nn.Module):
    def __init__(self,D_in,H,D_out):
        super(TwoLayerNet,self).__init__()
        self.linear1 = torch.nn.Linear(D_in,H,bias=False)
        self.linear2 = torch.nn.Linear(H,D_out,bias=False)
        
    def forward(self,x):
        y_pred = self.linear2(self.linear1(x).clamp(min=0))
        return y_pred
        
#w1 = torch.randn(D_in,H,device=device,dtype=dtype,requires_grad=True)
#w2 = torch.randn(H,D_out,device=device,dtype=dtype,requires_grad=True)

model = TwoLayerNet(D_in,H,D_out)
model = model.cuda()
#torch.nn.init.normal_(model[0].weight) #init effects the performance
#torch.nn.init.normal_(model[2].weight)

#损失函数和梯度下降算法
loss_fn = nn.MSELoss(reduction='sum')
learing_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(),lr=learing_rate)

for it in range(500):
    #Forward pass
    #h = x.mm(w1) # N*H
    #h_relu = h.clamp(min=0) 
    #y_pred = h_relu.mm(w2)
    #y_pred = x.mm(w1).clamp(min=0).mm(w2)
    y_pred = model(x)

    #compute loss 
    loss = loss_fn(y_pred,y)  #computation graph
    print(it,loss.item())

    optimizer.zero_grad()
    #Backwared auto compute the gradient
    loss.backward()

    # update model parameters
    optimizer.step() #一步更新 
