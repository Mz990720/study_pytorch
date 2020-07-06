import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # 1 input image  6 output 3*3 square conv
        self.conv1 = nn.Conv2d(1,6,3) 
        self.conv2 = nn.Conv2d(6,16,3)
        # an affine operation: y = Wx+b
        self.fc1 = nn.Linear(16*6*6,120) #维度是6
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        # Max Pooling over a (2,2)
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        #if is a square we can just use 2 to repalce (2,2)
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(-1,self.num_flat_features(x))  # -> 1*(size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self,x):
        size = x.size()[1:] # ? all dimensions except the batch dimension
        # pytorch only accept batch input so the whole dimension is 4 so [1:] we can turn the dimension to 3 
        # such as batch_number*4(number)*3*3 we use size()[1:]  -> (number)*3*3 so we focus on the real dimension
        num_features = 1
        for s in size:# (3,3,3)->3*3*3 = 27
            num_features *= s
        return num_features


net = Net()
#print(net)
params = list(net.parameters())
#print(len(params)) # 10 all the related params such as F,relu,
#for i in range(0,len(params)):
    #print(params[i].size())
#    torch.Size([6, 1, 3, 3])
#    torch.Size([6])  maybe demension 
#    torch.Size([16, 6, 3, 3])
#    torch.Size([16])
#    torch.Size([120, 102])
#    torch.Size([120])
#    torch.Size([84, 120])
#    torch.Size([84])
#    torch.Size([10, 84])
#    torch.Size([10])

input = torch.randn(1,1,32,32)
out = net(input)
target = torch.randn(10)
target = target.view(1,-1) # make the same shape as output
criterion = nn.MSELoss()
loss = criterion(out,target)

net.zero_grad()
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_( f.grad.data * learning_rate)

#


