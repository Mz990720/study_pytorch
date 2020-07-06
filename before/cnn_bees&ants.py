import numpy as np
import torchvision
from torchvision import datasets,transforms,models

import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import os
import copy


# Top level data directory. Here we assume the format of the directory conforms 
#   to the ImageFolder structure
data_dir = "./hymenoptera_data"
# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"
# Number of classes in the dataset
num_classes = 2
# Batch size for training (change depending on how much memory you have)
batch_size = 32
# Number of epochs to train for 
num_epochs = 15
# Flag for feature extracting. When False, we finetune the whole model, 
#   when True we only update the reshaped layer params
feature_extract = True

input_size = 224

all_imgs = datasets.ImageFolder(os.path.join(data_dir, "train"), transforms.Compose([
        transforms.RandomResizedCrop(input_size), #随便截区域 
        transforms.RandomHorizontalFlip(), #左右翻转 相当于数据填充
        transforms.ToTensor(),
    ]))
loader = torch.utils.data.DataLoader(all_imgs, batch_size=batch_size, shuffle=True, num_workers=0)

data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size), #测试的时候就中间取块
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ["train", "val"]}

dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], 
        batch_size=batch_size, shuffle=True, num_workers=0) for x in ["train", "val"]}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


img = next(iter(loader))[0]
print(img.shape)

unloader = transforms.ToPILImage()  # reconvert into PIL image 显示图片

plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title) 
    plt.pause(20) # pause a bit so that plots are updated


#plt.figure()
#imshow(img[0], title='Image')   #显示一张图片 

def set_parameter_requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():  # feature extract 只更改最后fc层的参数
            param.requires_grad = False 

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    if model_name == "resnet":
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes) #改变最后fc层的结构
        input_size = 224
    else:
        print("model not implemented")
        return None, None
    return model_ft, input_size

model_ft, input_size = initialize_model(model_name, 
                    num_classes, feature_extract, use_pretrained=True)

#print(model_ft) #显示模型的参数
#print(model_ft.fc.weight.requires_grad) #只有最后一个fc层改变

 #two dataloaders : one for train,one for eval    
def train_model(model, dataloaders, loss_fn, optimizer, num_epochs=2):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.
    val_acc_history = []
    for epoch in range(num_epochs):
        for phase in ["train","val"]:
            running_loss = 0.
            running_corrects = 0.

            #这里调整模式 训练和测试的时候网络有所不同 batch normalization & dropout
            if phase == "train":
                model.train()
            else:
                model.eval()
            
            for inputs,lables in dataloaders[phase]:
                inputs,lables = inputs.to(device),lables.to(device)

                with torch.autograd.set_grad_enabled(phase=="train"):
                    outputs = model(inputs)  #batch_size * 2
                    loss = loss_fn(outputs,lables)
                preds = outputs.argmax(dim=1)
                if phase == "train":
                    optimizer.zero_grad() #之前参数清零
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item()*inputs.size(0) 
                running_corrects += torch.sum(preds.view(-1)==lables.view(-1)).item()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            print("Times {} Phase {} loss: {},acc: {}".format(epoch,phase,epoch_loss,epoch_acc))

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "val":
                val_acc_history.append(epoch_acc)
    model.load_state_dict(best_model_wts)
    return model,val_acc_history,best_acc

model_ft = model_ft.to(device)
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,model_ft.parameters()),lr=0.001,momentum=0.9)
loss_fn = nn.CrossEntropyLoss()

_, ohist,the_acc = train_model(model_ft, dataloaders_dict, loss_fn, optimizer, num_epochs=num_epochs)
print(ohist)
print(the_acc)


   