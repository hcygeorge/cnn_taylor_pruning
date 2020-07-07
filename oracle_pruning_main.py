# refer https://jacobgil.github.io/deeplearning/pruning-deep-learning

#%% [markdown]
# Lenet5 range:(0,1)
# optim:Adam, lr:0.001, decay:5e-4, batch:128, epochs:20, accu:42%
# optim:Adam, lr:0.001, decay:5e-4, batch:128, epochs:20, batch_norm, accu:45%
# optim:SGD, lr:lr_decay, decay:5e-4, batch:128, epochs:150, batch_norm, accu:48%
# optim:SGD, xavier, lr:lr_decay, decay:5e-4, batch:128, epochs:150, batch_norm, accu:48%

#%%
# general
# check directory
import os
os.chdir('C:/works/PythonCode/ModelCompression/TaylorPruning')
os.getcwd()

# data science tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
torch.manual_seed(7)  # seed
# pre-build dataset
from torchvision import models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as Data
# check torch version and cuda
print('torch version :' , torch.__version__)
print('cuda available :' , torch.cuda.is_available())
print('cudnn enabled :' , torch.backends.cudnn.enabled)

# from prune import *
import argparse
from operator import itemgetter
from heapq import nsmallest
import time

from oracle_fine_tune import  ModifiedVGG16Model, PrunningFineTuner_VGG16
from oracle_utils import WarmUpLR
#%%
# CIFAR100 Dataset
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
# transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) cifar10
training_batch_size = 128
testing_batch_size = 128

train_dataset = datasets.CIFAR100(root='./data', 
                            train=True, 
                            transform=transform,  # to PIL or Tensor
                            download=True)  # no action if already downloaded

test_dataset = datasets.CIFAR100(root='./data',
                            train=False,  # ask for test dataset
                            transform=transform,
                            download=True)

# Dataset Loader (Input Pipline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=training_batch_size,  # batch training size
                                           shuffle=True,  # in each epochs
                                           drop_last=False,  # drop last batch if incomplete
                                           pin_memory=True)  # allocate the data in page-locked memory, which speeds-up the transfer to GPU memory

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=testing_batch_size, 
                                          shuffle=False,
                                          pin_memory=True)
#%%
# region training from scratch
#%%
# Display predictions of images

# for i in range(0,5):
#     # outputs = net(test_dataset.data[i].float())
#     # prob, predicted = torch.max(outputs, 1)
#     plt.imshow(test_dataset.data[i].numpy(), cmap='gray')
#     # plt.title('Prediction:%i   Probability:%.4f' % (test_dataset.targets[i].item(), prob.item()))
#     plt.show()

#%%
model = ModifiedVGG16Model(freeze=False, batch_norm=False, initial=True)
# model = torch.load("model", map_location=lambda storage, loc: storage)
model = model.cuda()

fine_tuner = PrunningFineTuner_VGG16(train_loader, test_loader, model)

#%%
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()

# optimizer = optim.SGD(model.parameters(), lr=0.0008, momentum=0.9, weight_decay=0.0005)
# optimizer = optim.Adam(model.parameters(),
#                       lr=learning_rate,
#                       weight_decay=decay)  

#%%
# training
start = time.time()
lr_decay = {0.002: 1, 0.01: 20, 0.001: 20, 0.0001: 10, 0.00001: 10}  # learning rate decay
for key, val in lr_decay.items():
    optimizer = optim.SGD(model.parameters(), lr=float(key), momentum=0.9, weight_decay=0.0005, nesterov=True)
    fine_tuner.train(optimizer, epoches=int(val))

end = time.time()
print('Training Time:', round((end - start)/60, 2), 'mins')


#%%
# Save trained model
torch.save(model.state_dict(), './model/VGG16_origin02_no_batch_norm.pkl')

#%%
# Load trained model
model = ModifiedVGG16Model(freeze=False, batch_norm=False, initial=True)
model.load_state_dict(torch.load('./model/VGG16_origin02_no_batch_norm.pkl'))
model.cuda()

#%%
# Load pruned model
# pruned = torch.load('./model/VGG16_prune02_no_batch_norm.pkl')
# model.cuda()
#%%
# pruning
start = time.time()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005, nesterov=True)
fine_tuner.prune(optimizer, rate=0.6, fine_tuned_iter=1)

end = time.time()
print('Pruning Time:', round((end - start)/60, 2), 'mins')

torch.save(fine_tuner.model, './model/VGG16_prune02_no_batch_norm.pkl') 


#%%
# Compute model inference time

model = model.cpu()
pruned = pruned.cpu()
model.eval(), pruned.eval()


test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=16, 
                                          shuffle=False,
                                          pin_memory=True)

#%%
# original model inference time
start = time.time()
for i, (batch, label) in enumerate(test_loader):
    output = model(batch)
end = time.time()

print('Inference Time:', round((end - start), 2), 'secs')
fps = round(test_dataset.data.shape[0] / (end - start), 2)
print('FPS:', fps, 'fps')

# pruned model inference time
start = time.time()
for i, (batch, label) in enumerate(test_loader):
    output = pruned(batch)
end = time.time()

print('Inference Time:', round((end - start), 2), 'secs')
fps = round(test_dataset.data.shape[0] / (end - start), 2)
print('pruned FPS:', fps, 'fps')

#%%
start = time.time()
for i, (batch, label) in enumerate(test_loader):
    output = pruned(batch)
end = time.time()

print('Inference Time:', round((end - start), 2), 'secs')
fps = round(test_dataset.data.shape[0] / (end - start), 2)
print('FPS:', fps, 'fps')
#
start = time.time()
for i, (batch, label) in enumerate(test_loader):
    output = model(batch)
end = time.time()

print('Inference Time:', round((end - start), 2), 'secs')
fps = round(test_dataset.data.shape[0] / (end - start), 2)
print('pruned FPS:', fps, 'fps')

#%%
# Before pruning
# ValidAccu :0.6877
# Input Size: ~1.57MB
# Parameters Size:136.03MB
# Intermediate Size:86.61MB
# Total Model Size:224.21MB
# FPS on CPU: 97.47 (batch 10000)
# FPS on CPU: 140.82 (batch 16)

# During pruning
# ValidAccu :0.2302

# After pruning
# ValidAccu :0.6418
# Input Size: ~1.57MB
# Parameters Size:86.84MB
# Intermediate Size:51.88MB
# Total Model Size:140.29MB
# FPS on CPU: 108.42 (batch 10000)
# FPS on CPU: 199.22 (batch 16)

# speed up: 1.11~1.41x