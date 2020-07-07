#%%
import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import dataset
from oracle_prune import *
import argparse
from operator import itemgetter
from heapq import nsmallest
import time

#%%
# ModifiedVGG16Model
# Build a model modified by pretrained VGG16
# Used in training process
class ModifiedVGG16Model(nn.Module):
    def __init__(self, freeze=False, batch_norm=True, initial=True):
        super(ModifiedVGG16Model, self).__init__()
        
        model = models.vgg16(pretrained=True)
        self.features = model.features  # extract only feature extracter
        
        if batch_norm:
            x = torch.randn(1, 3, 32, 32)  # same size with training images
            layers = []
            for layer in self.features:
                layers += [layer]
                x = layer(x)
                if isinstance(layer, nn.modules.conv.Conv2d):
                    layers += [nn.BatchNorm2d(x.size(1))]  # input should be output's channel of previous layer
            self.features = nn.Sequential(*layers)
 
        # Whether freeze all params in conv layers or not
        for param in self.features.parameters():
            param.requires_grad = not freeze
        
        self.classifier = nn.Sequential(  # define new linear layers 
            # original channel: (25088, 4096)
            nn.Linear(512, 4096),  # input channels should match the output of feature extracter
            nn.ELU(inplace=True), 
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ELU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 100))
        
        if initial:
            for layer in self.features.named_parameters():
                if isinstance(layer, nn.modules.conv.Conv2d):
                    nn.init.xavier_normal_(layer, gain=1)
            for layer in self.classifier.named_parameters():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer, gain=1)
                # print(layer[0])
                # if 'weight' in layer[0]:
                #     print(type(layer[1]))
                    # nn.init.xavier_normal_(layer[1], gain=1)
            # for layer in self.classifier.named_parameters():
            #     # print(layer[0])
            #     if 'weight' in layer[0]:
            #         pass
            #         nn.init.xavier_normal_(layer[1], gain=1)


         
    def forward(self, x):
        x = self.features(x)  # filters
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)  # classifier
        return x
#%%            
# region count conv layers output size

# model = models.vgg16(pretrained=True)  
# features = model.features  # extract only feature extracter

# gar02 = features(gar01)
# gar02.shape
# endregion

#%%
# Build the processes to prune filters
class FilterPrunner:
    def __init__(self, model):
        self.model = model  # trained model
        self.reset()  # create a empty filter_ranks dict 
        
    def reset(self):
        self.filter_ranks = {}  # store rank of filters (talyor) of each layer
        # layer_index: vector of ranks
    
    # execute forward and backward process to get activation and gradient values
    # in order to compute taylor
    def forward(self, x):
        '''Collect activations and gradients of each layer.
        
        Process:
            1. input x and to a single layer to get activations and append to self.activations
            2. get gradients of the layer and time with activation to get taylor
            3. compute rank of filters and store in self.filter_ranks
        
        '''
        self.activations = []  # store activation
        self.activation_to_layer = {}  # store corresponding layer index of the activations
    
        grad_index = 0
        activation_index = 0
        
        # 1. _modules.items() fetchs single layer from feature extracter
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            x = module(x)  # input x to a single layer and get activation
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                x.register_hook(self.compute_rank)  # 2. insert hook function to compute rank
                self.activations.append(x)
                self.activation_to_layer[activation_index] = layer  # layer index of the activations
                activation_index += 1
        
        return self.model.classifier(x.view(x.size(0), -1))
        
    def compute_rank(self, grad):
        '''Compute rank of filters of each layer and store them in self.filter_ranks.'''
        activation_index = len(self.activations) - grad_index - 1  # the last activation goes with the first gradient
        activation = self.activations[activation_index]
        
        taylor = activation * grad  # 4D matrix
        taylor = taylor.mean(dim=(0, 2, 3)).data  # it's the rank of filters in a layer  (reduce 0,2,3th dim)
        
        # 3. 
        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = \
                torch.FloatTensor(activation.size(1)).zero_()  # create "1st" dim 0 vector to store taylor
            self.filter_ranks[activation_index] = self.filter_ranks[activation_index].cuda()
                
        self.filter_ranks[activation_index] += taylor
        grad_index += 1
        
    def lowest_ranking_filters(self, num):
        '''Return "num" of the lowest ranking filters in all layers.'''
        data = []  # store all the rank of filters with layer and channel index
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))
        
        # get lowest ranking filters (tuple) sorted by self.filter_ranks[i][j]
        return nsmallest(num, data, itemgetter(2))  # key:itemgetter(2) = self.filter_ranks[i][j]

    def normalize_ranks_per_layer(self):
        '''Normalize the rank(mean) of filters by sum of sqrt'''
        for i in self.filter_ranks:
            self.filter_ranks[i] = self.filter_ranks[i].cpu()
            v = torch.abs(self.filter_ranks[i])  # vector of ranks
            v = v / np.sqrt(torch.sum(v * v))  # root sum of square
            
    def get_prunning_plan(self, num_filters_to_prune):
        '''Store tuple (l,i) of filters to prune in a list, cosidering the index change after each prunig step.'''
        # After each of the k filters are prunned,
        # the filter index of the next filters change since the model is smaller.
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)
        filters_to_prune_per_layer = {}
        # Set layer as key and list of filter index as value
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:  # if layer not in dict yet
                filters_to_prune_per_layer[l] = []  # list of filter index
            filters_to_prune_per_layer[l].append(f)  # add index of filters_to_prune to list 

        # Change the filter index of the next filters
        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
            for i in range(len(filters_to_prune_per_layer[l])):
                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

        # Collect index tuple (l,i) of filter to prune
        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))

        return filters_to_prune 

# region testing
# x = torch.randn(1, 3, 32, 32)
# for layer, (name, module) in enumerate(model.features._modules.items()):
#     print(layer)
#     print(name)
#     print(module)
#     x = module(x)
#     print(x.shape)
#     break

# x.register_hook()

# gar = {2: 'a', 3: 'b', 1: 'c'}
# sorted(gar.keys())
# endregion

#%%
# PrunningFineTuner
# structure
# fine_tuner = PrunningFineTuner_VGG16
#     prunner = FilterPrunner(model)
#     fine_tuner.train()
#         train_epoch(rank_filters)
#             train_batch(rank_filters, batch, label)
#                 output = prunner.forward(input)
                        compute_rank
#     fine_tuner.prune()

class PrunningFineTuner_VGG16:
    # Load datasets and pre-trained model
    # then training the model and test it first
    def __init__(self, train_loader, test_loader, model):
        self.train_data_loader = train_loader
        self.test_data_loader = test_loader
        
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.prunner = FilterPrunner(self.model)
        self.model.train()  # train the original model first
    
    # Test the trained model
    def test(self):
        correct = 0
        total = 0
        
        self.model.eval()
        for i, (batch, label) in enumerate(self.test_data_loader):
            # if args.use_cuda:
            #     batch = batch.cuda()
            batch = batch.cuda()
            output = self.model(batch)
            pred = output.data.max(1)[1]
            correct += pred.cpu().eq(label).sum()
            total += label.size(0)
            
        print("ValidAccu :", float(correct) / total)
    
    # Fine Tuned the pruned model and test it
    def train(self, optimizer=None, epoches=10):
        if optimizer is None:
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005, nesterov=True)
            print('No optimizer is assigned, use SGD instead.')
            
        for i in range(epoches):
            print("Epoch: ", i)
            self.train_epoch(optimizer)
            self.test()
        print('Finished Training.')
            
    # Iterations
    def train_batch(self, optimizer, batch, label, rank_filters):
        # if args.use_cuda:
        #     batch = batch.cuda()
        #     label = label.cuda()
        batch = batch.cuda()
        label = label.cuda()
        self.model.zero_grad()
        input = Variable(batch)

        if rank_filters:  # then prune the model
            output = self.prunner.forward(input)
            self.criterion(output, Variable(label)).backward()
        else:  # train original model
            self.criterion(self.model(input), Variable(label)).backward()
            optimizer.step()
                    
    def train_epoch(self, optimizer=None, rank_filters=False):
        for i, (batch, label) in enumerate(self.train_data_loader):
            self.train_batch(optimizer, batch, label, rank_filters)
    
    # Return layer_index and filter_index
    def get_candidates_to_prune(self, num_filters_to_prune):
        self.prunner.reset()
        self.train_epoch(rank_filters = True)
        self.prunner.normalize_ranks_per_layer()
        return self.prunner.get_prunning_plan(num_filters_to_prune)
    
    def total_num_filters(self):
        filters = 0
        for name, module in self.model.features._modules.items():
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                filters = filters + module.out_channels
        return filters

    def prune(self, optimizer=None, rate=0.5, pruned_per_iter=512, fine_tuned_iter=5):
        # Get the accuracy before prunning
        self.test()

        self.model.train()  # training mode
        # Make sure all the (conv) layers are trainable
        for param in self.model.features.parameters():
            param.requires_grad = True

        number_of_filters = self.total_num_filters()
        num_filters_to_prune_per_iteration = pruned_per_iter
        iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)  # num of iter to prune all filters

        iterations = int(iterations * rate)  # compute how many iterations it needs to prune the filters
        print("Number of filters to prune: {}".format(iterations * pruned_per_iter))
        print("Pruning iterations: {},reduce {}% filters totally.".format(iterations ,round(rate*100, 2)))
        
        # Collect layer_index and filter_index to prune
        for _ in range(iterations):
            print("Ranking filters.. ")
            prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration)
            layers_prunned = {}
            for layer_index, filter_index in prune_targets:
                if layer_index not in layers_prunned:
                    layers_prunned[layer_index] = 0
                layers_prunned[layer_index] = layers_prunned[layer_index] + 1 

            print("Layers that will be prunned", layers_prunned)
            print("Prunning filters.. ")
            model = self.model.cpu()
            # Create a compact pruned model
            for layer_index, filter_index in prune_targets:
                model = prune_vgg16_conv_layer(model, layer_index, filter_index, use_cuda=True)
                # model = prune_vgg16_conv_layer(model, layer_index, filter_index, use_cuda=args.use_cuda)

            # if args.use_cuda:
            #     self.model = self.model.cuda()
            self.model = self.model.cuda()
            
            filters_left = round(100*float(self.total_num_filters()) / number_of_filters, 2)
            print("Filters left: ", filters_left, '%')
            self.test()  # test the pruned model before fine-tuning
            
            print("Fine tuning to recover from prunning iteration.")
            print('Use SGD as optimizer while fine-tuning.')
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005, nesterov=True)
            self.train(optimizer, epoches=fine_tuned_iter)  # fine tune the pruned model
            self.test()

            # lr_decay = {'0.01': 2, '0.001': 2}  # learning rate decay
            # for key, val in lr_decay.items():
            #     optimizer = optim.SGD(model.parameters(), lr=float(key), momentum=0.9, weight_decay=0.0005, nesterov=True)
            #     self.train(optimizer, epoches=int(val))  # fine tune the pruned model
            #     self.test()

        print("Finished.")
        # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005, nesterov=True)
        # self.train(optimizer, epoches=1)




#%%
# Save the Model
# torch.save(net.state_dict(), './model/LeNet5_origin01.pkl')
