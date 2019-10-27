import torch
import torch.nn as nn
import time
import os

import torch.nn.functional as F
from graphsage.neigh_samplers import UniformNeighborSampler
from graphsage.models import SampleAndAggregate

class SupervisedGraphSage(SampleAndAggregate):


    def __init__(self, features, train_adj, adj, train_deg, deg, sampler, n_samples, agg_layers, fc, multiclass=False, identity_dim=0):

        """        
        multiclass: if True: output go through sigmoid function and use binary cross entropy loss
                    if False: use cross entropy loss
        """

        super(SupervisedGraphSage, self).__init__(features, train_adj, adj, train_deg, deg, sampler, n_samples, agg_layers, fc, identity_dim)
        self.multiclass = multiclass
        if self.multiclass:
            self.loss_fn = nn.BCELoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        loss = self.loss_fn(scores, labels.squeeze())
        return loss

    def forward(self, nodes, mode='train', input_samples = None):
        """
        nodes: LongTensor
        """

        #Input mode different than current mode
        if mode != self.mode:
            if mode == 'train':
                self.sample_fn = UniformNeighborSampler(self.train_adj)
            else:
                self.sample_fn = UniformNeighborSampler(self.adj)

        if input_samples is not None:
            samples = input_samples
        else:
            samples = self.sample(nodes)
            
        out = self.aggregate(samples)
        out = F.normalize(out, dim=1)        

        if mode == 'save_embedding':
            return samples, out
        
        #Normalize
        out = self.fc(out)        

        if self.multiclass:
            out = F.sigmoid(out)
        return out
        
    