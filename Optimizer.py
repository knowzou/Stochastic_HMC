import random
import numpy as numpy
import torch
from torch.autograd import grad as torchgrad

class Gradient(object):
    def __init__(self, data, batch_size, svrg_epoch_length=100, random_seed=1134):
        self.data = data
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.n = data.shape[0]
        self.p = data.shape[1]
        self.svrg_epoch_length = svrg_epoch_length
        self.svrg_count = 0
        self.epoch_count = 0
        
        
        
    def get_batch(self): ### get a batch of data
        batch_inds = random.sample(range(self.n), self.batch_size)
        return self.data[batch_inds, :], batch_inds
    
    def SG(self, energy, param, param_ref):
        batch_data, _ = self.get_batch()    
        return self.n*torchgrad(energy(param, batch_data).sum(),param)[0]
    
    def CVG(self, energy, param, param_ref):
        if self.epoch_count == 0:
            self.cvg_grad_ref = torchgrad(energy(param_ref, self.data).sum(),param_ref)[0]
            
        batch_data, _ = self.get_batch()
        sto_grad = torchgrad(energy(param, batch_data).sum(),param)[0]
        sto_grad_ref = torchgrad(energy(param_ref, batch_data).sum(),param_ref)[0]
   
        self.epoch_count += 1;
        return self.n*(sto_grad - sto_grad_ref + self.cvg_grad_ref)
    
    def SVRG(self, energy, param, param_ref):
        if self.epoch_count%self.svrg_epoch_length  == 0:
            self.svrg_grad_ref = torchgrad(energy(param, self.data).sum(),param)[0].clone()
            self.svrg_param_ref = param.clone()
            
        
        batch_data, _ = self.get_batch()
        sto_grad = torchgrad(energy(param, batch_data).sum(),param)[0]
        sto_grad_ref = torchgrad(energy(self.svrg_param_ref, batch_data).sum(),self.svrg_param_ref)[0]
   
        self.epoch_count += 1;
        return self.n*(sto_grad - sto_grad_ref + self.svrg_grad_ref)
    
    def SAG(self, energy, param, param_ref):
        if self.epoch_count == 0:

            
            self.grad_data = torch.stack([torchgrad(energy(param, self.data[[i],:]).sum(),param)[0] \
                                                for i in range(self.n)]) 
            self.grad_mean = self.grad_data.mean(axis=0)
            
        batch_data, batch_inds = self.get_batch()
        sto_grads = torch.stack([torchgrad(energy(param, self.data[[i],:]).sum(),param)[0] \
                                                for i in batch_inds])
        diff_grad = (sto_grads - self.grad_data[batch_inds,:]).mean(axis=0)
        sag_grad = diff_grad + self.grad_mean
        
        
        ############# update table
        self.grad_data[batch_inds,:] = sto_grads
        self.grad_mean = self.grad_mean +  diff_grad * self.batch_size/self.n
        
        self.epoch_count += 1
        
        return self.n * sag_grad
        