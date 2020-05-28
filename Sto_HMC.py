import matplotlib.pyplot as plt
from torch.autograd import grad as torchgrad
import numpy as np
import torch
from Optimizer import Gradient


def SGHMC(energy, data, batch_size, initial, out_epochs, in_epochs, lr, device, optimizer, enable_MH=True):
    checkpoints = []
    
    
    opt = Gradient(data, batch_size)
    param_ref = initial
    if optimizer == "SG":
        grad = opt.SG
    elif optimizer == "CVG":
        grad = opt.CVG
    elif optimizer == "SVRG":
        grad = opt.SVRG
    elif optimizer == "SAG":
        grad = opt.SAG

    n = data.shape[0]
    q = initial
    torch.set_grad_enabled(True)
    g = grad(energy, q.requires_grad_(), param_ref)
    E = energy(q, data)*n
    print (g)
    torch.set_grad_enabled(False)
    
    for _out in range(out_epochs):
        p = torch.randn(q.shape).to(device)
        H = 0.5*(p**2).sum(axis=1) + E
        q_new = q.detach()
        g_new = g.detach()
        
        for _ in range(in_epochs):
            p = p - lr * g_new/2.
            q_new = (q_new.detach() + lr * p)
            
            torch.set_grad_enabled(True)
            g_new = grad(energy, q.requires_grad_(), param_ref)
            torch.set_grad_enabled(False)
            
            p = p- lr * g_new.detach()/2.

            E_new = energy(q_new, data)*n
            H_new = 0.5*(p**2).sum(axis=1) + E_new

        
        if enable_MH:
            diff = H - H_new
            accept = diff.exp() > diff.uniform_()
            accept = accept.float()
            E = accept*E_new + (1 - accept)*E

            q = q_new.T*accept + q.T*(1 - accept)
            q = q.T
            g = g_new.T*accept + g.T*(1 - accept)
            g = g.T
        else:  
            q = q_new
            g = g_new
            
        if _out%10 == 0:
            checkpoints += [q.clone().detach()]
        
    torch.set_grad_enabled(True)

    return checkpoints