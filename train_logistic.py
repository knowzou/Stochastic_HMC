
import pandas as pd
import random
import numpy as np
import torch
from torch.autograd import grad as torchgrad
from Sto_HMC import SGHMC
import argparse
import time
from Optimizer import Gradient

parser = argparse.ArgumentParser(description='train stochastic HMC')
parser.add_argument('--sample_num', type=int, default = 2000,
                    help='number of generated samples')
parser.add_argument('--data_num', type=int, default = 500,
                    help='number of data')
parser.add_argument('--data_set',  default = "Pima",
                    help='number of training data')
parser.add_argument('--energy_func', default = "diff_mean",
                    help='type of target density')
parser.add_argument("--out_epochs", type=int, default = 2000,
                    help = "outer epochs")
parser.add_argument("--in_epochs", type=int, default = 100,
                    help = "inner epochs")
parser.add_argument("--lr", type=float, default = 2e-4,
                    help = "learning rate")
parser.add_argument("--optimizer", default = "SG",
                    help = "optimizer")
parser.add_argument("--batch_size", type=int, default = 8,
                    help = "batch size")
parser.add_argument("--prior", type=float, default = 5,
                    help = "batch size")
parser.add_argument("--enable_MH", type=bool, default = False,
                    help = "enable metropolis hasting")
args = parser.parse_args()
print (args)


if torch.cuda.is_available():
    device = "cuda" 
else: 
    device="cpu"


def logistic(q, data):
    
    n, d = data.shape
    data =  torch.from_numpy(data).float().to(device)
    output = torch.matmul(data, q.T)
    all_energy = (torch.log(1 + torch.exp(-output))).mean(axis=0)
    
    return all_energy

if args.data_set == "Pima":
    data = pd.read_csv("data/pima.csv").to_numpy()
    n, d = data.shape
    feature, label = data, data[0:, [-1]]
    feature[:, -1] = np.ones([n])
    feature = feature / feature.max(axis=0)
    label = (label - 1/2)*2
    feature = feature * label
    X = feature
    beta = torch.from_numpy(X.mean(axis=0)).float().to(device).unsqueeze(0)
    opt = Gradient(X, n)
    eta = 0.001
    for _ in range(10000):
        beta = beta - eta*args.prior*2*beta  - eta*opt.SG(logistic, beta.requires_grad_(), True)
    initial = beta.repeat(args.sample_num, 1)


if args.data_set == "Covtype":
    if args.data_num == 500 or args.data_num == 500 or args.data_num == 5000:
        file_name = "data/covtype" + str(args.data_num) + ".npy"
        data = np.load(file_name)
    else:
        data = np.load("data/covtypeall.npy")
        rand_ind = random.sample(range(data.shape[0]), args.data_num)
        data = data[rand_ind,:]

    n, d = data.shape
    print(n, d)
    feature, label = data, data[0:, [-1]]
    feature[:, -1] = np.ones([n])
    feature = feature / (np.abs(feature).max(axis=0)+0.01)
    feature = feature * label
    X = feature
    beta = torch.from_numpy(X.mean(axis=0)).float().to(device).unsqueeze(0)
    opt = Gradient(X, 500)
    eta = 0.0001
    for _ in range(10000):
        beta = beta - eta*args.prior*2*beta  - eta*opt.SG(logistic, beta.requires_grad_(), True)
    initial = beta.repeat(args.sample_num, 1) -0.05

    print (initial)


batch_size = n if args.enable_MH else args.batch_size

initial.requires_grad_(True)
start = time.time()
log, output = SGHMC(energy = logistic, 
                data = X,
                batch_size = batch_size,
                initial = initial,
                out_epochs = args.out_epochs,
                in_epochs = args.in_epochs,
                lr = args.lr, 
                device = device,
                optimizer = args.optimizer,
                prior = args.prior,
                enable_MH = args.enable_MH
                 )  
print (args.optimizer, " Run time: ", time.time()-start)
res = []
for _res in log:
    res += [_res]

save_name = "result/" + args.data_set + "_" + args.optimizer + "_batch_size_" + str(args.batch_size)  + "_lr_" + str(args.lr) + ".npy"
np.save(save_name, res) 
save_name_mean = "result/" + args.data_set + "_mean_" + str(args.data_num) + ".npy"
np.save(save_name_mean, output)
# print (np.linalg.inv(np.cov(res[-1].T)))















