import random
import numpy as numpy
import torch
from torch.autograd import grad as torchgrad
from Sto_HMC import SGHMC


parser = argparse.ArgumentParser(description='train stochastic HMC')
parser.add_argument('--sample_num', default = 2000,
                    help='number of generated samples')
parser.add_argument('--data_num', default = 200,
                    help='number of training data')
parser.add_argument('--energy_func', default = "diff_mean",
                    help='type of target density')
parser.add_argument("--out_epochs", default = 2000,
                    help = "outer epochs")
parser.add_argument("--in_epochs", default = 100,
                    help = "inner epochs")
parser.add_argument("--lr", default = 2e-5,
                    help = "learning rate")
parser.add_argument("--optimizer", default = "SG",
                    help = "optimizer")
parser.add_argument("--batch_size", defult = 8,
                    help = "batch size")
parser.add_argument("--enable_MH", defult = False,
                    help = "enable metropolis hasting")
args = parser.parse_args()

if torch.cuda.is_available():
    device = "cuda" 
else: 
    device="cpu"

if parser.energy_func == "diff_var": ##################### different variance
    n, d = args.data_num, 1
    X = np.random.rand(n, d)*10+1
    energy = energy1
    initial = torch.zeros(args.sample_num,d).to(device)
elif parser.energy_func == "diff_covar": ##################### different covariance
    d = 2
    X = np.ones([args.data_num, d, d])
    Y = np.asarray([[1, 0.7], [0.7, 1]])
    for i in range(args.data_num):
        X[i, :, :] = Y + np.random.randn(d,d)
    energy = energy2
    initial = torch.zeros(args.sample_num,d).to(device)+1.5
elif parser.energy_func == "diff_mean": ##################### different mean
    n, d = args.data_num, 2
    X = np.random.rand(n, d)*3
    initial = torch.zeros(args.sample_num,d).to(device)
    energy = energy3


def energy1(q, data):
    data =  torch.from_numpy(data).float().to(device)
    all_energy = torch.matmul(data**(-1), (q**2).sum(axis=1, keepdim=True).T).mean(axis=0)
    return all_energy
def energy2(q, data):
    data =  torch.from_numpy(data).float().to(device)
    
    all_energy = ((torch.matmul(data, q.T))**2).sum(axis=1).mean(axis=0)
    return all_energy
def energy3(q, data):
    
    n, N = args.data_num, args.sample_num
    cov = np.asarray([[1,1], [1,2]])
    cov =  torch.from_numpy(cov).float().to(device)
    cov_expand1 = (cov.T).unsqueeze(0).repeat(n, 1,1)
    data =  torch.from_numpy(data).float().to(device)    
    data = torch.matmul(data, cov)
    data = data.unsqueeze(2).repeat(1,1,N)
    q = torch.matmul(cov_expand1, q.T)
    all_energy = ((q - data)**2).sum(axis=1).mean(axis=0)
    
    return all_energy



start = time.time()
output_HMC = SGHMC(energy = energy, 
                data = X,
                batch_size = args.batch_size,
                initial = initial,
                out_epochs = args.out_epochs,
                in_epochs = args.in_epochs,
                lr = args.lr, 
                device = device,
                optimizer = args.optimizer,
                enable_MH = args.enable_MH
                 )  
print ("SG Run time: ", time.time()-start)