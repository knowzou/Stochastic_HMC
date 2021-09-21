# Stochastic_HMC
Code for the ICML 2021 paper [On the Convergence of Hamiltonian Monte Carlo with Stochastic Gradient](https://proceedings.mlr.press/v139/zou21b.html) 

HMC with different types of stochastic gradients (mini-batch, control variates, SVRG, SAG)

Requirement: Cuda, pytorch, pandas

## Synthetic experiment: 
baseline
```python
python3 train.py --sample_num 20000 --lr 2e-3 --data_num 500 --in_epochs 10 --out_epochs 5000 --batch_size 500 --optimizer SG --enable_MH True 
```
'''OPTIMIZER''' = SG/SVRG/SAG/CVG
```python
python3 train.py --sample_num 20000 --lr 2e-3 --data_num 500 --in_epochs 10 --out_epochs 5000 --batch_size 16 --optimizer OPTIMIZER 
```
## Logistic experiment: 

```baseline```, ```DATA_SET``` = Pima/Covtype
```python
python3 train_logistic.py --sample_num 10000 --data_set DATA_SET --data_num 500 --lr 0.002 --optimizer SG --in_epochs 10 --enable_MH True --out_epochs 5000
```
```OPTIMIZER``` = SG/SVRG/SAG/CVG, ```DATA_SET``` = Pima/Covtype
```python
python3 train_logistic.py --sample_num 10000 --data_set DATA_SET --data_num 500 --lr 0.002 --in_epochs 10  --out_epochs 5000 --batch_size 16 --optimizer OPTIMIZER
```
### Arguments
```data_num```: number of component functions/training data
```sample_num```: number of samples generated in parallel
```lr```: learning rate
```in_epochs```: # leapfrog in each inner loop
```out_epochs```: # HMC proposals
```batch_size```: mini-batch size
```optimizer```: SG/SVRG/SAG/CVG

#######
For SAG (require to store the gradient of all samples for all examples), the sample num cannot be set to be too large due to the memory limit of GPU.

## Reference
For more technical details, please refer to the [paper](http://proceedings.mlr.press/v139/zou21b/zou21b.pdf).
```
@inproceedings{zou2021convergence,
  title={On the Convergence of Hamiltonian Monte Carlo with Stochastic Gradients},
  author={Zou, Difan and Gu, Quanquan},
  booktitle={International Conference on Machine Learning},
  pages={13012--13022},
  year={2021},
  organization={PMLR}
}
``` 
