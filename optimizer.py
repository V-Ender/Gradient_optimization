import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torchvision.transforms import transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt


def t_bounds(p, d, a):
    '''
    Find minimum and maximum t, so optimized area is inside rectangle
    :param p: np.array, point
    :param d: np.array, vector
    :param a: float, bound
    :return: (float, float), min and max t
    '''
    p = p.flatten()
    d = d.flatten()
    dim = p.shape[0]
   
    pos_t, neg_t = torch.inf, -torch.inf
    pos_t2, neg_t2 = torch.inf, -torch.inf
    t_tmp = (a - p) / d
    #print(t_tmp.shape)
    t_tmp_pos = t_tmp[t_tmp >= 0]
    t_tmp_neg = t_tmp[t_tmp < 0]
    if len(t_tmp_pos) > 0:
        pos_t = t_tmp[t_tmp >= 0].min()
    if len(t_tmp_neg) > 0:
        neg_t = t_tmp[t_tmp < 0].max()
    
    t_tmp = (-a - p) / d
    t_tmp_pos = t_tmp[t_tmp >= 0]
    t_tmp_neg = t_tmp[t_tmp < 0]
    if len(t_tmp_pos) > 0:
        pos_t2 = t_tmp[t_tmp >= 0].min()
    if len(t_tmp_neg) > 0:
        neg_t2 = t_tmp[t_tmp < 0].max()
        
    pos_t = torch.min(torch.tensor(pos_t), torch.tensor(pos_t2))
    neg_t = torch.max(torch.tensor(neg_t), torch.tensor(neg_t2))

    return (neg_t, pos_t)


class GlobalGradientOptimizer(optim.Optimizer):
    
    def __init__(self, params, lr=1e-3, bound=1):
        super(GlobalGradientOptimizer, self).__init__(params, defaults={'lr': lr})
        self.state = dict()
        self.bound = bound
        
    def step(self, closure):
        for group in self.param_groups:
            for p in group['params']:
                min_f = closure()
                
                grad = p.grad.data
                point = p.data
                d = grad
                point = torch.clip(point, -self.bound, self.bound)
                lb, rb = t_bounds(point, d, self.bound)
                
                step = group['lr'] * (rb - lb)
                t_arr = torch.arange(lb, rb, step)
                min_t = 0

                for t in t_arr:
                    p.data = point + t * grad
                    current_f = closure()
                    if current_f < min_f:
                        min_t = t
                        min_f = current_f
                        
                p.data = point + min_t * grad


class RandomVectorOptimizer(optim.Optimizer):
    
    def __init__(self, params, lr=1e-3, bound=1, num_vectors=10):
        super(RandomVectorOptimizer, self).__init__(params, defaults={'lr': lr})
        self.state = dict()
        self.bound = bound
        self.num_vectors = num_vectors
        
    def step(self, closure):
        for group in self.param_groups:
            for p in group['params']:
                start_p = p.data
                
                min_t = 0
                min_f = closure()
                best_grad = torch.zeros(p.grad.shape).to(device)
                for i in range(self.num_vectors):
                    min_f = closure()

                    grad = (torch.rand(p.grad.shape) * 2 - 1).to(device)
                    point = p.data
                    d = grad
                    d2 = p.grad.data

                    point = torch.clip(point, -self.bound, self.bound)
                    lb, rb = t_bounds(point, d, self.bound)

                    step = group['lr'] * (rb - lb)
                    t_arr = torch.arange(lb, rb, step)
                    min_t = 0

                    for t in t_arr:
                        p.data = point + t * grad
                        current_f = closure()
                        if current_f < min_f:
                            min_t = t
                            min_f = current_f
                            best_grad = grad

                p.data = start_p + min_t * best_grad
