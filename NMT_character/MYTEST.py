from torch.nn import Embedding
import torch
import numpy as np
import torch.nn as nn
def t1():
    a = Embedding(10,3)
    input = torch.tensor(np.random.randint(10,size=(10,8,3)).astype(np.float32))
    print(input.shape)
    b = a(input)
    print(b.shape)

def t2():
    a = [1,2,3,4,5]
    for i in range(10,5,-1):
        print(i)

def t3():
    a = np.random.choice(9,9,replace=False)
    print(a)

def t4():
    a = torch.randn(2,3,4)
    print(a.shape)

    b = a.view(12,-1)
    print(b.shape)
    print(a)
    print(b)


def f5():
    a = [[[1],[2]],[[2],[3]],[[3],[4]]]
    a = torch.tensor(a,dtype=torch.long)
    e = nn.Embedding(10,5)
    print(a.shape)
    print(e(a).shape)

f5()
