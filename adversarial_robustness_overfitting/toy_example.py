import argparse
import sys
import time
import math
import torch
import os
from torch.utils.data import DataLoader, Dataset

class dataset(Dataset):
    def  __init__(self, num, dim):

        self.data = torch.randn(num, dim)

    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        data = self.data[idx]
        return data
        
parser = argparse.ArgumentParser()
parser.add_argument('--me', action='store_true')
args = parser.parse_args()

print(args)

dim = 10
testset = dataset(100,dim)   
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)  
rho = 5e-4

if args.me == True: 
    print('mesgd_gen_gap')
else:
    print('sgd_gen_gap')


for num in [10,100,1000,10000,100000]:

  trainset =  dataset(num,dim)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)
  w = torch.rand(dim, requires_grad = True)
  
  if args.me == True: 
      u = torch.rand(dim, requires_grad = False)
      u_counter = 0
  
  optimizer = torch.optim.SGD([w], lr=0.01, momentum=0.9, weight_decay=0)
  
  for epoch in range(50):
      for i, x in enumerate(trainloader):
          if args.me == True: 
              loss = torch.abs(w - x).sum() + rho * torch.sum((w-u)*(W-u))/2
          else:
              loss = torch.abs(w - x).sum()
          optimizer.zero_grad()
          loss.backward(retain_graph = True)
          optimizer.step()
          
          if args.me == True: 
            alpha = 1/ (1.0 + u_counter)
            u *= (1.0 - alpha)
            u += w * alpha
            u_counter += 1
  
  train_error = 0
  
  if args.me == True: 
      out = u
  else:
      out = w
  
  for i, x in enumerate(trainloader):
  
      loss = torch.abs(out - x).sum()
      train_error += loss.item()
  train_error = train_error/num
  
  test_error = 0
  
  for i, x in enumerate(testloader):
      loss = torch.abs(out - x).sum()
      test_error += loss.item()
  test_error = test_error/100
  
  gen_gap = test_error - train_error


  print(gen_gap)
