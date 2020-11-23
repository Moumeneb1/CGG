import os 
import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt


#nn.init.xavier_uniform(m.bias.data)

def weights_init_xavier(m):
    if isinstance(m, nn.Conv2d):
      nn.init.xavier_uniform_(m.weight.data)

def save_nets(net,amp,optimizer, model_dir):
    name = 'unet'

    if not os.path.exists(model_dir):
        print('Creating model directory: {}'.format(model_dir))
        os.makedirs(model_dir)

    if net is not None:
          torch.save({'model': net.state_dict(),
          'optimizer': optimizer.state_dict(),
          'amp': amp.state_dict()
          }, '{}/{}.pth'.format(model_dir, name))

def load_best_weights(model,amp,optimizer,opt_level, model_dir):
    checkpoint = torch.load('{}/unet.pth'.format(model_dir))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    amp.load_state_dict(checkpoint['amp'])
    #model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
    return model 

def plot_images(dataset,num_images):
  num_row = num_images
  num_col = 2# plot images
  fig, axes = plt.subplots(num_col,num_row, figsize=(2*num_row,1.5*num_col))
  for i in range(0,num_row*num_col,2):
      ax = axes[i%num_col,i//num_col, ]
      ax.imshow(dataset[i][0][0], interpolation='none')
      ax.set_title('Image')
      ax = axes[ (i+1)%num_col,(i+1)//num_col,]
      ax.imshow(dataset[i][1], interpolation='none')
      ax.set_title('Mask')
  plt.tight_layout()
  plt.show()


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            # self.step = lambda a: False

    def step(self, metrics):
        if self.patience == 0:
            return False, self.best, self.num_bad_epochs
            
        if self.best is None:
            self.best = metrics
            return False, self.best, 1

        if torch.isnan(metrics):
            return True, self.best, self.num_bad_epochs

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True, self.best, self.num_bad_epochs

        return False, self.best, self.num_bad_epochs

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)


