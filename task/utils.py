import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle

import torch
import torch.nn as nn
# import torch.optim as optim
import torch.utils.data

from torch.utils.data import TensorDataset

import my_matplotlib_style as ms


from fastai import basic_data, basic_train
from fastai import train as tr

from nn_utils import get_data

#standard deviation of error term
def std_error(x, axis=None, ddof=0):
    return np.nanstd(x, axis=axis, ddof=ddof) / np.sqrt(2 * len(x))

#plotting routine
def plot_activations(learn, figsize=(12, 9), lines=['-', ':'], save=None, linewd=1, fontsz=14):
    plt.figure(figsize=figsize)
    for i in range(learn.activation_stats.stats.shape[1]):
        thiscol = ms.colorprog(i, learn.activation_stats.stats.shape[1])
        plt.plot(learn.activation_stats.stats[0][i], linewidth=linewd, color=thiscol, label=str(learn.activation_stats.modules[i]).split(',')[0], linestyle=lines[i % len(lines)])
    plt.title('Weight means')
    plt.legend(fontsize=fontsz)
    plt.xlabel('Mini-batch')
    if save is not None:
        plt.savefig(save + '_means')
    plt.figure(figsize=(12, 9))
    for i in range(learn.activation_stats.stats.shape[1]):
        thiscol = ms.colorprog(i, learn.activation_stats.stats.shape[1])
        plt.plot(learn.activation_stats.stats[1][i], linewidth=linewd, color=thiscol, label=str(learn.activation_stats.modules[i]).split(',')[0], linestyle=lines[i % len(lines)])
    plt.title('Weight standard deviations')
    plt.xlabel('Mini-batch')
    plt.legend(fontsize=fontsz)
    if save is not None:
        plt.savefig(save + '_stds')
