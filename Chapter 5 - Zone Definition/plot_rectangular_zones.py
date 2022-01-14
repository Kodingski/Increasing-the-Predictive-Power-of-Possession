# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 14:25:19 2021

@author: timsc
"""
#%%
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import matplotlib as plt
from utilities.utils import *
from utilities.plot_utils import *
from utilities.metrics import *

from sklearn.cluster import KMeans
#%%


for k_root in range(1,7):
    
    fig, ax = pitch()
    plt.xlim(-1,101)
    plt.ylim(-1,101)
    plt.axis('off')
    
    
    for line in range(1,k_root):
    
        plt.axhline(y=100/k_root*line, color='black', linestyle='-')
        plt.axvline(x=100/k_root*line, color='black', linestyle='-')
    
    
    plt.suptitle('Rectangular zones', fontsize = 30)
    plt.title(f'k = {k_root**2}')
    fig.tight_layout()
    plt.show()