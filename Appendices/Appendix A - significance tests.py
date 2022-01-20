# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 11:58:58 2021

@author: timsc
"""

#%% imports
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from utilities.utils import *

#%%load perms
perms = load_perms()

#%%transform into poss value
poss = perms['perms'][0] / (perms['perms'][0] + perms['perms'][1])
poss_status = perms['perms_status'][0] / (perms['perms_status'][0] + perms['perms_status'][1])
poss_drawing = perms['perms_drawing'][0] / (perms['perms_drawing'][0] + perms['perms_drawing'][1])
poss_even = perms['perms_even'][0] / (perms['perms_even'][0] + perms['perms_even'][1])
poss_even_drawing = perms['perms_even_drawing'][0] / (perms['perms_even_drawing'][0] + perms['perms_even_drawing'][1])

#%%Appendix A - Hists
sns.histplot(poss)
plt.xlabel('Poss. of Winners')
plt.show()

sns.histplot(poss_status)
plt.xlabel('Poss. of Trailing Teams')
plt.show()

sns.histplot(poss_drawing)
plt.xlabel('Poss. of Winners')
plt.show()

sns.histplot(poss_even)
plt.xlabel('Poss. of Losers')
plt.show()

sns.histplot(poss_even_drawing)
plt.xlabel('Poss. of Winners')
plt.show()

#%%get quantiles
def get_empirical_p(poss, obs_poss, N = 10000):
    poss_sorted = np.sort(poss)
    position_ind = np.searchsorted(poss_sorted, obs_poss)
    p = 1 - (position_ind/N)
    return p
    
#%%Quantiles for EDA Chapter
p_outcome = get_empirical_p(poss, 0.528)
p_status = get_empirical_p(poss_status, 0.521)
p_drawing = get_empirical_p(poss_drawing, 0.561)
p_even = get_empirical_p(poss_even, 0.521)
p_even_drawing = get_empirical_p(poss_even_drawing, 0.527)

