# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 11:31:32 2022

@author: timsc
"""

#%%imports
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import matplotlib as plt
from utilities.utils import *
from utilities.plot_utils import *
from utilities.metrics import *


#%% load data and reduce to passes played in even matches
passes = load_passes()

#%%define functions used in this chapter
def plot_kde_event_on_field(data, sample_size=1000000, event = 'Pass', sub_events = False):
    """
    Generate density plots on the field for each event type
    
    Parameters
    ----------
    sample_size: int
        random sample of values to use (default: 10000). The code becomes slow is you increase this value
        significantly.
    """

    event_data = data[data['eventName'] == event]
    event_data = event_data[event_data['status'] == 'drawing']        
    event_grouped = event_data.groupby(['outcome'])

    for name, group in event_grouped:
        
        
        if sub_events:
            sub_groups = group.groupby(['subEventName'])
            
            for name_sub, subgroup in sub_groups:
                n = len(subgroup)
                print(f'Outcome: {name}, Pass type {name_sub}, n = {n}')
                
                fig, ax = pitch()

                if n >= sample_size:
                    x_y = subgroup[['start_x','start_y']].sample(sample_size).astype(float)
                    n = sample_size
                
                else:
                    x_y = subgroup[['start_x','start_y']].astype(float)
                sns.kdeplot(x = subgroup['start_x'], y = subgroup['start_y'], cmap = 'Greens', shade = True)
                
                plt.xlim(-1,101)
                plt.ylim(-1,101)
                plt.axis('off')
                fig.tight_layout()
                plt.show()
            
        else:    
            print(name)
            fig, ax = pitch()
            n = len(group)
            if n >= sample_size:
                x_y = group[['start_x','start_y']].sample(sample_size).astype(float)
                n = sample_size
            else:
                x_y = group[['start_x','start_y']].astype(float)
            sns.kdeplot(x = group['start_x'], y = group['start_y'], cmap = 'Greens', shade = True)
            
            plt.xlim(-1,101)
            plt.ylim(-1,101)
            plt.axis('off')
            fig.tight_layout()
            plt.show()
            
#%% Create Figures Appendix B (subtypes)
#Sample size was set very high for final thesis run for maximal stability,
#for quicker results put to default 
plot_kde_event_on_field(passes, sample_size=100000000, sub_events=True)




            