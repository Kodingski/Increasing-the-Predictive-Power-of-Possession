# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 11:00:05 2021

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

#%%
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
        

#%% load data, merge and reduce to passes
countries = ['Germany', 'Spain', 'Italy', 'England', 'France']

#even_events = []
events = []

for country in countries:
    events_country = pd.read_json(f'data/preprocessed/events_{country}_preprocessed.json')
    events.append(events_country)

del events_country

events = pd.concat(events, axis=0, ignore_index=True)

passes = events[events['eventName'] == 'Pass']

#%%
ratio = 0.5

passes_unique = passes.drop_duplicates(subset = 'matchId')
passes_unique['matchBalance'] = np.abs((passes_unique['percHome'] - (passes_unique['percAway'] + 0.15)))
passes_unique_sorted_even = passes_unique.sort_values(axis = 0, by = 'matchBalance')
passes_unique_ratio = passes_unique_sorted_even.head(int(len(passes_unique_sorted_even)*ratio))
matches_even = pd.Series(passes_unique_ratio['matchId'].unique())
passes_even = passes.loc[passes['matchId'].isin(matches_even)]
#%%
plot_kde_event_on_field(passes_even, sample_size=100000000)
#%%
def calc_kde(data, sample_size=10000, event = 'Pass', sub_events = False):
    """
    Generate kernel density estimates for each event type
    
    Parameters
    ----------
    sample_size: int
        random sample of values to use (default: 10000). The code becomes slow is you increase this value
        significantly.
    """

    event_data = data[data['eventName'] == event]        
    event_grouped = event_data.groupby(['status', 'outcome'])

    for name, group in event_grouped:
        
        n = len(group)
        if n >= sample_size:
            x_y = group[['start_x','start_y']].sample(sample_size).astype(float)
            n = sample_size
        else:
            x_y = group[['start_x','start_y']]
        
        if (name[0] == 'drawing'):
            if (name[1] == 'won'):
                passes_winner = x_y
                passes_winner_array = np.array(passes_winner).T
                kde_winner = scipy.stats.gaussian_kde(passes_winner_array)
                hist_winner = np.histogram2d(passes_winner_array[0], passes_winner_array[1])

            if (name[1] == 'lost'):
                passes_loser = x_y
                passes_loser_array = np.array(passes_loser).T
                kde_loser = scipy.stats.gaussian_kde(passes_loser_array)
                hist_loser = np.histogram2d(passes_loser_array[0], passes_loser_array[1])

    return(kde_winner, hist_winner, kde_loser, hist_loser)

#%%
kde_winner, hist_winner, kde_loser, hist_loser = calc_kde(passes_even, sample_size = 1000000)
X, Y = np.mgrid[0:100:100j, 0:100:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
estimates_winner = np.reshape(kde_winner(positions).T, X.shape)
estimates_loser = np.reshape(kde_loser(positions).T, X.shape)

#%%
fig, ax = pitch()
plt.imshow(np.rot90(estimates_winner - estimates_loser), cmap = 'RdYlGn',
          extent=[0, 100, 0, 100], aspect = 0.7)
plt.xlim(-1, 101)
plt.ylim(-1, 101)
plt.axis('off')
fig.tight_layout()

plt.show()

#%% For subtypes
plot_kde_event_on_field(passes_even, sample_size=100000000, sub_events=True)



#%%
def get_rgb(weight):
    #move weight into range of colormap by using sigmoid
    weight_stan = 1/(1+ np.exp(-(weight*20000)))
    return plt.cm.RdYlGn(weight_stan)
#%%

sub_passes = passes.groupby(['subEventName'])
            
for name_sub, subgroup in sub_passes:
    print(name_sub)
    kde_winner, hist_winner, kde_loser, hist_loser = calc_kde(subgroup, sample_size = 1000000)
    X, Y = np.mgrid[0:100:100j, 0:100:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    estimates_winner = np.reshape(kde_winner(positions).T, X.shape)
    estimates_loser = np.reshape(kde_loser(positions).T, X.shape)
    
    diff = (estimates_winner - estimates_loser)
    diff[np.abs(diff) < 0.00000001] = 0

    
    fig, ax = pitch()
    plt.imshow(get_rgb(np.rot90(diff)),
              extent=[0, 100, 0, 100], aspect = 0.7, norm = None)
    plt.xlim(-1, 101)
    plt.ylim(-1, 101)
    plt.axis('off')
    fig.tight_layout()
    
    plt.show()



