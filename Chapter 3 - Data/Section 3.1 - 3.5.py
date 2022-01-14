# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 12:18:34 2021

@author: timsc
"""

#%% imports
import numpy as np
import pandas as pd

from utilities.utils import *
from utilities.plot_utils import *
from utilities.metrics import *

#%% load  events
events = load_events()

#%% reduce to events of sample match
#ID of sample match Eintracht - Bremen: 2516834
events_example = events[events['matchId'] == 2516834]

#%%Table 3.1-3.3 events sample match
#choose interesting events during the match to illustrate 
events_example_samp = events_example.loc[[158037, 158365, 158430, 159079, 159797],]

#print table to latex
print(events_example_samp.to_latex())

#%%Figure 3.1 visualize sample events on pitch
#rename vars for nicer legend
events_example_samp.columns = ['Id','matchId', 'Team ID', 'playerId', 'home', 'muBalance', 'outcome',
      'Event Type', 'subEventName', 'matchPeriod', 'eventSec', 'own_score',
       'opp_score', 'status', 'start_x', 'start_y', 'country']

#define colors and shapes
team_colors = {2462:'red', 2443:'green'}
event_shapes = {'Pass':'o', 'Shot':'s', 'Free Kick' : 'D'}

fig, ax = pitch()
sns.scatterplot('start_x', 'start_y', data=events_example_samp, 
                hue='Team ID', style = 'Event Type', legend = 'auto', 
                palette = ['green', 'red'], s = 150, )
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, handletextpad = 0.7,
           labelspacing  = .7 )
plt.xlim(-2,101)
plt.ylim(-2,101)
plt.axis('off')
plt.show()

#%%Table 3.4 counts event types
print(events['eventName'].value_counts())

#%%Table 3.5 counts sub-types
print(events.loc[events['eventName'] == 'Pass']['subEventName'].value_counts())
