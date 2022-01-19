# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 10:59:25 2021

@author: timsc
"""

#%% imports
import numpy as np
import pandas as pd

from utilities.fitting import * 
from utilities.utils import *
from utilities.zones import *

import warnings
warnings.filterwarnings("ignore")

#%%Load results Even test
test_even = load_test_even()

#%%Table 7.4 - Results on even Subset
Columns = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']

MU_even_results = pd.DataFrame(test_even[(0,2,4),], columns = Columns)
MU_even_results['Mean'] = MU_even_results.mean(axis=1)
MU_even_results = np.round(MU_even_results, 3)
MU_even_results_latex = MU_even_results.to_latex()
