# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 16:19:26 2021

@author: timsc
"""
#%% imports
import os
import copy
import numpy as np
import pandas as pd
import scipy.stats
import scipy.optimize
import seaborn as sns
import math
import matplotlib as plt
import matplotlib.patches as patches
import random
import statsmodels.api as sm
import pickle



from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, ElasticNet
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


from utilities.utils import *
from utilities.plot_utils import *
from utilities.metrics import *

#%%load match overview
folder_results = 'application/'

matches_overview = pickle.load(open(folder_results+'matches_overview.pickle', "rb"))
matches_overview = matches_overview.drop('matchId', axis = 1)

#%%see distribution of normal possession
matches_overview['raw poss. Home'].hist(cumulative = True, bins = 100)


#%%

def calc_f(x):
    f = scipy.stats.ks_2samp(matches_overview['raw poss. Home'], 
                   matches_overview['pred Home'].apply(lambda data : (1 / (1 + math.exp(-x*data)))))[0]

    return(f)

#%%
optimization = scipy.optimize.minimize_scalar(calc_f)
minimal_x = optimization['x']
#%%

matches_overview['pred Home trans'] = matches_overview['pred Home'].apply(lambda x : (1 / (1 + math.exp(-minimal_x*x))))
#%%
matches_overview['pred Home trans'].hist(cumulative = True, bins = 100)


#%%prepare data to plot with sns

data_pred = matches_overview[['pred Home trans']]
data_pred['hue'] = 'Transformed Model Pred.'
data_pred.columns = ['Possession', 'Type']



data_raw = matches_overview[['raw poss. Home']]
data_raw['hue'] = 'Raw Possession'
data_raw.columns = ['Possession', 'Type']


data_plot = data_pred.append(data_raw, ignore_index = True)

#%%
sns.set_theme()

ax = sns.ecdfplot(data = data_plot, x = 'Possession', hue = 'Type')
ax.set(ylim=(-0.05, 1.05))

#sns.ecdfplot(data = matches_overview, x = )


#%%
matches_overview['diff'] = np.abs(matches_overview['pred Home trans'] - matches_overview['raw poss. Home'])


#%%load results
folder_results = 'results/final/'

files = os.listdir(folder_results)
results = {}


for f in files:
    if f == 'models.pickle':
        models = pickle.load(open(folder_results+f, "rb"))
    else:                        
        results[str(f.rstrip('.pickle'))] = pickle.load(open(folder_results+f, "rb"))
        

final_zones, final_model = results['final_kmeans'], results['final_mod']      


def get_rgb(weight):
    #move weight into range of colormap by using sigmoid
    weight_stan = 1/(1+ np.exp(-weight*40))
    return plt.cm.RdYlGn(weight_stan)

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)



def voronoi_polygons(centroids):
    vor = Voronoi(centroids)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    polygons = []
    for reg in regions:
        polygon = vertices[reg]
        polygons.append(polygon)
    return polygons
#%%
def plot_weights(model, kmeans, X, k = 11):
        
        centroids_right = kmeans.cluster_centers_
        centroids_left = centroids_right.copy()
        centroids_left[:,1] = 100 - centroids_left[:,1]
        centroids = np.vstack((centroids_left,centroids_right))


        vor = Voronoi(centroids)
        zones = voronoi_polygons(centroids)
        
        coeffs = model.coef_
        coeffs =    X
        coeffs = np.hstack((coeffs[0:k],coeffs[0:k], 
                               coeffs[k:(2*k)],coeffs[k:(2*k)],
                               coeffs[(2*k):(3*k)],coeffs[(2*k):(3*k)]
                               ))
        intercept = model.intercept_
        offsets = [0*k, 2*k, 4*k]
        
        for offset in offsets:   
            
            fig, ax = pitch()
            voronoi_plot_2d(vor, ax = ax, point_size = 0)
            for i_, zone in enumerate(zones):

                colored_cell = patches.Polygon(zone,
                                       linewidth=1, 
                                       alpha=1,
                                       facecolor=get_rgb((coeffs[i_+offset])),
                                       edgecolor="black"
                                       )
                ax.add_patch(colored_cell)
                if i_ == 3 or i_ == 14:
                    text_offset = 1.8
                else:
                    text_offset = -4

                plt.text(centroids[i_,0] + text_offset, centroids[i_,1],np.round(((coeffs[i_+offset])),2), size = 16)                

                    
                    
            plt.xlim(-1,101)
            plt.ylim(-1,101)
            plt.axis('off')
            fig.tight_layout()
            plt.show()
            
#%%
matches = pd.read_json(f"data\matches\matches_all_leagues_odds.json", encoding='unicode_escape')
matches_la_liga = matches[matches['roundId'] == 4406122] 



#%%build latex table matches
matches_latex = pd.merge(matches[['wyId', 'home_name', 'away_name', 'gameweek', 'date']],
                         matches_overview[['wyId', 'label', 'home_score', 
                                           'away_score', 'diff',
                                           'raw poss. Home', 'pred Home trans']],
                         how = 'left', left_on = 'wyId', right_on = 'wyId')

matches_latex = matches_latex.sort_values('diff', ascending = False).head(20)
matches_latex_red = matches_latex[['label', 'date', 'gameweek', 'raw poss. Home', 
                                   'pred Home trans', 'diff']]
matches_latex_red[['Matchup', 'Result']] = matches_latex_red['label'].str.split(',', expand = True)
matches_latex_red['Result'] = matches_latex_red['Result'].str.strip()
#%%
matches_latex_red['Date'] = matches_latex_red['date'].str.split('at', expand = True)[0]
matches_latex_red['Date'] = matches_latex_red['Date'].str.strip()



#%%
matches_latex = matches_latex_red[['Matchup', 'gameweek', 'Result', 'raw poss. Home', 
                                   'pred Home trans', 'diff']]
matches_latex.columns = ['Match-up', 'Match-day', 'Result', 'Raw Poss.', 'Trans. Model Pred.', 'Difference']


#%%

matches_latex_str = matches_latex.to_latex(float_format="%.2f", index = False)
#%%
testt = test.iloc[0][0]
#%%analyze single match Betis - Real

betis_real = matches_overview[matches_overview['wyId'] == 2565780]


X_betis_real = betis_real.iloc[0,6:39].values

plot_weights(final_model, final_zones, X_betis_real)


#%%
matches_la_liga_overview = pd.merge(matches_la_liga[['wyId', 'home_name', 'away_name']], matches_overview, how = 'left', left_on = 'wyId', right_on = 'wyId')




#%%
home_poss_per_team = matches_la_liga_overview.groupby('home_name').mean()
away_poss_per_team = matches_la_liga_overview.groupby('away_name').mean()
away_poss_per_team.iloc[:,0:4] = 1 - away_poss_per_team.iloc[:,0:4]
away_poss_per_team.iloc[:,4:38] = away_poss_per_team.iloc[:,4:38]*-1
away_poss_per_team.iloc[:,38] = 1 - away_poss_per_team.iloc[:,38]


poss_per_team = ((home_poss_per_team + away_poss_per_team) / 2).reset_index()




#%%
poss_per_team['diff'] = poss_per_team['raw poss. Home'] - poss_per_team['pred Home trans']

#%%build table la liga
table_la_liga = poss_per_team[['home_name', 'raw poss. Home', 'pred Home trans', 'diff']]
table_la_liga.columns = ['Club Name', 'Raw Poss.', 'Trans. Model Pred.', 'Difference']

table_la_liga['Position'] = [16,2,1,13,14,18,9,11,8,10,19,17,15,20,6,3,12,7,4,5]
table_la_liga['Points'] =[43,79,93,49,47,29,51,49,55,51,22,43,46,20,60,76,49,58,73,61]
table_la_liga['Goal Diff.'] = [-8,36,70,-1,-10,-38,-6,-6,9,-9,-50,-17,-14,-37,-1,50,7,-9,27,7]

col_order = ['Position', 'Club Name','Goal Diff.', 'Points', 'Raw Poss.', 'Trans. Model Pred.', ]

table_latex_ready = table_la_liga[col_order]
#%%
table_latex = table_latex_ready.sort_values('Position').to_latex(float_format="%.2f", index = False)

#%%
check = poss_per_team['raw poss. Home'].mean()
#%%
poss_atletico = poss_per_team[poss_per_team['home_name'] == 'Atlético Madrid']
X_atletico = poss_atletico.iloc[0,6:39].values

plot_weights(final_model, final_zones, X_atletico)

#%%
poss_las_palmas = poss_per_team[poss_per_team['home_name'] == 'Las Palmas']
X_las_palmas = poss_las_palmas.iloc[0,6:39].values

plot_weights(final_model, final_zones, X_las_palmas)







#%%


match = matches_overview[matches_overview['wyId'] == 	2565562]
X = match.iloc[0,12:45].values


plot_weights(final_model, final_zones, X)




