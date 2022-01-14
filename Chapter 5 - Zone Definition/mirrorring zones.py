# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 09:23:14 2021

@author: timsc
"""

#%% imports
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import matplotlib as plt
import matplotlib.patches as patches
import random
import statsmodels.api as sm
import pickle


from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import KMeans
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from utilities.utils import *
from utilities.plot_utils import *
from utilities.metrics import *


#%% load data, merge and reduce to passes
countries = ['Germany', 'Spain', 'Italy', 'England', 'France']

#even_events = []
events = []

for country in countries:
    events_country = pd.read_json(f'data/preprocessed/events_{country}_preprocessed.json')
    events.append(events_country)

del events_country

events = pd.concat(events, axis=0, ignore_index=True)

#%%helpers
def mirror_passes(passes):
    passes_mirrored = passes.copy()
    passes_mirrored['start_y'] = 100 - passes_mirrored['start_y']
    passes_merged = passes.append(passes_mirrored)
    return(passes_merged)
    
#%%
passes = events[events['eventName'] == 'Pass']
#%%pass count per half
passes_right_half = passes[passes['start_y'] <= 50]
passes_left_half = passes[passes['start_y'] >= 50]

passes_per_tenth_per_half = np.zeros((10,2))
step = 10
for i, x in enumerate(range(0,100,step)):
    
    passes_per_tenth_per_half[i,0] = len(passes[(passes['start_y'] >= 50) & (passes['start_x'].between(x, x+step))])
    passes_per_tenth_per_half[i,1] = len(passes[(passes['start_y'] <= 50) & (passes['start_x'].between(x, x+step))])    
#%%
perc_left = passes_per_tenth_per_half[:,0]/ (passes_per_tenth_per_half[:,0] + passes_per_tenth_per_half[:,1])


#%%

samp = passes.sample(1000)

fig, ax = pitch()



sns.scatterplot('start_x', 'start_y', data=samp)

plt.xlim(-1,101)
plt.ylim(-1,101)
plt.axis('off')

fig.tight_layout()

plt.show()

#%%
samp_mirrored = mirror_passes(samp)
samp_mirrored = samp_mirrored[samp_mirrored['start_y'] <= 50]



fig, ax = pitch()


#sns.scatterplot(data=tips, x="total_bill", y="tip", hue="time", style="time")

sns.scatterplot('start_x', 'start_y', data=samp_mirrored)
plt.axhline(50, color ='black')

plt.xlim(-1,101)
plt.ylim(-1,101)
plt.axis('off')

fig.tight_layout()

plt.show()

#%%

kmeans = KMeans(n_clusters = 10, max_iter = 500).fit(samp_mirrored[['start_x', 'start_y']])
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

def plot_weights(kmeans, k, mirror = False):
        
        centroids_right = kmeans.cluster_centers_
        
        if mirror:
            centroids_left = centroids_right.copy()
            centroids_left[:,1] = 100 - centroids_left[:,1]
            centroids = np.vstack((centroids_left,centroids_right))
        else:
            centroids = centroids_right

        vor = Voronoi(centroids)
        zones = voronoi_polygons(centroids)
        

        

        fig, ax = pitch()
        voronoi_plot_2d(vor, ax = ax, point_size = 0)
        for i_, zone in enumerate(zones):
            
            colored_cell = patches.Polygon(zone,
                                   linewidth=1, 
                                   alpha=1,
                                   color = 'white',
                                   edgecolor="black"
                                   )
            ax.add_patch(colored_cell)
       
        plt.axhline(50, color = 'black')
        
        plt.xlim(-1,101)
        plt.ylim(-1,101)
        plt.axis('off')

        fig.tight_layout()
        plt.show()
   
#%%
plot_weights(kmeans, 10, mirror = True)
         
#%%plot vor zones for different k 
passes_mirrored = mirror_passes(passes)
passes_mirrored = passes_mirrored[passes_mirrored['start_y'] <= 50]


for k in [5,10,15,20]:
    kmeans = KMeans(n_clusters = k, max_iter = 500).fit(passes_mirrored[['start_x', 'start_y']])
    plot_weights(kmeans, k, mirror = True)
#%%
def plot_rect_zones_mirrored(k):
    
    if k == 1:
        k_root = 1
    elif k == 6:
        k_root = 3
    elif k == 15:
        k_root = 5
    else:
        k_root = int(np.sqrt(k*2))
    
    
    center_coords, anchor_coords = get_rect_zone_center_coords(k_root, k)
    center_coords_mirrored, anchor_coords_mirrored = center_coords.copy(), anchor_coords.copy()
    center_coords_mirrored[:,1] = 100- center_coords_mirrored[:,1] 
    anchor_coords_mirrored[:,1] = 100 - anchor_coords_mirrored[:,1] - (100/k_root)


    

    fig, ax = pitch()
    plt.xlim(-1,101)
    plt.ylim(-1,101)
    plt.axis('off')
    
    
    for i_ in range(len(center_coords)):
        
        rectangle = patches.Rectangle(anchor_coords[i_],
                                        100/k_root, 
                                        100/k_root,
                                        linewidth = 1,
                                        alpha=1,
                                        facecolor = 'white',
                                        edgecolor="black",
                                        )
        ax.add_patch(rectangle)
        ax.plot(anchor_coords[i_,0], anchor_coords[i_,1]+rectangle.get_height(), '.', markersize = 13, color = 'tab:orange')
        
        ax.plot(anchor_coords[i_,0]+rectangle.get_width(), anchor_coords[i_,1], '.', markersize = 13, color = 'tab:orange')

        ax.plot(anchor_coords[i_,0], anchor_coords[i_,1], '.', markersize = 13, color = 'tab:orange')
        
        
        
        rectangle_mirrored = patches.Rectangle(anchor_coords_mirrored[i_],
                            100/k_root, 
                            100/k_root,
                            alpha=1,
                            facecolor ='white',
                            edgecolor="black"
                            )
        ax.add_patch(rectangle_mirrored)  
        
        ax.plot(anchor_coords_mirrored[i_,0]+rectangle_mirrored.get_width(), 
                anchor_coords_mirrored[i_,1]+rectangle_mirrored.get_height(), '.', markersize = 13, color = 'tab:orange')

        ax.plot(anchor_coords_mirrored[i_,0], anchor_coords_mirrored[i_,1]+rectangle_mirrored.get_height(),
                '.', markersize = 13, color = 'tab:orange')
            
        ax.plot(anchor_coords_mirrored[i_,0]+rectangle_mirrored.get_width(), 
                anchor_coords_mirrored[i_,1], '.', markersize = 13, color = 'tab:orange')          
        
    plt.axhline(50, color = 'black')

    plt.xlim(-1,101)
    plt.ylim(-1,101)
    plt.axis('off')
    fig.tight_layout()
    plt.show()



def get_rect_zone_center_coords(k_root, k):
    
    zone_anchors = np.empty([k,2])
    #zone_points = np.empty([])
    zone_centers = np.empty([k, 2])        
    step = 100/k_root
    
    zone = 0
    
    for x in np.arange(0, 100, step):
        for y in np.arange(0, 50, step):
            
            
            zone_anchors[zone, 0] = x
            zone_anchors[zone, 1] = y
            
            zone_centers[zone, 0] = x+(step/2)
            zone_centers[zone, 1] = y+(step/2)
            zone += 1
            

    return(zone_centers, zone_anchors)


#%%
for k in [2,6,8,15,18]:
    plot_rect_zones_mirrored(k)


