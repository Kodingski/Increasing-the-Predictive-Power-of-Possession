# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 16:34:18 2022

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

def mirror_passes(passes):
    passes_mirrored = passes.copy()
    passes_mirrored['start_y'] = 100 - passes_mirrored['start_y']
    passes_merged = passes.append(passes_mirrored)
    return(passes_merged)

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

def plot_passes_on_zones(kmeans, k, passes_match, mirror = False):
        
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
        
        voronoi_plot_2d(vor, ax = ax, point_size = 0, line_width = 1.5, 
                         show_vertices = True, line_alpha = 1)

        #ax.scatter(x = 'start_x', y = 'start_y', data=passes_match)

        #for i_, zone in enumerate(zones):
        #    
        #    colored_cell = patches.Polygon(zone,
        #                           linewidth=1, 
        #                           alpha=1,
        #                           facecolor = 'green',
        #                           edgecolor="black"
        #                           )
        #    ax.add_patch(colored_cell)
       
        plt.axhline(50, color = 'black')
        
        sns.scatterplot(x='start_x', y = 'start_y', hue = 'Team', data = passes_match,
                        ax = ax, palette=['green', 'red']
                        )
        #ax.scatter(passes_match['start_x'],passes_match['start_y'], c = 'black')

        
        plt.xlim(-1,101)
        plt.ylim(-1,51)
        plt.axis('off')

        fig.tight_layout()
        plt.show()

def get_rgb_model(weight):
    #move weight into range of colormap by using sigmoid
    weight_stan = 1/(1+ np.exp(-weight*2))
    return plt.cm.RdYlGn(weight_stan)

def plot_poss(kmeans, k, vector, status = False, colors = False):
        
        centroids = kmeans.cluster_centers_
    

        vor = Voronoi(centroids)
        zones = voronoi_polygons(centroids)
        

        offsets = [0*k]
        
        color = 'white'

        if status:
            offsets = [0*k, 1*k, 2*k]


        for offset in offsets:
                
            fig, ax = pitch()
            voronoi_plot_2d(vor, ax = ax, point_size = 0)
            for i_, zone in enumerate(zones):
                if colors:
                    color = get_rgb_model(vector[i_+offset]*5)
                
                colored_cell = patches.Polygon(zone,
                                       linewidth=1, 
                                       alpha=1,
                                       facecolor = color,
                                       edgecolor="black"
                                       )
                ax.add_patch(colored_cell)
                if i_ == 4:
                    plt.text(centroids[i_,0]-8.5, centroids[i_,1]-8,
                             np.round(((vector[i_+offset])),2),
                             size = 18)                
                
                elif i_ == 6:
                    plt.text(centroids[i_,0]+ 1.5, centroids[i_,1],
                             np.round(((vector[i_+offset])),2),
                             size = 18)                
    
                else:
                    plt.text(centroids[i_,0]-1, centroids[i_,1],
                             np.round(((vector[i_+offset])),2),
                             size = 18)                
    

            
            plt.axhline(50, color = 'black')
            
            plt.xlim(-1,101)
            plt.ylim(-1,51)
            plt.axis('off')
    
            fig.tight_layout()
            plt.show()
            
def plot_poss(kmeans, k, vector, status = False, colors = False):
        
        centroids = kmeans.cluster_centers_
    

        vor = Voronoi(centroids)
        zones = voronoi_polygons(centroids)
        

        offsets = [0*k]
        
        color = 'white'

        if status:
            offsets = [0*k, 1*k, 2*k]


        for offset in offsets:
                
            fig, ax = pitch()
            voronoi_plot_2d(vor, ax = ax, point_size = 0)
            for i_, zone in enumerate(zones):
                if colors:
                    color = get_rgb(vector[i_+offset]*5)
                
                colored_cell = patches.Polygon(zone,
                                       linewidth=1, 
                                       alpha=1,
                                       facecolor = color,
                                       edgecolor="black"
                                       )
                ax.add_patch(colored_cell)
                if i_ == 4:
                    plt.text(centroids[i_,0]-8.5, centroids[i_,1]-8,
                             np.round(((vector[i_+offset])),2),
                             size = 18)                
                
                elif i_ == 6:
                    plt.text(centroids[i_,0]+ 1.5, centroids[i_,1],
                             np.round(((vector[i_+offset])),2),
                             size = 18)                
    
                else:
                    plt.text(centroids[i_,0]-1, centroids[i_,1],
                             np.round(((vector[i_+offset])),2),
                             size = 18)                
    

            
            plt.axhline(50, color = 'black')
            
            plt.xlim(-1,101)
            plt.ylim(-1,51)
            plt.axis('off')
    
            fig.tight_layout()
            plt.show()

def get_mean_poss_vector(passes_train, k, status = True):
    
    passes_home = passes_train[passes_train['home'] == passes_train['teamId']]
    passes_away = passes_train[passes_train['home'] != passes_train['teamId']]
    
    
    if status:
        pass_count_per_zone_per_status_home = passes_home.groupby(['status','zone'])['id'].count().values
        #rearrange to have trailing first
        pass_count_per_zone_per_status_home = np.hstack(
                                        (pass_count_per_zone_per_status_home[(2*k):(3*k)],
                                        pass_count_per_zone_per_status_home[0:(2*k)])
                                        )
        
        
        pass_count_per_zone_per_status_away = passes_away.groupby(['status','zone'])['id'].count()
        #rearrange to have trailing first
        pass_count_per_zone_per_status_away = np.hstack(
                                        (pass_count_per_zone_per_status_away[(k):(2*k)],
                                        pass_count_per_zone_per_status_away[0:(k)],
                                        pass_count_per_zone_per_status_away[(2*k):(3*k)])
                                        )
    
        mean_poss_vector = pass_count_per_zone_per_status_home / (pass_count_per_zone_per_status_home +
                                                             pass_count_per_zone_per_status_away)
        
    else:
        pass_count_per_zone_home = passes_home.groupby(['zone'])['id'].count().values
        pass_count_per_zone_away = passes_away.groupby(['zone'])['id'].count().values
        
        mean_poss_vector = pass_count_per_zone_home / (pass_count_per_zone_home +
                                                             pass_count_per_zone_away)

    return mean_poss_vector