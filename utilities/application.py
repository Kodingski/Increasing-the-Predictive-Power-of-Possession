# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 12:24:06 2022

@author: timsc
"""

#%% imports
import numpy as np
import scipy.stats
import scipy.optimize
import math
import matplotlib as plt
import matplotlib.patches as patches
import random


from scipy.spatial import Voronoi, voronoi_plot_2d
from utilities.plot_utils import * 


def get_rgb_post(weight):
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

def plot_weights_post(model, kmeans, X, k = 11, coeffs_inc = True):
        
        centroids_right = kmeans.cluster_centers_
        centroids_left = centroids_right.copy()
        centroids_left[:,1] = 100 - centroids_left[:,1]
        centroids = np.vstack((centroids_left,centroids_right))


        vor = Voronoi(centroids)
        zones = voronoi_polygons(centroids)
        
        coeffs = model.coef_
        if coeffs_inc == True:
            coeffs = coeffs*X
        else:
            coeffs = X
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
                                       facecolor=get_rgb_post((coeffs[i_+offset])),
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
            
