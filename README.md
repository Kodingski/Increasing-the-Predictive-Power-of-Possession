# Increasing-the-Predictive-Power-of-Possession

This repository provides the code used in my thesis project "Increasing the Predictive Power of the Possession Metric in Football by Adding Spatio-temporal Context" 
submitted for the degree Master of ScienceStatistics: Data Science at Leiden University.

## Abstract:
In recent years, statistics play an increasing role in professional football.
A controversial topic inside the emerging field of football data science is the effect of ball possession on match outcomes. 
We contribute to this discussion by analyzing the effect of possession on match outcomes while controlling for match status and match-up balance. 
We examine the importance of the position of possession by comparing the kernel density estimate of winning and losing teams.
Based on these findings we split the football pitch into distinct zones using Voronio cells based on the centroids of a k-means clustering.
We fit a multiple linear regression model that regresses a matches final goal difference on possession per match status per zone using a 5x5-fold nested cross-validation.
The resulting model splits the football pitch into 11 zones.
Our metric holds higher predictive power than the traditional metric.
To demonstrate the potential of this work for both analysts and journalists, 
we analyze a teams performance over a whole season as well as individual match performances using the metric.

**Keywords:** football, soccer, possession, event data, spatio-temporal data, match status, match-up balance, kernel density estimation, k-means clustering, voronoi cells

The PDF containing the full thesis can be found in this repository.

## Data Source and Preprocessing

