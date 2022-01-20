# Increasing-the-Predictive-Power-of-Possession

![Final model](./cover%20image.png = x250)

This repository provides the code used to produce the results in my thesis project "Increasing the Predictive Power of the Possession Metric in Football by Adding Spatio-temporal Context" 
submitted for the degree Master of Science Statistics: Data Science at Leiden University.

## Abstract
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

The PDF containing the full thesis report can be found in this repository.

## Requirements

* fuzzywuzzy==0.18.0
* matplotlib==3.51
* pandas==1.24
* plotly==4.9
* scikit-learn==0.241
* seaborn==0.90
* shapely==1.7.1

## Data
The project is based on the data set ["A public data set of spatio-temporal match events in soccer competitions."](https://www.nature.com/articles/s41597-019-0247-7 "Data Descriptor").

The code used to preprocess the raw data provided under the creative commons license into the data set used in this thesis is provided.
For convenience, the preprocessed version of the data set used in this thesis, the results of the nested cross-validation in **Chapter 7**, the postprocessed data used for the application in **Chapter 8**, as well as the results of the Permutation test in **Appendix A** are provided [here](https://drive.google.com/drive/folders/1B9aaF8TcRx21tiJPoMrGMuPFYgDvzX4W?usp=sharing "Data Download").

**Download the data, put it in folder data/ and you are good to go!**

## Structure
The files are named by the chapters and sections of the report which results they reproduce.
Every Table or Figure used in a section can be reproduced by running the code in the respective file.

For **Chapter 3**, **Chapter 7** and **Chapter 8** as well as **Appendix A** there are additional scripts to reproduce the provided data sets (see Data).
Especially the nested cross-validation is computationally expensive, so only run this code with the given hardware and time.



