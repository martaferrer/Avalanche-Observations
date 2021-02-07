# Avalanche-Observations

This folder contains python scripts which analyses the avalanche activity in Davos (Switzerland) during the years 1998-2019 (21 years).
A blog post from this project is published at [this](https://mferresu.medium.com/characteristics-of-avalanche-occurrence-e8a48d367ad3) website. 

## Installation
The notebook uses Python 3.7.9 (Anaconda 4.9.2 distribution).
Other libraries used within the code are:
* pandas (1.2.0) and numpy (1.19.3)  for data analysis 
* matplotlib (3.3.1) and seaborn (0.11.1) for data visualization.

## Project Motivation
Avalanche forecasting tries to predict the probability of avalanche occurrence relative to a given area and time. As a mountain lover, I was curious about knowing which features influence the most when referring to avalanche danger.

## File Description
The files included in this project are:
* AvalancheObservation.ipynb - Jupyter Notebook including the main project code.
* Utils.py - Python file including the functions to create figures
* Definitions.py - Python file including hardcoded defines used in the main code

## Project Summary
This project is a descriptive statistical study about Avalanches Observations, and it tries to find answers to the following questions:
- Is there a relation between different characteristics of avalanches?
- What defines the avalanche danger level?
- How the danger relates to avalanche snow type, trigger type and size?

Within this analysis we have seen there are some avalanche features which correlates more than others. On one hand, we found some tendencies between the snow type, trigger type or AAI (Avalanche Activity Index) with respect to the avalanche danger level. On the other hand, there are other features such as the avalanche size which shows no relationship to the avalanche risk.

This analysis was done with a non fully clean data set and its results were not reviewed by any expert. Hence, the given information is not intended to be taken as a ground truth.

## Acknowledgements & Licensing
The *Avalanche Observations* data set is provided by the SLF (Schnee und Lawinenforschung) institute and it is available in [here](https://www.envidat.ch/dataset/snow-avalanche-data-davos). This data was previously analysed and the results published by [Schweizer et al. (2020)](https://tc.copernicus.org/articles/14/737/2020/). 

All files included in this repository are free to use. 
