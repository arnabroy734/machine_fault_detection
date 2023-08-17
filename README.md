# Machine Fault Detection
## Table of Contents
- [Problem statement](#problem-statement)
  - [Dataset Used](#dataset-used)
  - [Approach to Solve the Problem](#approach-to-solve-the-problem)
    - [Regression model](#3-building-and-tuning-model-to-solve-the-price-prediction-problem)
    - [Recommendation system](#4-build-a-simple-recommendation-system-to-recommend-similar-laptops)
  - [Project description and architecture](#project-architecture)
    - [Module webcrawler](#1-webcrawler)
    - [Module validation](#2-validation)
    - [Module preprocessing](#3-preprocessing)
    - [Module training](#4-training)
    - [Module recommendation](#5-recommendation)
    - [Module prediction](#6-prediction)
    - [Module logs](#7-logs)
    - [Architecture of retraining pipeline](#here-is-the-architecture-of-retraining-pipeline)
  - [How to run this project](#how-to-run-this-project)
 - [Technologies used](#technologies-used)

&nbsp;

## Problem Statement
Predictive maintenance is used to pre-empt faults in running machines. Different types of predictive maintenance techniques are available e.g., vibration analysis, current signture analysis, thermography, partial discharge etc. Machine learning can also be used as a predictive maintenance tool. In this project a predictive model is created using past trend of parameters of an Air Production Unit (APU).

## Dataset Used
The dataset used in this project is available in [https://archive.ics.uci.edu/dataset/791/metropt+3+dataset](). From a metro train in an operational context, readings from pressure, temperature, motor current, and air intake valves were collected from a compressor's Air Production Unit (APU). 

## Approach to Solve the Problem
The dataset is a multivariate time series data where different machine parameters are availale at interval of 10 seconds from Feb, 2020 to Aug, 2020. The original dataset is unlabeled but as per failure report provided by the company **some intervals are mentioned where Air Leak was detected in the system and breakdown maintenance was required.**

So the data is first labeled as healthy and unhealthy and then EDA is performed. It is oberved that out of 1.5 Million timestamps only 0.2% records are available for fault data. As this dataset is highly imbalanced simple classification technique cannot be used. Simple clasification techniqe result in lot of false negatives. **So this problem is solved using anomaly detection technique.**

