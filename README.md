# Machine Fault Detection
## Table of Contents
- [Problem statement](#problem-statement)
- [Dataset Used](#dataset-used)
- [Approach to Solve the Problem](#approach-to-solve-the-problem)
- [Anomaly Detection](#anomaly-detection)
- [Project Architecture](#project-architecture)
- [Sample Test Result](#sample-test-result)
- [Deployed version of this project](http://13.233.94.224:8080/)
   
&nbsp;

## Problem Statement
Predictive maintenance is used to pre-empt faults in running machines. Different types of predictive maintenance techniques are available e.g., vibration analysis, current signture analysis, thermography, partial discharge etc. Machine learning can also be used as a predictive maintenance tool. In this project a predictive model is created using past trend of parameters of an Air Production Unit (APU).

## Dataset Used
The dataset used in this project is available in [https://archive.ics.uci.edu/dataset/791/metropt+3+dataset](). From a metro train in an operational context, readings from pressure, temperature, motor current, and air intake valves were collected from a compressor's Air Production Unit (APU). 

## Approach to Solve the Problem
The dataset is a multivariate time series data where different machine parameters are availale at interval of 10 seconds from Feb, 2020 to Aug, 2020. The original dataset is unlabeled but as per failure report provided by the company **some intervals are mentioned where Air Leak was detected in the system and breakdown maintenance was carried out.**

So the data is first labeled as healthy and unhealthy and then EDA is performed. It is oberved that out of 1.5 Million timestamps only 0.2% records are available for fault data. As this dataset is highly imbalanced simple classification technique cannot be used. Simple clasification techniqe results in lot of false negatives. **So this problem is solved using anomaly detection technique.**

## Anomaly Detection
During EDA it is observed that **trends of faulty parameters are completely different from those of healthy ones**. 

![image](https://github.com/arnabroy734/machine_fault_detection/assets/86049035/5ab5f836-38db-42ec-aad3-e469d3c2556f)

Another important observation is that **the data is periodic in nature**. Even if the trends of healthy and faulty data are different peak values might be same (in above graph max/peak value of TP2 is same for healthy and faulty data both though the trend is completely different). So simply applying some anomaly detection technique will not work. **That is why smoothening (moving average - exponential and simple) is done in preprocessing step.** Simple and exponential moving average with different window sizes are used as hyperparameters and tuned for best model performance.

Two techniques of anomaly detection detection are used - **KMeans Clustering with 2 cluster and Isolation Forest**. Out of these two KMeans clustering produces better result. Please refer [**clustering logs**](/log/clustering_log.txt) and [**isolation forest logs**](/log/isolation_forest_logs.txt) for training result.

## Project Architecture
The predcive model should be integrated with real time data source. In this project real time data is simulated by **module data_synthetic** as every 1 second. At the begining the source of simulated data is healthy data slice taken from the original dataset, but there is also a **simulate fault** button to change the data source to faulty data slice. 

The real time data is fed to saved model to get the prediction and result is updated in the frontend. This process is also done at the frequency of real time data i.e., every 1 second.

![image](https://github.com/arnabroy734/machine_fault_detection/assets/86049035/1e16297b-d000-46e8-b572-35e239119316)

## Sample Test Result

At system startup when enough data is not available
![Screenshot (1)](https://github.com/arnabroy734/machine_fault_detection/assets/86049035/26af9897-226f-4ec1-9b81-325d50968525)

Healthy condition
![Screenshot (2)](https://github.com/arnabroy734/machine_fault_detection/assets/86049035/392c478f-bfcf-44cc-8308-8075ba3d0595)

Faulty condition
![Screenshot (5)](https://github.com/arnabroy734/machine_fault_detection/assets/86049035/73fbe912-132b-473b-8be7-ee871300e7be)





