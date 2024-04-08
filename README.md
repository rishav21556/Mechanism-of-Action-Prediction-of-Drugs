# Mechanism of Action (MoA) Prediction

This repository contains the code and documentation for a machine learning project focused on predicting the mechanism of action (MoA) of drugs. The project was conducted as part of Machine Learning Program at Indraprastha Institute of Information Technology, Delhi.

## Abstract

The aim of this project was to develop a predictive model capable of determining the biological activity of molecules, specifically their mechanism of action (MoA). MoA prediction is crucial in drug development as it helps understand how drugs interact with biological systems. The project employed various machine learning algorithms and techniques to achieve this goal.

## Literature Survey

The project was inspired by existing research in the field of drug discovery and MoA prediction. Several relevant papers were reviewed to understand the methodologies and techniques used in similar projects. Key insights from the literature survey informed the approach taken in this project.

## Introduction

The project was conducted in collaboration with Rishav Raj, Parth Kaushal, Shubham Pal and Garv Makkar, drawing on data provided by [Kaggle](https://www.kaggle.com/competitions/lish-moa/data). The primary objective was to develop a predictive model using machine learning algorithms to accurately classify drugs based on their MoA.

## Dataset and Preprocessing

The dataset used in this project was sourced from [Kaggle](https://www.kaggle.com/competitions/lish-moa/data), comprising gene expression and cell viability data for various drugs. The dataset was preprocessed to handle missing values, scale features, and encode categorical variables. Additional preprocessing steps such as outlier removal and dimensionality reduction were also performed.

## Methodology

Several machine learning algorithms were evaluated for MoA prediction, including Gaussian Naive Bayes, Logistic Regression, Support Vector Machines (SVM), Random Forest, K-Nearest Neighbors (KNN), Convolutional Neural Networks (CNN), and Artificial Neural Networks (ANN). Each algorithm was implemented and tuned using appropriate hyperparameters.

## Results and Analysis

The performance of each algorithm was evaluated using metrics such as log loss. Based on the results, CNN emerged as the top-performing model, achieving least loss on unseen data. The analysis of results provided insights into the strengths and limitations of each algorithm.

## Conclusion

In conclusion, the project successfully developed a predictive model for MoA prediction using machine learning techniques. The findings contribute to the field of drug discovery and demonstrate the potential of machine learning in biomedical research.
