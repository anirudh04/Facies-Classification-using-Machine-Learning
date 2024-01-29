# Facies Classification using Machine Learning

# Facies Classification Pipeline on Machine Learning

## Overview
This project aims to develop an end-to-end machine learning pipeline for facies classification in petroleum exploration. The objective is to accurately predict lithology/facies based on well-log measurements using various machine-learning techniques. The pipeline follows best practices in data engineering and machine learning, ensuring robustness, scalability, and actionable insights for the petroleum industry.

## Pipeline Components
### Data Collection
- Gather well-log data from the Volve Dataset, including GR (Gamma-ray), RT (true resistivity), RHOB (bulk density), and NPHI (neutron porosity).
  
### Preprocessing
- Perform comprehensive preprocessing tasks:
  - Outlier detection to identify and handle outliers in the data.
  - Scaling to standardize the range of features for better model performance.
  - Co-linearity check using Pearson correlation to identify highly correlated features.
  - Association test using F-test and Mutual Information to assess the relationship between features and target.
  - Feature transformation using Quantile Transformer for improved model performance.

### Feature Engineering
- Create three new features to enhance the predictive power of the models.

### Model Development
- Utilize a diverse set of machine learning techniques including:
  - Logistic regression
  - K-Nearest Neighbor
  - Gradient Boosting Classifier
  - Adaboost
  - Gaussian Naive Bayes
  - Support Vector Machine
  - Random Forest
  - Neural Network
- Hyperparameter optimization to fine-tune model parameters for improved performance.

### Evaluation
- Evaluate classifiers using sklearn metrics:
  - F1 score: Harmonic mean of precision and recall.
  - Area under the Curve (AUC): Measure of classifier performance across all thresholds.
  - Matthews correlation coefficient: Measure of the quality of binary classifications.
- Handle class imbalance to ensure fair evaluation of model performance.

### Deployment
- Deploy top-performing models on Well in the Volve Dataset, ensuring all feature engineering and preprocessing steps are replicated accurately.

### Analysis
- Analyze predictions of the deployed models to identify:
  - Which facies are easy to predict and why.
  - Which facies are hard to predict and potential reasons.
- Generate confusion matrices to visualize model performance.

## Goals
- Develop accurate classifiers for lithology/facies classification.
- Implement a robust and scalable machine learning pipeline following best practices.
- Provide actionable insights for petroleum exploration based on facies classification results.

## Technologies Used
- Python: Programming language for data preprocessing, model development, and analysis.
- scikit-learn: Machine learning library for model development and evaluation.
- Jupyter Notebook: Interactive development environment for data exploration and experimentation.
- AWS (optional for cloud deployment): Cloud platform for scalable and cost-efficient deployment of machine learning models.

## Deliverables
- Train_Test.ipynb notebook 
- Deploy.ipynb notebook 
- Prediction.xlsx 
- Plot.pdf 

## References
- https://library.seg.org/doi/10.1190/tle35100906.1
- https://www.sciencedirect.com/science/article/pii/S0920410517308094
- https://www.sciencedirect.com/science/article/pii/S009830041930923
