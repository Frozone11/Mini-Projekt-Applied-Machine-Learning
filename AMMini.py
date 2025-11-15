
#import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from ucimlrepo import fetch_ucirepo 

  
# fetch dataset 
steel_plates_faults = fetch_ucirepo(id=198) 
  
# data (as pandas dataframes) 
X = steel_plates_faults.data.features 
y = steel_plates_faults.data.targets 
  
# metadata 
print("\n",steel_plates_faults.metadata) 
  
# variable information 
print("\n",steel_plates_faults.variables) 

#Binary classification for steel plates fault detectionS

# View the dimensions of the dataset
print("\nFeatures shape:", X.shape)
print("\nTargets shape:", y.shape)

"""
Here we see that there are 27 features and the last 7 columns are the target variables indicating if there is a fault or not.
We also see there are 1941 samples.
There are no missing values in the dataset.
"""

# Make a sum of the fault columns to create a binary target variable
y_fault = (y.sum(axis=1)>0).astype(int) 
y_fault.name = "defect_steel_plate" # Name the 7 fault columns defect_steel_plate

print(y_fault.value_counts())
# We can now se the 7 fault collumns have been combined into one

# Make train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_fault, test_size=0.25, random_state=42,)

# Scale the train and test features
X_train_scaled = StandardScaler().fit_transform(X_train)
X_test_scaled = StandardScaler().fit_transform(X_test)

print("\nX_train_scaled shape:", X_train_scaled.shape) # See the shape of the scaled train features 

# Here I create and fit the logistic regression model
LogReg_model = LogisticRegression(random_state=42)
LogReg_model.fit(X_train_scaled, y_train)

# Prediction for the model and evaluation 
y_predict = LogReg_model.predict(X_test_scaled)
accuracy = LogReg_model.score(y_test, y_predict)
print("\n Model accuracy:", accuracy)




