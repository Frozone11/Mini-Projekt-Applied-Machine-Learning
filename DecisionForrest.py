from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from ucimlrepo import fetch_ucirepo 

  
# fetch dataset 
steel_plates_faults = fetch_ucirepo(id=198) 
  
# data (as pandas dataframes) 
X = steel_plates_faults.data.features 
y = steel_plates_faults.data.targets 

print(y.sum(axis=1))

DF = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
