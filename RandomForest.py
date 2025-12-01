# DT
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#poopsdf
# RF
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Both DT and RF
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Other
from ucimlrepo import fetch_ucirepo

# fetch dataset 
steel_plates_faults = fetch_ucirepo(id=198) 
  
# data (as pandas dataframes) 
X_dt = steel_plates_faults.data.features 
y1 = steel_plates_faults.data.targets 

# Checking that each steel plate has exactly one fault,\n to varify if a have to do data preprocessing
print((y1.sum(axis=1) == 1).all())

# Convert one-hot encoded targets to single label
y_dt = y1.idxmax(axis=1)
print("Fault classes:", sorted(y_dt.unique()))

# I will first create the decision tree as a baseline for the decision forest
# Added class_weight = "balanced" to handle the class imbalance
DT = DecisionTreeClassifier(criterion='gini', max_depth = 5, class_weight = "balanced", random_state = 42)

# Crating the train test split for the DT model
X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(X_dt, y_dt, test_size=0.25, random_state=42, stratify=y_dt)

# Now scale X_test and X_train
X_train_scaled_dt = StandardScaler().fit_transform(X_train_dt)
X_test_scaled_dt = StandardScaler().fit_transform(X_test_dt)

# Fit the decision tree model
DT.fit(X_train_scaled_dt, y_train_dt)

# Make predictions of the decision tree test set and the probabilities
y_pred_dt = DT.predict(X_test_scaled_dt)
y_proba_dt = DT.predict_proba(X_test_scaled_dt)

# Evaluate the performance of the decision tree model.
accuracy_dt = accuracy_score(y_test_dt, y_pred_dt)
precision_dt = precision_score(y_test_dt, y_pred_dt, average='weighted', zero_division=0)
recall_dt = recall_score(y_test_dt, y_pred_dt, average='weighted', zero_division=0)
f1_dt = f1_score(y_test_dt, y_pred_dt, average='weighted', zero_division=0)
conf_matrix_dt = confusion_matrix(y_test_dt, y_pred_dt)

#---------------------------------------------------------------------------------------------------------------------

# Now I will create the random forest model
X_rf = steel_plates_faults.data.features
y_rf = y1.idxmax(axis=1)

# Create the train test split and scale the features
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size = 0.25, random_state=42, stratify=y_rf)
X_train_scaled_rf = StandardScaler().fit_transform(X_train_rf)
X_test_scaled_rf = StandardScaler().fit_transform(X_test_rf)

# Create and fit the random forest model
rf = RandomForestClassifier(n_estimators = 100, random_state=42, class_weight = "balanced")
rf.fit(X_train_scaled_rf, y_train_rf)

# Make predictions of the test set and the probabilities
y_pred_rf = rf.predict(X_test_scaled_rf)
y_proba_rf = rf.predict_proba(X_test_scaled_rf)

# Evaluate the performance of the random forest model.
accuracy_rf = accuracy_score(y_test_rf, y_pred_rf)
precision_rf = precision_score(y_test_rf, y_pred_rf, average='weighted', zero_division=0)
recall_rf = recall_score(y_test_rf, y_pred_rf, average='weighted', zero_division=0)
f1_rf = f1_score(y_test_rf, y_pred_rf, average='weighted', zero_division=0)
conf_matrix_rf = confusion_matrix(y_test_rf, y_pred_rf)

#---------------------------------------------------------------------------------------------------------------------
# Prints and visualizations
# Print the results
print(f"Accuracy of Decision Tree: {accuracy_dt:.4f}")
print(f"Accuracy of Random Forest: {accuracy_rf:.4f}")
print(f"Precision of Decision Tree: {precision_dt:.4f}")
print(f"Precision of Random Forest: {precision_rf:.4f}")
print(f"Recall of Decision Tree: {recall_dt:.4f}")
print(f"Recall of Random Forest: {recall_rf:.4f}")
print(f"F1 Score of Decision Tree: {f1_dt:.4f}")
print(f"F1 Score of Decision Tree: {f1_rf:.4f}")
print(f"Confusion Matrix:\n {conf_matrix_dt}")
print(f"Confusion Matrix:\n {conf_matrix_rf}")

# Visualizing the Confusion Matrix
labels = sorted(y_dt.unique())
conf_matrix_dt = confusion_matrix(y_test_dt, y_pred_dt, labels=labels)
conf_matrix_rf = confusion_matrix(y_test_rf, y_pred_rf, labels=labels)
# Create DataFrames for better visualization
cm_dt = pd.DataFrame(conf_matrix_dt, index=labels, columns=labels)
cm_rf = pd.DataFrame(conf_matrix_rf, index=labels, columns=labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix (Decision Tree)")
plt.show()


plt.figure(figsize=(10, 8))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix (Random Forest)")
plt.show()

