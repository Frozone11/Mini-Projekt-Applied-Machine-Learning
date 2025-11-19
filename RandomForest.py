# DT
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# RF
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

#Both DT and RF

#Other 
from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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

# Predicting with the decision tree model
y_pred_dt = DT.predict(X_test_scaled_dt)

#Peform cross validation on the decision tree model
cv_scores_dt = cross_val_score(DT, X_dt, y_dt, cv=5)
print(f"Decision tree CV accuracy scores: {cv_scores_dt}")
print(f"Decission tree mean CV score {cv_scores_dt.mean():.2f} +/- {cv_scores_dt.std():.2f}")

# Model evalation for decision tree
accuracy_dt = accuracy_score(y_test_dt, y_pred_dt)
print(f"Test accuracy for DT: {accuracy_dt:.2f}")
print("Classification Report for DT:\n", classification_report(y_test_dt, y_pred_dt, zero_division=0))

# Plot the decision tree
plt.figure(figsize=(30,50))
plot_tree(DT, feature_names = X_dt.columns.tolist(), class_names = sorted(y_dt.unique()), filled=True)
plt.title("Decision Tree for Steel Plates Faults")
#plt.savefig("decision_tree.pdf", format="pdf", bbox_inches='tight')
plt.show()

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

# Print the results
print(f"Accuracy: {accuracy_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall: {recall_rf:.4f}")
print(f"F1 Score: {f1_rf:.4f}")
print(f"Confusion Matrix:\n {conf_matrix_rf}")


labels = sorted(y_dt.unique())
conf_matrix_rf = confusion_matrix(y_test_dt, y_pred_rf, labels=labels)
cm_df = pd.DataFrame(conf_matrix_rf, index=labels, columns=labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix (Random Forest)")
plt.show()

