# DT
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# DF
from sklearn.ensemble import RandomForestClassifier

#Other 
from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt


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
plt.figure(figsize=(20,10))
plot_tree(DT, feature_names = X_dt.columns.tolist(), class_names = sorted(y_dt.unique()), filled=True)
plt.title("Decision Tree for Steel Plates Faults")
plt.show()
