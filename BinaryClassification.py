
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
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
Due to there being exactly one fault per steeel plate i will make the binary classification based 
on what I classify as major and minor faults.
"""
# Define two types of faults: removable and non-removable
minor_faults = ["Pastry", "Stains", "Dirtiness", "Bumps"]
major_faults = ["Z_Scratch", "K_Scratch", "Other_Faults"]

# Make a sum of the fault columns to create a binary target variable for major faults
y_fault = (y[major_faults].sum(axis=1)>0).astype(int)  # major faults = 1, minor only = 0

y_fault.name = "Major_fault" 

# Se the distribution of minor 0 and major 1 faults
print("HEEEEEEERRRREE:\n",y_fault.value_counts()) 


# Make train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_fault, test_size=0.25, random_state=42,stratify=y_fault)

# Scale the train and test features
X_train_scaled = StandardScaler().fit_transform(X_train)
X_test_scaled = StandardScaler().fit_transform(X_test)

print("\nX_train_scaled shape:", X_train_scaled.shape) # See the shape of the scaled train features 

# Here I create and fit the logistic regression model
# Write about hoe the weightet balancing works in the report
LogReg_model = LogisticRegression(class_weight = "balanced", random_state=42)
LogReg_model.fit(X_train_scaled, y_train)

# Prediction for the model and evaluation 
accuracy = LogReg_model.score(X_test_scaled, y_test)
print(f"\n Model accuracy: {accuracy:.3f}") #The number decides how many decimals you want to show

#Confusion Matric
y_pred = LogReg_model.predict(X_test_scaled)
cf_matrix = confusion_matrix(y_test, y_pred)

group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
labels = np.asarray(labels).reshape(2, 2)

#Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Cross validatiaon 
cv_score = cross_val_score(LogReg_model, X, y_fault, cv=5)
cross_validation = pd.DataFrame({
    "Fold Score": cv_score
})
print("\nCross Validation:")
print(cross_validation.describe())

#ROC Curve and AUC
y_pred_proba = LogReg_model.predict_proba(X_test_scaled)[::,1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Print coefficients
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': LogReg_model.coef_[0]})
coefficients = coefficients.sort_values(by='Coefficient', key=abs, ascending=False)
print("Model Coefficients:")
print(coefficients)


# Visualizing the ROC Curve
plt.figure(figsize=(8,8))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random chance")
plt.title("ROC Curve for Steel Plate Defect Classification")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid(True)
plt.legend(loc="lower right")
plt.show()

# Visualizing the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues')
plt.title("Confusion Matrix (Logistic Regression)")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=coefficients)
plt.title('Feature Importance')
plt.show()