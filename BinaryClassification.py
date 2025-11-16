
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
Here we see that there are 27 features and the last 7 columns are the target variables indicating if there is a fault or not.
We also see there are 1941 samples.
There are no missing values in the dataset.
"""

# Make a sum of the fault columns to create a binary target variable
y_fault = (
    y[["Z_Scratch", "K_Scratch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]].
    sum(axis=1) > 0
).astype(int)
y_fault.name = "defect_steel_plate" # Name the 7 fault columns defect_steel_plate

print(y_fault.value_counts()) # We can see the binary distrubution of the target variable


# Make train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_fault, test_size=0.25, random_state=42,stratify=y_fault)

# Scale the train and test features
X_train_scaled = StandardScaler().fit_transform(X_train)
X_test_scaled = StandardScaler().fit_transform(X_test)

print("\nX_train_scaled shape:", X_train_scaled.shape) # See the shape of the scaled train features 

# Here I create and fit the logistic regression model
LogReg_model = LogisticRegression(random_state=42)
LogReg_model.fit(X_train_scaled, y_train)

# Prediction for the model and evaluation 

"""
I keep getting the error, that the data only contains one class. 
This confused me but i checked and every single data in Pastry is 1.
So there is never not an error when i use Pastry.
Therefor i will now do the Machine Learning model without Pastry.
"""
accuracy = LogReg_model.score(X_test_scaled, y_test)
print(f"\n Model accuracy: {accuracy:.3f}") #The number decides how many decimals you want to show

#Confusion Matric
y_pred = LogReg_model.predict(X_test_scaled)
cf_matrix = confusion_matrix(y_test, y_pred)

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
plt.figure(figsize=(8,8))
sns.heatmap(cf_matrix, annot=True, fmt="d")
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Visualizing the Confusion Matrix in Percentage
plt.figure(figsize=(8,8))
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='crest')
plt.title('Confusion Matrix in Percentage')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()