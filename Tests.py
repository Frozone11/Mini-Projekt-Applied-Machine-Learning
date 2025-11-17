from ucimlrepo import fetch_ucirepo 

  
# fetch dataset 
steel_plates_faults = fetch_ucirepo(id=198) 
  
# data (as pandas dataframes) 
X = steel_plates_faults.data.features 
y = steel_plates_faults.data.targets 
  
minor_faults = ["Pastry", "Stains", "Dirtiness"]
major_faults = ["Z_Scratch", "K_Scratch", "Other_Faults"]

# Compute binary target
y_minor = y[minor_faults].sum(axis=1) > 0      # returns True if any minor fault
y_major = y[major_faults].sum(axis=1) > 0      # returns True if any major fault

y_binary = y_major.astype(int)                 # major faults = 1, minor only = 0
y_binary.name = "fault_importance_binary"

print("Class distribution:")
print(y_binary.value_counts())
print(y["Bumps"].value_counts())