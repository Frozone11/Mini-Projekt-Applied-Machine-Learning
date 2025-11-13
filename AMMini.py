import pandas as pd


with open("/Users/hjaltefrost/Library/Mobile Documents/com~apple~CloudDocs/Robotteknologi/7. semester/Applied Machine Learning/mini/steel+plates+faults/Faults27x7_var", "r") as f:
    columns= [line.strip() for line in f.readlines()]

print(len(columns))

df = pd.read_csv(
    "/Users/hjaltefrost/Library/Mobile Documents/com~apple~CloudDocs/Robotteknologi/7. semester/Applied Machine Learning/mini/steel+plates+faults/Faults.NNA",
    sep=r"\s+",      # split on any whitespace (tabs/spaces)
    header=None,
    names=columns
)


print(df.shape)
print(df.head())