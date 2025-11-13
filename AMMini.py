import pandas as pd

# Read names from Faults27x7_var
with open("Faults27x7_var", "r") as f:
    names = [line.strip() for line in f.readlines()]

print(len(names))
