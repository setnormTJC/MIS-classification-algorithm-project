import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 1. FETCH THE GOLD STANDARD DATA (UCI Statlog German Credit)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric"
# This version is space-delimited and has no header row
df = pd.read_csv(url, sep=r'\s+', header=None)


# Force Pandas to show all columns
pd.set_option('display.max_columns', None)
print(df.head());


# Apply mappings to a copy so we don't break the numeric data for the model
export_df = df.copy()
filename = "German_Credit_Data_1994.csv";
export_df.to_csv(filename, index=False);

#asdfasdfasdf
#asdfasdfasdf
