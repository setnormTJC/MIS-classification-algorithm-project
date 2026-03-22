import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

#fetch data from the web
# url = "TBC"
df = pd.read_csv("Loan approval data.csv"); #replace with URL later

# Force Pandas to show all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
rowWidth = 200; #will vary depending on IDE and zoom level
pd.set_option('display.width', rowWidth)


print(df.head());


