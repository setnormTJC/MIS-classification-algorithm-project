import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

#fetch data from the web
url = "https://github.com/setnormTJC/MIS-classification-algorithm-project/raw/master/Loan%20approval%20data.csv" #note the %20 is for spaces
df = pd.read_csv(url);

# Force Pandas to show all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
rowWidth = 200; #will vary depending on IDE and zoom level
pd.set_option('display.width', rowWidth)


print(df.head());


