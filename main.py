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

column_names = [
    'Checking_Account_Status',
    'Loan_Duration(Months)',
    'Credit_History',
    'Purpose',
    'Credit_Amount',
    'Savings_Account',
    'Employment_Since(Months)',
    'Installment_Rate',
    'Personal_Status',
    'Other_Debtors',
    'Residence_Since(months)',
    'Property',
    'Age(Years)',
    'Other_Installment_Plans',
    'Housing',
    'Num_Existing_Credits',
    'Job_Status',
    'Num_People_Liable',
    'Telephone',
    'Foreign_Worker',
    'Extra_1',
    'Extra_2',
    'Extra_3',
    'Extra_4',
    'Bank_Decision'
]

#apply meaningful column headers
df.columns = column_names;





print(df.head());

# Apply mappings to a copy so we don't break the numeric data for the model
export_df = df.copy()

checking_map = {1: '< 0 DM', 2: '0-200 DM', 3: '> 200 DM', 4: 'No Account'}


filename = "German_Credit_Data_1994.csv";
export_df.to_csv(filename, index=False);

#asdfasdfasdf
#asdfasdfasdf
