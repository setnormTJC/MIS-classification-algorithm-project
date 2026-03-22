import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def formatTable(pd):
    # Force Pandas to show full table with reasonable row widths
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    rowWidth = 200;  # will vary depending on IDE and zoom level
    pd.set_option('display.width', rowWidth)

def demoClassifier():
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

    #
    df.columns = column_names;

    print(df.head());

    # Apply mappings to a copy so we don't break the numeric data for the model
    export_df = df.copy()



    filename = "German_Credit_Data_1994.csv";
    export_df.to_csv(filename, index=False);

    #asdfasdfasdf
    #asdfasdfasdf


    # 2. SELECT OUR "NUTS AND BOLTS" FEATURES
    # Column 0: Duration (Months)
    # Column 1: Credit Amount (DM)
    # Column 2: Installment Rate (% of income)
    # Column 24: Target (1=Good, 2=Bad)

    # X = df[[0, 1, 2]]
    # y = df[24].map({1: 1, 2: 0}) # Map to 1 (Approved/Good) and 0 (Denied/Bad)



    # # 3. SCALE THE DATA (To fix the "Scale Trap")
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    #
    # # 4. TRAIN THE LOGISTIC REGRESSION (Gradient Descent)
    # model = LogisticRegression(solver='liblinear')
    # model.fit(X_scaled, y)
    #
    # # 5. RESULTS & VISUALIZATION
    # weights = model.coef_[0]
    # feature_names = ['Duration (Mos)', 'Credit Amount', 'Installment Rate']
    #
    # print(f"Model Accuracy on Real-World Data: {model.score(X_scaled, y)*100:.2f}%")
    # print("-" * 30)
    # for name, w in zip(feature_names, weights):
    #     print(f"{name:18} | Weight: {w:.4f}")
    #
    # plt.figure(figsize=(8, 5))
    # plt.bar(feature_names, weights, color=['darkblue', 'darkred', 'orange'])
    # plt.axhline(0, color='black', linewidth=1)
    # plt.title('UCI German Credit Data: Feature Weights')
    # plt.show()
