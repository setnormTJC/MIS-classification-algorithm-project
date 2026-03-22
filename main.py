import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import functions;


print('\n\n')

#fetch data from the web
# url = "https://github.com/setnormTJC/MIS-classification-algorithm-project/raw/master/Loan_approval_data.csv" #note the RAW!
data = pd.read_csv("Loan_approval_data.csv");

functions.formatTable(pd);
#
# print(data.head());

# print(data.columns.tolist())

factorsToTrack = [
    'Duration (months)',
    'Loan amount',
    'Debt-to-income ratio (percent)',
    'Age',
    'Number of credit lines'
]

influencingFactors = data[factorsToTrack]; #add remaining ones
loanDecisions = data['Loan decision'];

# convert human-readable labels (loan decisions) to integers for playing nicely with classification algorithm
# let 0 = "Approved" and 1 = "Denied"
loanDecisions = loanDecisions.map({'Denied': 0, 'Approved': 1})

# SCALE THE DATA (To fix the "Scale Trap")
scaler = StandardScaler()
scaledInfluencingFactors = scaler.fit_transform(influencingFactors)

#TRAIN THE LOGISTIC REGRESSION (Gradient Descent)
model = LogisticRegression(solver='liblinear')
model.fit(scaledInfluencingFactors, loanDecisions);

#RESULTS & VISUALIZATION
weights = model.coef_[0];

print(f"Model Accuracy on Real-World Data: {model.score(scaledInfluencingFactors, loanDecisions)*100:.2f}%")
print("-" * 30)
for name, w in zip(factorsToTrack, weights):
     print(f"{name:30} | Weight: {w:.2f}")

plt.barh(factorsToTrack, weights, color='darkblue') # Note the 'h' in barh
plt.tight_layout()
plt.axvline(0, color='black', linewidth=0.8) # Add a vertical line at zero

plt.show()



