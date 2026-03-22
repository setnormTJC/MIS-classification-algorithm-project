import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def predict_loan(duration, amount, dti, age, credits, s):
    user_df = pd.DataFrame([[duration, amount, dti, age, credits, s]], columns=factorsToTrack)
    scaled_input = scaler.transform(user_df)

    prediction = model.predict(scaled_input)[0]

    # Get the probability of the chosen outcome
    probs = model.predict_proba(scaled_input)[0]
    confidence = probs[prediction] * 100

    verdict = "Approved" if prediction == 1 else "Denied"
    return f"{verdict} ({confidence:.1f}% confidence)"


print('\n\n')

#fetch data from the web
url = "https://github.com/setnormTJC/MIS-classification-algorithm-project/raw/master/Loan_approval_data.csv" #note the RAW!
#data = pd.read_csv("Loan_approval_data.csv");
data = pd.read_csv(url)

factorsToTrack = [
    'Duration (months)',
    'Loan amount',
    'Debt-to-income ratio (percent)',
    'Age',
    'Number of credit lines',
    'Credit Score'
]

influencingFactors = data[factorsToTrack]; #add remaining ones
loanDecisions = data['Loan decision'];

# convert human-readable labels (loan decisions) to integers for playing nicely with classification algorithm
# let 0 = "Approved" and 1 = "Denied"
loanDecisions = loanDecisions.map({'Denied': 0, 'Approved': 1})

# SCALE THE DATA
scaler = StandardScaler()
scaledInfluencingFactors = scaler.fit_transform(influencingFactors)

#TRAIN THE LOGISTIC REGRESSION (Gradient Descent)
model = LogisticRegression(solver='liblinear')
model.fit(scaledInfluencingFactors, loanDecisions);

weights = model.coef_[0];

print(f"Model Accuracy on Real-World Data: {model.score(scaledInfluencingFactors, loanDecisions)*100:.2f}%")

plt.title('Relative importances for loan decision')
plt.barh(factorsToTrack, weights, color='darkblue')
plt.tight_layout()
plt.axvline(0, color='black', linewidth=0.8) # Add a vertical line at zero
plt.savefig("Weighted_loan_approval_factors.png")

plt.show()


#part 2: Visualize the confusion matrix:
predictions = model.predict(scaledInfluencingFactors)
cm = confusion_matrix(loanDecisions, predictions)

display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Denied', 'Approved'])
display.plot(cmap='Blues')
plt.title('Loan approval confusion matrix')

plt.savefig("Loan approval confusion matrix.png")

plt.show();



#part 3: predicting loan approval with some test data:
d   = float(input("1. Loan Duration (months): "))
a   = float(input("2. Loan Amount ($): "))
dti = float(input("3. Debt-to-Income Ratio (e.g. 25): "))
age = float(input("4. Applicant Age: "))
c   = float(input("5. Number of Credit Lines: "))
score   = float(input("6. Credit score: "))

prediction = predict_loan(d, a, dti, age, c, score);

print(f"Prediction: {prediction}")

# predictedLoanDecision = predict_loan(12, 5000, 25, 30, 2, 650) #example inputs
