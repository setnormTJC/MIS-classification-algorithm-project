import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. SETUP & DATA LOADING
url = "https://github.com/setnormTJC/MIS-classification-algorithm-project/raw/master/Loan_approval_data.csv" #note the RAW!
data = pd.read_csv(url)

factorsToTrack = [
    'Duration (months)', 'Loan amount', 'Debt-to-income ratio (percent)',
    'Age', 'Number of credit lines', 'Credit Score'
]

X = data[factorsToTrack]
y = data['Loan decision'].map({'Denied': 0, 'Approved': 1})

# 2. THE TRAIN-TEST SPLIT
# 80% to train the "Robot", 20% to test it on people it hasn't met.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. SCALING (The "Scale Trap" fix)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. TRAINING
model = LogisticRegression(solver='liblinear')
model.fit(X_train_scaled, y_train)

# 5. EVALUATION
train_acc = model.score(X_train_scaled, y_train) * 100
test_acc = model.score(X_test_scaled, y_test) * 100

print(f"Training Accuracy: {train_acc:.2f}%")
print(f"Test Accuracy (Unseen Data): {test_acc:.2f}%")

# 6. VISUALIZATION (Weights & Confusion Matrix)
weights = model.coef_[0]

# Plot 1: Feature Importance
plt.figure(figsize=(10, 5))
plt.barh(factorsToTrack, weights, color='darkblue')
plt.axvline(0, color='black', linewidth=0.8)
plt.title("What Influences the Bank's Decision?")
plt.tight_layout()
plt.savefig("Weights.png")

# Plot 2: Confusion Matrix
cm = confusion_matrix(y_test, model.predict(X_test_scaled))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Denied', 'Approved'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix (Test Results)")
plt.savefig("Confusion_Matrix.png")
plt.show()


# 7. INTERACTIVE PREDICTION
def predict_loan():
    print("\n--- New Loan Application ---")
    try:
        d = float(input("1. Duration (months): "))
        am = float(input("2. Loan Amount ($): "))
        dti = float(input("3. DTI Ratio (e.g. 25): "))
        age = float(input("4. Age: "))
        cl = float(input("5. Number of credit lines you have open: "))
        scr = float(input("6. Credit Score (300-850): "))

        # Create DF to match training features (avoids warnings)
        user_df = pd.DataFrame([[d, am, dti, age, cl, scr]], columns=factorsToTrack)
        user_scaled = scaler.transform(user_df)

        pred = model.predict(user_scaled)[0]
        prob = model.predict_proba(user_scaled)[0][pred] * 100

        verdict = "APPROVED" if pred == 1 else "DENIED"
        print(f"\nDecision: {verdict} ({prob:.1f}% confidence)")
    except ValueError:
        print("Invalid input. Please enter numbers only.")


predict_loan()