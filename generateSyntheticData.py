import pandas as pd
import numpy as np

np.random.seed(42)
rows = 100

data = {
    'ID': range(1, rows + 1),
    'Duration (months)': np.random.randint(6, 60, rows),
    'Loan amount': np.random.randint(1000, 50000, rows),
    'Debt-to-income ratio (percent)': np.random.randint(5, 50, rows),
    'Age': np.random.randint(18, 75, rows),
    'Number of credit lines': np.random.randint(1, 10, rows),
    'Credit Score': np.random.randint(300, 850, rows)
}

df = pd.DataFrame(data)

# --- NEW REASONABLE LOGIC ---
# We calculate a "Propensity Score" (0 to 100)
# Higher Score = More likely to be Approved

# 1. Base Score from Credit Score (The anchor)
score = (df['Credit Score'] / 850) * 50

# 2. Age Bonus: "Stability Factor" (Older = +10 points max)
score += (df['Age'] / 75) * 10

# 3. Loan Amount Penalty: "Size Risk" (Larger Loan = -15 points max)
score -= (df['Loan amount'] / 50000) * 15

# 4. Duration Penalty: "Time Risk" (Longer Term = -10 points max)
score -= (df['Duration (months)'] / 60) * 10

# --- FINAL DECISION ---
# If the calculated score is > 25, they are likely approved.
# We add a small bit of randomness so it's not a "Perfect" rule.
noise = np.random.normal(0, 5, rows)
final_score = score + noise

df['Loan decision'] = np.where(final_score > 25, 'Approved', 'Denied')

# 3. SAVE IT
df.to_csv("Loan_approval_data.csv", index=False)
print("CSV Updated with Weighted Logic for Age and Loan Amount!")