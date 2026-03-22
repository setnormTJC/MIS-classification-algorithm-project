

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
