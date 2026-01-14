import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Load diabetes dataset
diabetes = load_diabetes()
df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
df['Progression'] = diabetes.target  # Add target variable to DataFrame

# Display basic information about the dataset
print(df.head())
print("\n")
print(df.info())
print("\n")
print(df.describe())

# Graphical representation:
plt.figure(figsize=(11, 7))
plt.scatter(df['bmi'], df['Progression'], alpha=0.5)
plt.xlabel('Body Mass Index (BMI)')
plt.ylabel('Disease Progression')
plt.title('Disease Progression vs BMI')
plt.grid(True)
plt.tight_layout()
plt.show()

# Prepare data for training
X = df.drop('Progression', axis=1)  # Features
y = df['Progression']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Print dataset sizes
print("\n" + "="*50)
print("Training model using Random Forest Regressor")
print("="*50 + "\n")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples\n")

# Cross-validation before training
print("\n" + "="*50)
print("CROSS-VALIDATION (on training set)")
print("="*50)

model = RandomForestRegressor(
    n_estimators=200, oob_score=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train,
                            cv=5, scoring='neg_mean_squared_error')
cv_rmse = (-cv_scores)**0.5
print(
    f"5-Fold Cross-Validation RMSE Scores: {cv_rmse.mean():.4f}(±{cv_rmse.std():.4f})\n")

# Train the model
print("\n" + "="*50)
print("Training model")
print("="*50)
model = RandomForestRegressor(
    n_estimators=200, random_state=42, oob_score=True)
model.fit(X_train, y_train)
print("Model training completed.\n")
print(f"OOB Score (R²): {model.oob_score_:.4f}")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)
# Print evaluation results
print("\nResults on Test Set:")
print(f"RMSE: {rmse:.4f}")
print(f"R^2 Score: {r2:.4f}\n")

# Feature importance
feature_importances = pd.DataFrame({
    'features': X.columns,
    'importance': model.feature_importances_}).sort_values(by='importance', ascending=False)
print("\n" + "="*50)
print("Feature Importances:")
print("="*50)
print(feature_importances)

# Visualization
plt.figure(figsize=(11, 7))
plt.barh(feature_importances['features'], feature_importances['importance'])
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.title('Feature Importances from Random Forest Regressor')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
# 2. Prediction vs Actual
plt.figure(figsize=(11, 7))
plt.scatter(y_test, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
plt.plot([y_test.min(), y_pred.max()], [y_test.min(), y_test.max()],
         'r--', lw=2, label='Ideal Prediction')
plt.xlabel('Actual Progression')
plt.ylabel('Predicted Progression')
plt.title('Actual vs Predicted Disease Progression')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Residuals plot
residuals = y_test - y_pred
plt.figure(figsize=(11, 7))
plt.scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
plt.axhline(0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Progression')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Disease Progression')
plt.grid(True)
plt.tight_layout()
plt.show()

# Sample prediction
for i in range(6):
    print(f" Sample {i+1}")
    print(f"  Actual Progression: {y_test.iloc[i]:.2f}")
    print(f"  Predicted Progression: {y_pred[i]:.2f}\n")
    print(f" Model missed by: {abs(y_test.iloc[i] - y_pred[i]):.2f}\n")

# Model comparison with Linear Regression)

print("\n" + "="*50)
print("Comparing with Linear Regression Model")
print("="*50 + "\n")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_y_pred = lr_model.predict(X_test)
lr_rmse = mean_squared_error(y_test, lr_y_pred) ** 0.5
lr_r2 = r2_score(y_test, lr_y_pred)
print("\nLinear Regression Results:")
print(f"RMSE: {lr_rmse:.4f}")
print(f"R^2 Score: {lr_r2:.4f}\n")

print("Random Forest:")
print(f"RMSE: {rmse:.4f}")
print(f"R^2 Score: {r2:.4f}\n")
improvment = ((lr_rmse - rmse) / lr_rmse) * 100
print(
    f"Random Forest is {improvment:.2f}% better than Linear Regression in RMSE.\n")
print("\n" + "="*50)
print("End of model comparison")
print("="*50 + "\n")
# Visualization of comparison
models = ['Random Forest', 'Linear Regression']
rmse_values = [rmse, lr_rmse]
plt.figure(figsize=(8, 5))
plt.bar(models, rmse_values, color=['purple', 'orange'], alpha=0.8, width=0.4)
plt.ylabel('RMSE')
plt.title('Model Comparison: RMSE')
plt.tight_layout()
plt.show()
# End of script
