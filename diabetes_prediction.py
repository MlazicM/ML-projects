# diabetes_prediction.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# Load diabetes dataset from a CSV file
diabetes = load_diabetes()
df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
df['Progression'] = diabetes.target  # Add target variable to DataFrame

# Display basic information about the dataset
print(df.head())
print("\n")
print(df.info())
print("\n")
print(df.describe())

# Graphical representation: Scatter plot of Disease Progression vs BMI
plt.figure(figsize=(11, 7))
plt.scatter(df['bmi'], df['Progression'], alpha=0.5)
plt.xlabel('Body Mass Index (BMI)')
plt.ylabel('Disease Progression')
plt.title('Disease Progression vs BMI')

# Prepare data for training
X = df.drop('Progression', axis=1)  # Features
y = df['Progression']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print("\n" + "="*50)
print("Training model using Linear Regression")
print("="*50 + "\n")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples\n")

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)
print("Model training completed.\n")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

# Print evaluation results
print("\nResults on Test Set:")
print(f"RMSE: {rmse:.4f}")
print(f"R^2 Score: {r2:.4f}\n")

# Sample prediction
for i in range(8):
    print(f" Sample {i+1}")
    print(f"  Actual Progression: {y_test.iloc[i]:.2f}")
    print(f"  Predicted Progression: {y_pred[i]:.2f}\n")
    print(f"Missed by: {abs(y_test.iloc[i] - y_pred[i]):.2f}\n")
# Graphical representation: Show plot
plt.show()
