# house_prediction.py
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

# Load California housing dataset
housing = fetch_california_housing()

# Create DataFrame from dataset
df = pd.DataFrame(data=housing.data, columns=housing.feature_names)
df['Price'] = housing.target  # Add target variable to DataFrame

# Display basic information about the dataset
print(df.head())
print("\n")
print(df.info())
print("\n")
print(df.describe())

# Graphical representation: Scatter plot of House Price vs Median Income
plt.figure(figsize=(10, 6))
plt.scatter(df['MedInc'], df['Price'], alpha=0.5)
plt.xlabel('Median Income (10k USD)')
plt.ylabel('House Price (100k USD)')
plt.title('House Price vs Median Income')


# Prepare data for training
X = df.drop('Price', axis=1)  # Features
y = df['Price']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("\n" + "="*50)
print("Training model using Linear Regression")
print("="*50 + "\n")
print(f"Training set: {X_train.shape[0]} houses")
print(f"Testing set: {X_test.shape[0]} houses\n")

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

# Display some sample predictions
print("Sample Predictions:")
for i in range(5):
    print(f"\nHouse {i+1}:")
    print(f"  Actual price: {y_test.iloc[i]:.2f}")
    print(f"  Predicted price: {y_pred[i]:.2f}")
    print(f"  Missed by: {abs(y_test.iloc[i] - y_pred[i]):.2f}")
plt.show()
# End of house_prediction.py
