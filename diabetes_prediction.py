import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes

# Load the diabetes dataset
diabetes_data = load_diabetes()
df = pd.DataFrame(data=diabetes_data.data, columns=diabetes_data.feature_names)
df['target'] = diabetes_data.target

# Display the first few rows
print(df.head())


# Exploratory Data Analysis :
# Descriptive statistics
print(df.describe())

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Features')
plt.show()

# Distribution of target variable
sns.histplot(df['target'], kde=True)
plt.title('Distribution of Diabetes Progression')
plt.show()


# Train/Test Split:
# Split data into features and target variable
X = df.drop(columns=['target'])
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Machine Learning Models:

# Linear Regression:
# Train Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predictions and evaluation
y_pred_lin = lin_reg.predict(X_test)
mse_lin = mean_squared_error(y_test, y_pred_lin)
r2_lin = r2_score(y_test, y_pred_lin)

print(f'Linear Regression - MSE: {mse_lin:.2f}, R²: {r2_lin:.2f}')

# Random Forest Regressor:
# Train Random Forest
rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(X_train, y_train)

# Predictions and evaluation
y_pred_rf = rf_reg.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f'Random Forest - MSE: {mse_rf:.2f}, R²: {r2_rf:.2f}')

# Gradient Boosting Regressor:
# Train Gradient Boosting
gb_reg = GradientBoostingRegressor(random_state=42)
gb_reg.fit(X_train, y_train)

# Predictions and evaluation
y_pred_gb = gb_reg.predict(X_test)
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

print(f'Gradient Boosting - MSE: {mse_gb:.2f}, R²: {r2_gb:.2f}')


# Model Comparison:
# Compare model performance
models = ['Linear Regression', 'Random Forest', 'Gradient Boosting']
mse_values = [mse_lin, mse_rf, mse_gb]
r2_values = [r2_lin, r2_rf, r2_gb]

# Plot comparison
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.barplot(x=models, y=mse_values)
plt.title('Mean Squared Error Comparison')
plt.subplot(1, 2, 2)
sns.barplot(x=models, y=r2_values)
plt.title('R² Score Comparison')
plt.show()

