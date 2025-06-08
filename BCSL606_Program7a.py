# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 10:30:14 2025

@author: Madhuri
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("Boston_housing_dataset.csv")
data.shape
data.info()
import seaborn as sns
df = data.copy()

numerical = df.select_dtypes(include=['int','float']).columns
categorical = df.select_dtypes(include=['object']).columns
print(numerical)
print(categorical)
df.isnull().sum()

df['CRIM'].fillna(df['CRIM'].mean(), inplace=True)
df['ZN'].fillna(df['ZN'].mean(), inplace=True)
df['CHAS'].fillna(df['CHAS'].mode()[0], inplace=True)
df['INDUS'].fillna(df['INDUS'].mean(), inplace=True)
df['AGE'].fillna(df['AGE'].median(), inplace=True) # Median is often preferred for
df['LSTAT'].fillna(df['LSTAT'].median(), inplace=True)

df.isnull().sum()


corr_data = df[numerical].corr(method='pearson')

plt.figure(figsize=(10, 8)),
sns.heatmap(corr_data, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.xticks(rotation=90, ha='right')
plt.yticks(rotation=0)
plt.title("Correlation Matrix Heatmap")
plt.show()


X = df.drop('MEDV', axis=1) # All columns except 'MEDV'
y = df['MEDV'] # Target variable
# Scale the features
scale = StandardScaler()
X_scaled = scale.fit_transform(X)
# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled , y, test_size=0.2,random_state=42)
# Initialize the linear regression model
model = LinearRegression()
# Fit the model on the training data
model.fit(X_train, y_train)
# Predict on the test set
y_pred = model.predict(X_test)
y_pred
# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
# Calculate R-squared value
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual MEDV")
plt.ylabel("Predicted MEDV")
plt.title("Actual vs Predicted MEDV")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # perfect prediction line
plt.show()