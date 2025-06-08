# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 10:09:55 2025

@author: Dell
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('housing.csv')
df.head()   #display the first few rows of a DataFrame, providing a quick overview of its structure and contents

df.shape    #provides the dimensions of a DataFrame as a tuple, indicating its number of rows and columns.

df.info()   #provides a concise summary of a DataFrame, offering insights into its structure and contents.
df.nunique()    #returns the number of unique values for each column in a DataFrame.
df.isnull().sum()    # identifies missing values within a DataFrame.
df.duplicated().sum()    # identifies duplicate rows in a Dataframe.
df['total_bedrooms'].median()

df.head()
df.describe().T
Numerical = df.select_dtypes(include=[np.number]).columns
print(Numerical)

#Plotting Histograms

for col in Numerical:
    plt.figure(figsize=(10, 6))
    df[col].plot(kind='hist', title=col, bins=60, edgecolor='black')
    plt.ylabel('Frequency')
    plt.show()

#Plotting Boxplots

for col in Numerical:
    plt.figure(figsize=(6, 6))
    sns.boxplot(df[col], color='blue')
    plt.title(col)
    plt.ylabel(col)
    plt.show()
