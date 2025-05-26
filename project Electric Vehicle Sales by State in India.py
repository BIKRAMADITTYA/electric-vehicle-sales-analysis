# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# Load the dataset
df = pd.read_csv("C:/Users/admin/Desktop/Electric Vehicle Sales by State in India/Electric Vehicle Sales by State in India.csv")

# Display first few rows
print(df.head())

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Check for missing values
print("Missing values before filling:\n", df.isnull().sum())


# Set visualization style
sns.set(style='whitegrid', palette='deep')



# EV Sales by State Over the Years
plt.figure(figsize=(12, 7))
sns.lineplot(data=df, x='Year', y='EV_Sales_Quantity', hue='State', marker='o')
plt.title('EV Sales by State Over the Years', fontsize=15)
plt.xlabel('Year')
plt.ylabel('EV Sales Quantity')
plt.legend(title='State', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()     

# EV Sales by Vehicle Category
plt.figure(figsize=(10, 6))
sns.barplot(x='Vehicle_Category', y='EV_Sales_Quantity', data=df, errorbar=None, color='red')
plt.title('EV Sales by Vehicle Category', fontsize=14)
plt.xlabel('Vehicle Category')
plt.ylabel('EV Sales Quantity')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

