import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("C:/Users/admin/Desktop/Electric Vehicle Sales by State in India/Electric Vehicle Sales by State in India.csv")

# Check for missing values
print("Missing values before filling:\n", df.isnull().sum())

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Year'] = df['Date'].dt.year

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, 
                             columns=['State', 'Vehicle_Class', 'Vehicle_Category', 'Vehicle_Type'], 
                             drop_first=True)

# Drop unused columns
df_encoded.drop(['Date', 'Month_Name'], axis=1, inplace=True)

# Define features and target
X = df_encoded.drop('EV_Sales_Quantity', axis=1)
y = df_encoded['EV_Sales_Quantity']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=442)

# Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error (RF): {rmse:.2f}')

# Create yearly total sales
yearly_sales = df.groupby('Year')['EV_Sales_Quantity'].sum().reset_index()

# Linear Regression on yearly data
X_year = yearly_sales['Year'].values.reshape(-1, 1)
y_year = yearly_sales['EV_Sales_Quantity'].values

linear_model = LinearRegression()
linear_model.fit(X_year, y_year)

# Forecast future sales (2025–2030)
future_years = np.array(range(2025, 2031)).reshape(-1, 1)
future_preds = linear_model.predict(future_years)

# Plot actual vs forecasted sales
plt.figure(figsize=(10, 6))
plt.plot(yearly_sales['Year'], y_year, marker='o', label='Actual Sales')
plt.plot(future_years, future_preds, marker='x', linestyle='--', color='green', label='Forecasted Sales')
plt.xlabel("Year")
plt.ylabel("EV Sales Quantity")
plt.title("EV Sales Forecast in India (2025–2030)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print forecasted values
print("🔮 Forecasted EV Sales (2025–2030):")
for year, pred in zip(future_years.ravel(), future_preds):
    print(f"{year}: {int(pred):,} units")

# Feature Importance (Random Forest)
importances = rf_model.feature_importances_
features = X.columns
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
plt.title('Top 10 Feature Importances (Random Forest)')
plt.tight_layout()
plt.show()




forecast_df = pd.DataFrame({
    'Year': future_years.ravel(),
    'Predicted_EV_Sales': future_preds.astype(int)
})

forecast_df.to_csv("C:/Users/admin/Desktop/Electric Vehicle Sales by State in India/EV_Sales_Forecast_2025_2030.csv", index=False)
print(f"✅ CSV saved successfully at: {forecast_df}")
