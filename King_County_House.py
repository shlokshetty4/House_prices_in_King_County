import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("/Users/shlokshetty/kc_house_data.csv")
print(df.head())

print(df.info())
print(df.describe())
print(df.isnull().sum())

df.dropna(inplace=True)

selected_features = ['sqft_living', 'bedrooms', 'bathrooms', 'floors', 
                     'waterfront', 'view', 'condition', 'grade', 
                     'sqft_above', 'sqft_basement', 'yr_built']

X = df[selected_features]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error: $", round(mae, 2))

plt.figure(figsize=(8,5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs. Predicted House Prices")
plt.show()

new_house = [[2000, 3, 2, 1, 0, 0, 3, 7, 1500, 500, 1995]]
predicted_price = model.predict(new_house)
print("Predicted House Price: $", round(predicted_price[0], 2))
