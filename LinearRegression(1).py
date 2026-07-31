import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

# 1. Load Dataset from CSV
dataset = pd.read_csv("/content/california_housing.csv")

# Display first 5 rows
print(dataset.head())

# 2. Select Feature (Total Rooms)
X = dataset[['total_rooms']]

# Target Variable (House Value)
y = dataset['median_house_value']

# 3. Split Dataset into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create Linear Regression Model
model = LinearRegression()

# 5. Train the Model
model.fit(X_train, y_train)

# 6. Predict House Prices
y_pred = model.predict(X_test)

# 7. Evaluate the Model
print("Mean Squared Error (MSE):",
mean_squared_error(y_test, y_pred))

print("R2 Score:", r2_score(y_test, y_pred))

# 8. Visualize the Results
plt.figure(figsize=(8,5))

# Scatter plot of actual data
plt.scatter(X_test, y_test, color='gray',
label='Actual Data')

# Sort values for proper regression line
sorted_index = X_test['total_rooms'].argsort()

plt.plot(
X_test.iloc[sorted_index],
y_pred[sorted_index],
color='red',
linewidth=2,
label='Regression Line'
)

plt.title("Linear Regression using California Housing Dataset")
plt.xlabel("total_rooms")
plt.ylabel("median_house_value")
plt.legend()

plt.show()