# Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load Dataset
df = pd.read_csv("/content/auto-mpg.csv")
print(df.head())

# Step 2: Remove Missing Values
df.dropna(inplace=True)

# Step 3: Select Feature and Target
X = df[['displacement']]
y = df['mpg']

# Step 4: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Step 6: Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

# Step 7: Performance Comparison
print("Linear Regression")
print("MSE :", mean_squared_error(y_test, y_pred_linear))
print("R2 Score :", r2_score(y_test, y_pred_linear))
print()
print("Polynomial Regression")
print("MSE :", mean_squared_error(y_test, y_pred_poly))
print("R2 Score :", r2_score(y_test, y_pred_poly))

# Step 8: Visualization
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color="gray", alpha=0.5, label="Actual Data")

# Create DataFrame instead of NumPy array
X_range = pd.DataFrame({
    "displacement": np.linspace(
        X["displacement"].min(),
        X["displacement"].max(),
        100
    )
})

# Linear Regression Line
plt.plot(
    X_range["displacement"],
    linear_model.predict(X_range),
    color="blue",
    linewidth=2,
    label="Linear Regression"
)

# Polynomial Regression Curve
plt.plot(
    X_range["displacement"],
    poly_model.predict(poly.transform(X_range)),
    color="red",
    linewidth=2,
    label="Polynomial Regression (Degree 2)"
)

plt.xlabel("Engine Displacement")
plt.ylabel("Miles Per Gallon (MPG)")
plt.title("Linear Regression vs Polynomial Regression")
plt.legend()

plt.show()