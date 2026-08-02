import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load and Standardize
data = pd.read_csv(r"C:\\Users\\HP\\Downloads\\archive (1)\\diabetes.csv")

# Last column is the target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 2. Linear Regression
lin = LinearRegression().fit(X_train, y_train)

# 3. Regularized Models with CV
ridge = RidgeCV(alphas=[0.1, 1.0, 10.0]).fit(X_train, y_train)
lasso = LassoCV(cv=5).fit(X_train, y_train)

# 4. Compare
for name, model in [('Linear', lin), ('Ridge', ridge), ('Lasso', lasso)]:
    pred = model.predict(X_test)
    print(f"{name} - MSE: {mean_squared_error(y_test, pred):.2f}, R2: {r2_score(y_test, pred):.2f}")