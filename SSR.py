import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("data.csv")
print("\nData from CSV before insertion:")
print(df)

X = df[['age']].values
Y = df['experiance'].values

model = LinearRegression()
model.fit(X, Y)

w1 = model.coef_[0]
w0 = model.intercept_
print(f"\nModel Coefficients:\nW1 (Slope): {w1:.4f}\nW0 (Intercept): {w0:.4f}")

X_input = int(input("\nEnter your age: "))
Y_output = model.predict([[X_input]])[0]

Y_pred = model.predict(X)
Y_mean = np.mean(Y)

SSE = np.sum((Y - Y_pred)**2)
SSR = np.sum((Y_pred - Y_mean)**2)
SST = np.sum((Y - Y_mean)**2)
R_squared = r2_score(Y, Y_pred)
MSE = mean_squared_error(Y, Y_pred)
RMSE = np.sqrt(MSE)

lower_bound = Y_output - RMSE
upper_bound = Y_output + RMSE

print(f"\nPredicted Experience for age {X_input}: {Y_output:.2f} ± {RMSE:.2f}")
print(f"Range: [{lower_bound:.2f}, {upper_bound:.2f}]")

print("\nEvaluation Metrics:")
print(f"SSE  (Sum of Squared Errors):            {SSE:.4f}")
print(f"SSR  (Regression Sum of Squares):        {SSR:.4f}")
print(f"SST  (Total Sum of Squares):             {SST:.4f}")
print(f"R²   (Coefficient of Determination):     {R_squared:.4f}")
print(f"MSE  (Mean Squared Error):               {MSE:.4f}")
print(f"RMSE (Root Mean Squared Error):          {RMSE:.4f}")

plt.scatter(X, Y, color='blue', label='Data Points')
plt.plot(X, Y_pred, color='red', label='Regression Line')
plt.xlabel('Age')
plt.ylabel('Experience')
plt.title('Age vs Experience (Linear Regression)')
plt.legend()
plt.grid(True)
plt.show()
