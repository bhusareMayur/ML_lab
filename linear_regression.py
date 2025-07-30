import pandas as pd
import numpy as np
import matplotlib .pyplot as plt


df = pd.read_csv("data.csv")
print("\nData from CSV before insertion:")
print(df)
X=df['age'].values
Y=df['experiance'].values
n=len(X)
mean_X = sum(X)/n
mean_Y = sum(Y)/n
print("Mean of X :",mean_X)
print("Mean of Y :",mean_Y)
numarator = 0
denomerator = 0

for i in range(n):
    numarator += ((X[i] - mean_X)*(Y[i] - mean_Y))
    denomerator += (X[i] - mean_X)**2



w1 = numarator / denomerator
print("W1 :",w1)
w0 = mean_Y - (w1 * mean_X)
print("W0 :",w0)
# print("Enter your age :")
X_input = int(input("Enter your age :"))
Y_output = w0 + w1 * X_input
print("The approximate exiperiance you have is ",Y_output)

# plot the data 
regression_Y =[]
for i in range(n):
    regression_Y.append(w0 + w1 * X[i])

plt.scatter(X,Y,color='blue',label='Data points')
plt.plot(X,regression_Y,color='red',label='regression Line')
plt.xlabel('Age')
plt.ylabel('Experience')
plt.title('Age vs Experience Regression')
plt.legend()
plt.grid(True)
plt.show()