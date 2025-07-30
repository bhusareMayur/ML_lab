import pandas as pd
import numpy as np
#data types
x=10
y=10.5
z='M'
a=True
b='Mayur'
print("\nData Types:")
print(f"x: {type(x)}, y: {type(y)}, z: {type(z)}, a: {type(a)}, b: {type(b)}")


#pandas implimentation
data = pd.read_csv("data.csv")
print("\nData from CSV before insertion:")
print(data)

with open("data.csv", "a") as f:
    f.write("Shreyas,21,95,3rd\n")
    f.write("ABC,20,97,2rd\n")
    f.close()
data = pd.read_csv("data.csv")
print("\nData from CSV after insertion:")
print(data)
print("New rows added successfully!")



#numpy implimentation
arr = np.array([14,5,6,17,8,9,10])
print("\nNumpy Array:")
print("Actual Array : ",arr)
print("10 is at position : ", np.where(arr == 10))
print("Sorted array : ",np.sort(arr))
print("Array after adding 5 : ",np.add(arr,5))
print("standard deviation : ",np.std(arr))
print("Mean of array : ",np.mean(arr))
print("Median of array : ",np.median(arr))
print("Variance of array : ",np.var(arr))
print("Sum of array : ",np.sum(arr))
print("Max of array : ",np.max(arr))
print("Min of array : ",np.min(arr))
print("Index of max element : ",np.argmax(arr))
print("Index of min element : ",np.argmin(arr))
print("Array after removing 5 : ",np.delete(arr,np.where(arr == 5)))
print("Array after inserting 100 at index 2 : ",np.insert(arr,2,100))

