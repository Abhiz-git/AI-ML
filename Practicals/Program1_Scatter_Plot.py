#####################################################################################################
#
#   Name: Abhishek Dilipkumar Nale
#   
#   Original:   
#   Date: 29/11/2024
#
#   ML Practical:
#   Question 1:	Write a python program to Prepare Scatter Plot (Use Forge Dataset / Iris Dataset)
#
######################################################################################################


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import time as time

data = load_iris(as_frame=True)
iris = data.frame

print(iris.head(20))

start = time.time()
# Line plot for Sepal
plt.plot(iris.index, iris["sepal length (cm)"], "r--")
plt.title("Line plot for sepal")
plt.xlabel("Index")
plt.ylabel("Sepal length (cm)")
plt.show()

# Scatter Plot Sepal vs Petal
iris.plot(kind="scatter", x="sepal length (cm)", y="petal length (cm)")
plt.title("Scater plot sepal VS petal")
plt.show()
Stop = time.time()

total = Stop-start

print("total time taken {0} secs".format(round(total,2)))

