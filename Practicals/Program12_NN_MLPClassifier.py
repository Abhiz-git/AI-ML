import pandas as pd

# load moon dataset
from sklearn.datasets import make_moons

# assign features in variables
x, y = make_moons(400, noise=0.1)

# Data Spliting
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=200)

# Neural Network model whith 10 hidden units
from sklearn.neural_network import MLPClassifier
nn = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1200, random_state=200)

# model training
nn.fit(x_train,y_train)
n_predict= nn.predict(x_test)

# model accuracy
print(f"Accuracy is {(nn.score(x_test,y_test))*100:.2f}")

#plotting Decision boundary

import matplotlib.pyplot as plt
plt.figure(figsize=(13,10))
plt.scatter(x[:, 0], x[:, 1], c=y, cmap="coolwarm")         # scattering datapoints

# data point to plot linear decision boundary
x_min, x_max =min(x[:,0]), max(x[:, 0])         
x1_min, x1_max = min(x[:,1]), max(x[:,1])

plt.plot([x_min, x_max],[x1_min, x1_max], "k-", lw=2)      # line 

plt.show()











