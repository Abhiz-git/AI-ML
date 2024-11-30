import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# Step 1: Generate Two Moons Dataset
np.random.seed(0)
X, Y = make_moons(500, noise=0.1)

# Step 2: Split into Train/Test Sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=73)

# Step 3: Train a Neural Network with 10 Hidden Units
nn = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
nn.fit(X_train, Y_train)

# Step 4: Create a Grid for Plotting the Decision Boundary
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
Z = nn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Step 5: Plot Decision Boundary and Training Data
plt.figure(figsize=(12, 8))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.cividis)
plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap=plt.cm.cividis, edgecolor='k', s=20)
plt.title('Decision Boundary with Neural Network')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()
