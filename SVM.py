import pandas as pd

# dataset formation
import numpy as np

x = np.array([[1,2],[9,5],[10,11],[11,21],[13,21],[12,31]])
y = np.array((1,0)*3)

# train SVM model
from sklearn.svm import SVC

svm = SVC(kernel='linear', C=1.0).fit(x,y)

print(svm.predict([[21,12]]))

#accuracy
print(f"\nAccuracy is : {(svm.score(x,y))*100:.2f}%")

# plotting graph line = y= mx + b 
# b-> intercept m->slope-> coef_[]
# svm line => b-> intercept_[0]/coef_[1]
# svm line => m-> -coef_[0]/coef_[1]
w = svm.coef_[0]
b = -(svm.intercept_[0]/w[1])
m = -(w[0]/w[1])
xx = np.linspace(0,12)

# final equation of SVC Line is

yy = m*xx + b

#ploting
import matplotlib.pyplot as plt

plt.figure(figsize=(13,10))
plt.scatter(x[:,0], x[:,1], c=y, cmap="coolwarm")
plt.plot(xx, yy, "k-", label="Non-weighted division")
plt.legend()
plt.show()
