# make own data set
import numpy as np

x = np.array([[1,2],[3,4],[5,6],[7,8]])
y = np.array([1,1,2,2])
test = np.array([[9,10],[10,11],[12,13],[14,15]])

# importing Naive Bayes
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB().fit(x,y)
predict=gnb.predict(test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y,predict)
print(f"Accuracy is {(accuracy)*100:.2f}")



