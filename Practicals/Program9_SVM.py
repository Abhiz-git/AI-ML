import pandas as pd
import numpy as np
from sklearn.svm import SVC

x = np.array([[1, 2], [2,5], [4,9],[8,8],[0,1]])
y = [1,0,1,0]

data = pd.DataFrame(x)

svm = SVC(kernel='linear', C = 1.0).fit(x,y)

print("Acuracy: {:.2f}%".format(svm.score(x,y)))

# formula for line is y = mx + b where b -> intercept and m ->slope(coef)
# for SVM m = -(coef[0]/coef[1]) b = intercept[0]/coef[1]
w = svm.coef_[0]

b = -(svm.intercept_[0]/w[1])
m = -(w[0]/w[1])
xx = np.linespace(0,12)
yy = m*xx + b

import matplotlib.pyplot as plt
plt.plot(xx, yy, 'k-', label="Non Weighted Division")
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.legend()
plt.show()





