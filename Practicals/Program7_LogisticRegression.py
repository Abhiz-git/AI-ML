import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

data = pd.read_csv(r'..\csv\insurance_data.csv')

data.info()
print(data.head())

x = data.iloc[:,0:1].values
y = data['bought_insurance']

# visualization of possible curve on actual datapoints
plt.scatter(x,y)
import seaborn as sns
sns.regplot(data=data, x='age',y='bought_insurance',logistic=True)
plt.show()

# Model Training
LogR= LogisticRegression().fit(x,y)
y_pred = LogR.predict(x)


# s-curve prediction vs actual data points
import numpy as np
X_range = np.linspace(x.min() - 1, x.max() + 1, 12).reshape(-1, 1)
y_prob = LogR.predict_proba(X_range)[:, 1]  # Predicted probabilities

# Accuracy
print("accuracy is: {:.2f}%".format(LogR.score(x,y)*100))

# Plot the data points and S-curve
plt.scatter(x, y, color='blue', label='Actual Data', alpha=0.6)

plt.plot(X_range, y_prob, color='red', label='S-Curve')
plt.xlabel('Feature')
plt.ylabel('Probability / Class')
plt.title('Logistic Regression Prediction (S-Curve)')
plt.legend()
plt.show()
