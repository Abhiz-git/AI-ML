import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

data = pd.read_csv(r'..\csv\insurance_data.csv')

data.info()
print(data.head())

x = data.iloc[:,0:1].values
y = data['bought_insurance']

plt.scatter(x,y)
import seaborn as sns
sns.regplot(data=data, x='age',y='bought_insurance',logistic=True)
plt.show()

LogR= LogisticRegression().fit(x,y)
y_pred = LogR.predict(x)

plt.scatter(x,y)
plt.plot(x,y_pred, 'r--')
plt.show()

print("accuracy is: {:.2f}%".format(LogR.score(x,y)*100))
