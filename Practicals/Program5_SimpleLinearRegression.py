#####################################################################################################
#
#   Name: Abhishek Dilipkumar Nale
#   
#   Original:   
#   Date: 30/11/2024
#
#   ML Practical:
#   Question 5: Write a python program to implement Simple Linear Regression for predicting house price.
#
######################################################################################################

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv(r'..\csv\kc_house_data.csv')
print("\nbefore drop",df.shape)

df = df.dropna()
print("\nafter drop",df.shape)

# create a column floor wich has values of floors column in INT
df['floor'] = np.floor(df['floors']).astype(int)

x = df.loc[:, df.columns == 'floor']
y = df['price']
print(x.head())

# Data spliting
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=77)


#model training
linearRegression = LinearRegression().fit(x_train,y_train)
y_pred = linearRegression.predict(x_train) 

accuracy = r2_score(y_train,y_pred)*100 
print("accuracy = {:.2f}".format(accuracy))

# plt.scatter(x=x_train, y=y_train, marker=',', color='red', label='Actual data point')
# plt.plot(x_test, linearRegression.predict(x_test), color = 'blue', label='Prediction line')

sns.lmplot(data = df, x='floor', y='price') 

from matplotlib.ticker import MultipleLocator
plt.gca().xaxis.set_major_locator(MultipleLocator(1))

plt.xlabel('Floors')
plt.ylabel('Price')
plt.title('Linear Regression Fit')

plt.legend()
# Display the plot
plt.show()



#####################################################################################################
#
#   This is just the example of it, due to visualization i came to know simpleLinear regression is not appropriate algo.
# 
#
######################################################################################################



