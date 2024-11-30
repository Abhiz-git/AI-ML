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

data = pd.read_csv(r"..\csv\homeprices.csv")
print(data.info())

# from dataset i am using only 2columns
cols = ['area','price']
df = data[cols]

# manipulating missing value
print("\nbefor drop",df.shape)
print("\nafter drop",df.dropna().shape)

# two ways of assigning labels in variable
x = df.loc[:, df.columns == 'area']           # as i am using loc method programmer must know the labels
y = df[cols[1:]]

# spliting data into train and test groups
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

#model_training
LE = LinearRegression().fit(x_train,y_train)    # here data passed for traing is return value of train_test_split() that why data is 2D array if we are using data without spliting we need convert it into 2D array 1 way for same is [[col_nm_1,...col_nm_n]]

# testing data
y_pred = LE.predict(x_test)

# checking accuracy
accuracy = r2_score(y_test, y_pred)*100
print("accuracy is {:.2f}%".format(accuracy))

import matplotlib.pyplot as plt
import seaborn as sns

#sns.scatterplot(data=df, x='area', y='price', label='datapoints')
sns.lmplot(data=df, x='area', y='price')

plt.show()







