#####################################################################################################
#
#   Name: Abhishek Dilipkumar Nale
#   
#   Original:   
#   Date: 29/11/2024
#
#   ML Practical:
#   Question 4: Write a python program to implement Multiple Linear Regression for predicting house price.
#
######################################################################################################

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import seaborn as sns


data = pd.read_csv(r'..\csv\kc_house_data.csv')

df = pd.DataFrame(data)
print(df.head(),"\n")

f = ['price', 'bedrooms', 'bathrooms', 'sqft_living',
     'floors', 'condition','yr_built', 'yr_renovated']

df = df[f]
print(df.shape,"\n")

print(df.head(3),"\n")

df = df.dropna()
print(df.shape,"\n")

print(df.describe(),"\n") # use to get all aggregate function values or Stats of datasets

X = df[f[1:]]
y = df['price']

#print(df.yr_built.mean())

# displays the linear relationship with each feature vs target
for i in f:
     if(i == "yr_built"):
          sns.lmplot(data=df, x= i, y='price')
          plt.xticks(range(1900, int(df['yr_built'].max()) + 30,30 ))
          
     if i == "yr_renovated":
          df_filtered = df[df['yr_renovated'] != 0]
          sns.lmplot(data=df_filtered, x='yr_renovated' , y='price')
          plt.xticks(range(1940, int(df['yr_renovated'].max()) + 10, 5))
          
     elif i != 'price':
          sns.lmplot(data=df, x= i, y='price')

plt.show()

# Data Spliting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1000)

print(X_train.shape,"\n")
print(X_test.shape,"\n")

print(y_train.shape,"\n")
print(y_test.shape,"\n")

# Model Training
LR = LinearRegression()
LR.fit(X_train, y_train)

# print(LR.coef_,"\n")          # helps to determine the weights of independent columns on target if + then increse/uprise if - the decrease or downfall

# testing the Model
y_pred = LR.predict(X_test)

# display Residual Error
plt.plot((y_test - y_pred), marker='o', linestyle='')
plt.title('Residual Error')
plt.show()

# Accuracy of model
score = r2_score(y_test, y_pred)
score *= 100
print("accuracy is {:.2f}".format(score))

