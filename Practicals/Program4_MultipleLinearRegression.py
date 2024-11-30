#####################################################################################################
#
#   Name: Abhishek Dilipkumar Nale
#   
#   Original:   
#   Date: 29/11/2024
#
#   ML Practical:
#   Question 4: Write a python program to implement Simple Linear Regression for predicting house price.
#
######################################################################################################

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


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

# Data Spliting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape,"\n")
print(X_test.shape,"\n")

print(y_train.shape,"\n")
print(y_test.shape,"\n")

# Model Training
LR = LinearRegression()
LR.fit(X_train, y_train)

print(LR.coef_,"\n")          # helps to determine the weights of independent columns on target if + then increse/uprise if - the decrease or downfall

# testing the Model
y_pred = LR.predict(X_test)
print(y_pred)

g = plt.plot((y_test - y_pred), marker='o', linestyle='')
#plt.show()

score = r2_score(y_test, y_pred)
score *= 100
print("accuracy is {:.2f}".format(score))








# # Add the target variable to the dataframe for visualization
# df_with_target = df.copy()
# df_with_target['Predicted_Price'] = LR.predict(X)

# # Pair plot using seaborn
# import seaborn as sns
# sns.pairplot(df_with_target, vars=f, diag_kind="kde", kind="reg", height=2.5)
# plt.suptitle("Pairwise Relationships", y=1.02)
# plt.show()

# # Plot the regression line (y = mx + b)
# plt.plot(X, y_pred, color='red', label='Regression Line')
# plt.show

# # # Add labels and title
# # plt.xlabel('Square Feet Living')
# # plt.ylabel('Price')
# # plt.title('Linear Regression - Price vs. Square Feet Living')

# # # Display the legend
# # plt.legend()

# # # Show the plot
# # plt.show()


