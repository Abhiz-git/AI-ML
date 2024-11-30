#####################################################################################################
#
#   Name: Abhishek Dilipkumar Nale
#   
#   Original:   
#   Date: 31/11/2024
#
#   Question 6: Write a python program to implement Polynomial Regression for predicting Salary.
#
######################################################################################################

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def Main():
    print(50*"-"+"Polynomial Regression Practice"+"-"*50)

    # reading data
    data = pd.read_csv(r'..\csv\Salary_Data.csv')

    # print(data.shape)
    print(data.info())

    x = data.iloc[:, 0:1].values # 2D array
    y = data.iloc[:, 1].values # 1D array

    plt.scatter(x,y)
    plt.plot(x,y)
    plt.show()

    slr= LinearRegression().fit(x,y)
    slr_pred = slr.predict(x)

    print("Acuracy of Slr is {:.2f}%\n".format(slr.score(x,y)*100))

#   Plot Display to decide degree
    # for i in range(5,13):
            
    #     x1 = PolynomialFeatures(degree=i).fit_transform(x)
    #     plr = LinearRegression().fit(x1,y)

    #     plt.scatter(x,y)
        
    #     plt.plot(x,plr.predict(x1))
    #     plt.show()


    x1 = PolynomialFeatures(degree=12).fit_transform(x)
    plr = LinearRegression().fit(x1,y)

    plt.scatter(x,y)
        
    plt.plot(x,plr.predict(x1))
    plt.show()

    plr_pred = plr.predict(x1)

    print("Acuracy of Plr is {:.2f}%\n".format(plr.score(x1, y)*100))
    





if __name__ == '__main__':
    Main()