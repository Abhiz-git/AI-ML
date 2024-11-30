#####################################################################################################
#
#   Name: Abhishek Dilipkumar Nale
#   
#   Original:   
#   Date: 30/11/2024
#
#   Question 6: Write a python program to implement Polynomial Regression for predicting Salary.
#
######################################################################################################


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def Main():
    print(50*"-"+"Polynomial Regression Practice"+"-"*50)

    # reading data
    data = pd.read_csv(r'..\csv\Salary_Data.csv')

    print()
    print(data.info())



if __name__ == '__main__':
    Main()