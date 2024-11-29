#####################################################################################################
#
#   Name: Abhishek Dilipkumar Nale
#   
#   Original:   
#   Date: 29/11/2024
#
#   ML Practical:
#   Question 2: Write a python program to find all null values in a given data set and remove them.
#
######################################################################################################


import pandas as pd

dataset = {
           'first':[100, 90, None, 43],
           'second':[30, 45, 43, None],
           'third':[None, 43,10, 9]
          }

df = pd.DataFrame(dataset)
print(df,"\n")

# return true for none entity
print(df.isnull(),"\n")

# return false for none entity
print(df.notnull(),"\n")

# fill all none values with 0
print(df.fillna(0),"\n")

# to delete rows containing none values
print(df.dropna(),"\n")

# to delete clumns containing none values
print(df.dropna(axis=1),"\n")

#to fill the value of forward entity instead of none
# print(df.fillna(method='ffill'))
print(df.ffill(),"\n")

# to fill the value of backward entity instead of none
# print(df.fillna(method='bfill'))
print(df.bfill(),"\n")











           


