#####################################################################################################
#
#   Name: Abhishek Dilipkumar Nale
#   
#   Original:   
#   Date: 29/11/2024
#
#   ML Practical:
#   Question 3:	Write a python program the Categorical values in numeric format for a given dataset.
#
######################################################################################################

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# reading csv from parent directory
data=pd.read_csv(r'..\csv\PlayPredictor.csv')       # (r) state raw_string (for path which excludes escape charachters)

df = pd.DataFrame(data)
print(df.shape,"\n")
#print(df.head(10))

# Encoding the mention Column (binary)
LabelEncoder = LabelEncoder()
label = LabelEncoder.fit_transform(df['Play'])

df['Encode']= label                                  # adds column "encode" in dataframe and shows encoding 

print(df,"\n")
print(label,"\n")
