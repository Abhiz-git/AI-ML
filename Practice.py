import pandas as pd
import numpy as np
import sklearn as tree

data = pd.read_csv("PlayPredictor.csv")

print(data.shape)

arr = np.array(data)
#print(arr.shape)
print(data.info())