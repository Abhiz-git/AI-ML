import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'..\csv\CC GENERAL.csv')

# data cleaning
df = df.dropna()
df = df.drop('CUST_ID', axis=1)
df.info()

half=(len(df))//2

df = df.iloc[:half,]
# data normalization
from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(df)

# implementation and training of model
from sklearn.cluster import AgglomerativeClustering

ag = AgglomerativeClustering(n_clusters=4).fit_predict(x)

#plot dendogram

from scipy.cluster.hierarchy import dendrogram,linkage

linked = linkage(x, method='ward')

plt.figure(figsize=(13,10))
plt.title("Dendrogram")
dendrogram(linked)
plt.show()


