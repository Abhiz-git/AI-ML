import pandas as pd
from sklearn.datasets import load_iris

# dataset loaded
df = load_iris()

print(dir(df))
print(df.data.shape)

# variables initialized
x = df['data']
y = df['target']

# Spliting data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.4, random_state=64)

# Model training
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=50).fit(x_train,y_train)
y_predict=clf.predict(x_test)

# Accuracy
print(f"\nAccuracy is : {(clf.score(x_test,y_test))*100:.2f}%")

# classification report
from sklearn.metrics import classification_report as clfr

print(clfr(y_test,y_predict))

# feature importance

imp_feature = clf.feature_importances_

feature = df['feature_names']

table_impFeatures = pd.DataFrame({'Importance': imp_feature, 'Features': feature }).sort_values(by='Importance', ascending=False)

print(table_impFeatures)

# Plotting Importance using graph
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,6))
sns.barplot(data=table_impFeatures, x=table_impFeatures['Features'], y=table_impFeatures['Importance'])
plt.show()










  