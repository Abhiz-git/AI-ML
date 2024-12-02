import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv(r'..\csv\PlayPredictor.csv')

data.info()

# data cleaning
df = data.drop(data.columns[0], axis=1)

df.info()

Encoder = LabelEncoder()

df['Whether'] = Encoder.fit_transform(df['Whether'])
df['Temperature'] = Encoder.fit_transform(df['Temperature'])

df['Play'] = Encoder.fit_transform(df['Play'])

print(df.head())

x = df.loc[:,df.columns != 'Play'].values
y = df['Play']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=14)

classifier = DecisionTreeClassifier().fit(x_train,y_train)
predict=classifier.predict(x_test)

accuracy=classifier.score(x_test,y_test)
print(f"Accuracy is {(accuracy*100):.2f}%")

plot_tree(classifier, filled=True)
plt.show()



