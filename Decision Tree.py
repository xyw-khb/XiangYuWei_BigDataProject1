import pandas as pd
from sklearn.metrics import accuracy_score #for accuracy calculation
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus
import os

col_names = ['gameDuration', 'seasonId','winner', 'firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron', 'firstDragon','firstRiftHerald', 't1_towerKills',\
             't1_inhibitorKills','t1_baronKills','t1_dragonKills','t1_riftHeraldKills','t2_towerKills','t2_inhibitorKills','t2_baronKills','t2_dragonKills','t2_riftHeraldKills']
# load dataset
pima1 = pd.read_csv("new_data.csv", header=None, names=col_names)
pima1 = pima1.iloc[1:] # delete the first row of the dataframe
pima1.head()
feature_cols = ['gameDuration', 'seasonId', 'firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron', 'firstDragon','firstRiftHerald', 't1_towerKills',\
             't1_inhibitorKills','t1_baronKills','t1_dragonKills','t1_riftHeraldKills','t2_towerKills','t2_inhibitorKills','t2_baronKills','t2_dragonKills','t2_riftHeraldKills']
X1 = pima1[feature_cols] # Features
y1 = pima1.winner # Target variable

pima2 = pd.read_csv("test_set.csv", header=None, names=col_names)
pima2 = pima2.iloc[1:] # delete the first row of the dataframe
pima2.head()
X2 = pima2[feature_cols] # Features
y2 = pima2.winner # Target variable

X_train, X_test, y_train, y_test = X1,X2,y1,y2
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",accuracy_score(y_test, y_pred))

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
# Configure environment variables
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names =
feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes.png')
Image(graph.create_png())

