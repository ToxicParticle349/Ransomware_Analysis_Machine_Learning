
from msilib import Feature
import os
from pyexpat import version_info
from sqlite3 import DatabaseError
from sysconfig import get_python_version
from tkinter.filedialog import Open
import pandas as pd
import numpy
import pickle
import pefile
import sklearn.ensemble as ek
#from sklearn import cross_validation, tree, linear_model
#from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn import tree
#from sklearn.externals import joblib
import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn import svm
from sklearn.linear_model import LinearRegression

database = pd.read_csv('D:\Project Coding\Saving_and_using_trained_Models\database.csv')

print(database.head)

print(database.describe())

print(database.groupby(database['Files']).size())

X = database.drop(['Files','Entropy'],axis=1).values
y = database['Files'].values

extratrees = ek.ExtraTreesClassifier().fit(X,y)
model = SelectFromModel(extratrees, prefit=True)
X_new = model.transform(X)
nbfeatures = X_new.shape[1]

print(nbfeatures)

X_train, X_test, y_train, y_test = train_test_split(X_new, y ,test_size=0.2)

features = []
index = numpy.argsort(extratrees.feature_importances_)[::-1][:nbfeatures]

for f in range(nbfeatures):
    print("%d. feature %s (%f)" % (f + 1, database.columns[2+index[f]], extratrees.feature_importances_[index[f]]))
    features.append(database.columns[2+f])

model = { "DecisionTree":tree.DecisionTreeClassifier(max_depth=10),
         "RandomForest":ek.RandomForestClassifier(n_estimators=50),
         "Adaboost":ek.AdaBoostClassifier(n_estimators=50),
         "GradientBoosting":ek.GradientBoostingClassifier(n_estimators=50),
         "GNB":GaussianNB(),
         "LinearRegression":LinearRegression()   
}

from csv import reader

for line in reader('D:\Project Coding\Saving_and_using_trained_Models\database.csv'):
    try:
        results = {}
        for algo in model:
            clf = model[algo]
            clf.fit(X_train,y_train)
            score = clf.score(X_test,y_test)
            print ("%s : %s " %(algo, score))
            results[algo] = score
    except ValueError:
        continue

winner = max(results, key=results.get)

import pickle

import joblib

joblib.dump(model[winner],'classification\model.pkl')

open('classification\\features.pkl','wb').write(pickle.dumps(features))

clf = model[winner]
res = clf.predict(X_new)
mt = confusion_matrix(y, res)
print("False positive rate : %f %%" % ((mt[0][1] / float(sum(mt[0])))*100))
print('False negative rate : %f %%' % ( (mt[1][0] / float(sum(mt[1]))*100)))

# Load classifier
clf = joblib.load('classification\model.pkl')
#load features
features = pickle.loads(open(os.path.join('classification\\features.pkl'),'rb').read())
