from copyreg import pickle
from inspect import ClassFoundException
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv("D:\Project Coding\KNN\Database_KNN.csv")
print( len(dataset))
print( dataset.head() )

zero_not_accepted = ['SizeOfStackCommit','Files','SizeOfStackReserve','MinorSubsystemVersion','Checksum','MinorImageVersion','MajorImageVersion','MinorOperatingSystemVersion','LoaderFlag','DllCharacteristicsflags']
for column in zero_not_accepted:
    dataset[column] = dataset[column].replace(0, np.NaN)
    mean = int(dataset[column].mean(skipna=True))
    dataset[column] = dataset[column].replace(np.NaN, mean)

X = dataset.iloc[:, 0:8]
y = dataset.iloc[:, 8]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

import math
print(math.sqrt(len(y_test))) #the square root here is 19 which is odd number and suits perfect to implement for KNN




#KNN 

classifier = KNeighborsClassifier(n_neighbors=19, p=2, metric='euclidean')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(y_pred)
cm = confusion_matrix(y_test, y_pred)
print (cm)
print("KNN F1 Score", f1_score(y_test, y_pred)*100)
print("KNN Accuracy", accuracy_score(y_test,y_pred)*100,"%")




#Random Forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

folds = KFold(n_splits=10)
folds.get_n_splits(X)
for train_index, test_index in folds.split(X):
    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    folds=1
    print("Accurac in fold {folds}:", accuracy_score(y_pred, y_test))
cm = confusion_matrix(y_test, y_pred)
print (cm)
print("Random Forest F1 Score",f1_score(y_test, y_pred))
print("Random Forest Accuracy", accuracy_score(y_test,y_pred))

"""
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
for train_index, test_index in loo.split(X):
    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index] 
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(y_pred)
cm = confusion_matrix(y_test, y_pred)
print (cm)
print(f1_score(y_test, y_pred))
print(accuracy_score(y_test,y_pred))
"""



# Naive Bayes

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(y_pred)
cm = confusion_matrix(y_test, y_pred)
print (cm)
print("Naive Bayes F1 Score",f1_score(y_test, y_pred))
print("Naive Bayes Accuracy",accuracy_score(y_test,y_pred))



# Support Vector Machine

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler() 
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

from sklearn.svm import SVC

classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score

print(y_pred)
cm = confusion_matrix(y_test, y_pred)
print (cm)
print("SVM F1 Score",f1_score(y_test, y_pred))
print("SVM Accuracy",accuracy_score(y_test, y_pred))




# Logistic Regression

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)

from sklearn import metrics

print(y_pred)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
print("Logistic Regression F1 Score",f1_score(y_test, y_pred))
print("Logistic Regression of Accuracy:",metrics.accuracy_score(y_test, y_pred))





def main():
    global dataset, zero_not_accepted, X, y, sc_X, X_train, X_test, classifier, y_pred
#KNeighborsClassifier(algorithm='auto',leaf_size=30, metric='euclidean',metric_params=None, n_jobs=1, n_neighbors=11, p=2, weights='uniform')



