import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
import seaborn as sns


data = pd.read_csv('D:\Project Coding\SVM\SVM.csv') 

print(data.shape)

print(data.head())

print(data.isnull().sum())

print(data['Files'].value_counts())

sns.countplot(data['Files'])
#plt.show()

data['Files'].value_counts().plot(kind="pie",autopct="%1.1f%%")
#plt.axis("equal")
#plt.show()

Data = data.dropna(how="any",axis=0)
print(Data.head()) 

X = Data.drop(['id', 'SizeOfHeapCommit','SizeOfHeapReserve', 'SizeOfStackCommit','SizeOfStackReserve',	'subsystem', 'MajorSubsystemVersion', 'MinorSubsystemVersion','Checksum','SizeOfHeaders','MinorImageVersion','MajorImageVersion','MinorOperatingSystemVersion','MajorOperatingSystemVersion','LoaderFlag','RVA ','BaseofCode','DllCharacteristics flags','SizeofImage','FileAlignment','SectionAlignment','ImageBase','AddressofEntryPoint','SizeOfInitializedData','SizeofCode','e_lfnew','Entropy'],axis=1)
Y = Data['Files']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.2, random_state=1)
X_train.head()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_scaled, Y_train)

Y_pred=gnb.predict(X_test)

from sklearn import metrics 
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(Y_test, Y_pred)*100)

X_new = pd.DataFrame(X_scaled, columns=X.columns)
X_new.head()

from sklearn.decomposition import PCA
skpca = PCA(n_components=1)
X_pca = skpca.fit_transform(X_new)
print('Variance sum : ', skpca.explained_variance_ratio_.cumsum()[-1])

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import classification_report, confusion_matrix

model = RFC(n_estimators=100, random_state=0, 
                         oob_score = True,
                         max_depth = 16, 
                         max_features = 'sqrt')
model.fit(X_pca, Y_train)

X_test_scaled = scaler.transform(X_test)
X_test_new = pd.DataFrame(X_test_scaled, columns=X.columns)
X_test_pca = skpca.transform(X_test_new)

Y_pred = model.predict(X_test_pca)
print(classification_report(Y_pred, Y_test))

print("Random Forest model accuracy(in %):", metrics.accuracy_score(Y_test, Y_pred)*100)

sns.heatmap(confusion_matrix(Y_pred, Y_test), annot=True, fmt="d", cmap=plt.cm.Blues, cbar=False)

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
clf = DecisionTreeClassifier()
clf = clf.fit(X_scaled,Y_train)
Y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred)*100)

X_scaled=scaler.fit_transform(X)
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict 
svm=SVC(kernel="linear")
Y_pred = cross_val_predict(svm, X_scaled, Y, cv=10)
conf_mat = confusion_matrix(Y, Y_pred)
print(conf_mat)

print("Linear SVC Classifier accuracy(in %):", metrics.accuracy_score(Y, Y_pred)*100)


