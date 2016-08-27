import numpy as np
from sklearn import preprocessing,cross_validation,neighbors
import pandas as pd

#read data into pandas
df = pd.read_csv('breast_cancer_data')
# #get rid of missing data -- -99999 is an outlier and treats it like that
df.replace('?',-99999,inplace=True)
# #drop the id column because it is not needed
df.drop(['id'],1,inplace=True)


#
X = np.array(df.drop(['class'],1))
y = np.array(df['class'])


#
# #shuffle data and split out 20% for a training set
X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size = 0.2)

#
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)

accuracy = clf.score(X_test,y_test)
print(accuracy)

example_measure = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,1,2,3,2,1]])
print(len(example_measure))
example_measure = example_measure.reshape(len(example_measure),-1)
prediction = clf.predict(example_measure)
print(prediction)
