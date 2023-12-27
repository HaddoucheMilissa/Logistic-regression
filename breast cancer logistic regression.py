#model for breast cancer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

ds= datasets.load_breast_cancer()
x,y = ds.data, ds.target 
print(ds.target_names)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

from sklearn.linear_model import LogisticRegression 
lr= LogisticRegression()
lr.fit(x_train,y_train)
prediction= lr.predict(x_test)

def accuracy(y_true, y_prediction):
  y_true== y_prediction 
  return np.sum(y_true== y_prediction)/len(y_true)
print('accuracy class:',accuracy(y_test,prediction))
