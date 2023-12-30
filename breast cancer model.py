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

# LOGISTIC REGRESSION FROM SCRATCH
import numpy as np
class LogisticRegression:
  def __init__(self,learning_rate=0.01,n_iters=1000):
    self.lr = learning_rate
    self.n_iters=n_iters
    self.weights=None
    self.bias=None

  def fit(self,x,y):
     n_samples,n_features = x.shape #n_samples is the number of my datapoints
     self.weights=np.zeros(n_features)
    #approximation
     linear_model= np.dpt(x,self.weight)+self.bias
     y_pred=self.sig(linear_model)

     #compute the gradient
     dw=(1/n_samples) * np.dot(x,t(y_pred - y))
     db=(1/n_samples) * np.sum(y_pred-y)
     
     #update
     self.weights-= self.lr*dw 
     self.bias-= self.lr*db 


  def predict(self,x):
    linear_model= np.dpt(x,self.weight)+self.bias
    y_pred=self.sig(linear_model)
    y_pred_cls=[1 if i>0.5 else 0 for i in y_pred]
    return np.array(y_pred_cls)


  def sig(self,x):
    return 1/(1+np.exp(-x))