from statistics import mean
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

#DATA CREATION:
def create_data(hm,variance,step=2,correlation=False):
  val=1
  ys=[]
  for i in range(hm):
    y=val+random.randrange(-variance, variance)
    ys.append(y)
    if correlation and correlation=='pos':
      val+=step
    elif correlation and correlation=='neg':
      val-=step
    xs =[i for i in range(len(ys))]
  return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)
xs,ys=create_data(40,20,2,'neg')  # correlation='neg' so we see in the curve that the data is descending
plt.scatter(xs,ys,label='data')
def best_fit(xs,ys):
  m=(((mean(xs)*mean(ys))-mean(xs*ys)))/((mean(xs)**2)-mean(xs*xs))
  b= mean(ys)-m*mean(xs)
  return m,b
m,b=best_fit(xs,ys)
#calculat the regression line
regression_line=[(m*x)+b for x in xs]
plt.plot(xs,regression_line,label='regression line')
