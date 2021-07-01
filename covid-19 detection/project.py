import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import  PolynomialFeatures
from sklearn import Linear_model


### LOAD DATA ###
data=pd.read_csv("StatewiseTestingDetails.csv", sep=',')
data= data[['TotalSamples','Positive']]
print('-'*30);print('HEAD');print('-'*30)
print(data.head())

### PREPARE DATA  ###
print('-'*30);print('PREPARE DATA');print('-'*30)
x = np.array(data['TotalSamples']).reshape(-1,1)
y = np.array(data['Positive']).reshape(-1,1)
#plt.plot(y, '-m')
#plt.show()

polyfit =  PolynomialFeatures(degree=2)
x=polyfit.fit_transform(x)
print(x)

### TRAINING DATA ###
print('-'*30);print('TRAINING DATA');print('-'*30)
model = linear_model.LinearRegression()
model.fit(x,y)
accuracy = model.score(x,y)
print(f'Accuracy:{round(accuracy*100,3)} %')
y0=model.predict(x)
plt.plot(y0,'--b')
plt.show()

### PREDICTION ###
days=5
print('-'*30);print('PREDICTION');print('-'*30)
print(f'Prediction - Cases after{days} days:',end='')
print(round(int(model.predict(polyfit.fit_transform([[135596+days]])))/1000,2),'Thousands')

x1=np.array(list(range(1,135596+days))).reshape(-1,1)
y1=model.predict(polyfit.fit_transform(x1))
plt.plot(y1, '--r')
plt.plot(y0,'--b')
plt.show()