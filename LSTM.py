import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import datetime as dt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
import math
from sklearn.metrics import mean_squared_error



#Cross Validation
dataset=pd.read_excel('Zinc_Full_Script.xlsx')
Close=dataset['Close']
Close.index=dataset['Date/Time']
train=Close[:11000]
test=Close[11000:]
train, validation = train_test_split(dataset, test_size=0.40, random_state = 5)


kf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 2)
result = kf.get_n_splits(Close,y=None,groups= None)
print (result)
train = Close.iloc[0]
test =  Close.iloc[result[1]]



#Converting to df
train.index=train['Date/Time']
train=train['Close']
train=train.values.reshape(11000,1)
test=validation['Close']



#Feature scaling applying normalization
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
train=sc.fit_transform(train)

#look at 60 previous stock prices to predict the the 1 next 
X_train=[]
Y_train=[]
for i in range(60, 11000):#11000 rows 
    X_train.append(train[i-60:i,0])#Selecting 0 to 59 stock prices for learning ie (60-60 to 60)
    Y_train.append(train[i, 0])#will contain the 60th stock price which will learnt from X_train 
X_train,Y_train=np.array(X_train),np.array(Y_train)#converting onto numpy arrays

#Reshaping or creating a indicator 
X_train=np.reshape(X_train,(10940,60,1))
#Creating a indicator which will be the 3rd dimension based on the(observation,total time steps) 

from keras.models import Sequential#Iniliatize 
from keras.layers import Dense#Add the output layer
from keras.layers import LSTM#Add LSTM layer
from keras.layers import Dropout#Add some droput regulariztion to avoid overfitting 

regressor=Sequential()

regressor.add(LSTM(units= 10 ,return_sequences=True,input_shape=(60,1)))
#(number of nuerons,return the current output sequence to the next layer,
#input shape which will be accepted by the layer)
regressor.add(Dropout(0.2))#20% nuerosn will be dropped during eating iteration 

regressor.add(LSTM(units= 10 ,return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units= 10))#last LSTM layer default parameter is false 
regressor.add(Dropout(0.2))

#Output Layer
regressor.add(Dense(units=1))#number of dimensions in the output layer 

#Compiling the RNN 
regressor.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])

#Fitting the nueral network to the dataset 
history=regressor.fit(X_train,Y_train,epochs=25,batch_size=20)

print(history.history['loss'])
print(history.history['acc'])

#Plotting Validation and traning loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

test=test.values
test=test.reshape(-1,+1)
test=sc.fit_transform(test)

X_test=[]
for i in range(60, 5845):#test set contains only 5845 finacial days   
    X_test.append(test[i-60:i,0])#Selecting 0 to 59 stock prices for learning ie (60-60 to 60)
X_test=np.array(X_test)
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted_price=regressor.predict(X_test)
predicted_price=sc.inverse_transform(predicted_price)
X_test=np.reshape(X_test(X_test.shape[0],X_test.shape[1]))
test=sc.inverse_transform(test)


plt.plot(test,color='red',label='The Original')
plt.plot(predicted_price,color='blue',label='The Predicted')
plt.title('Google Stock price prediction')
plt.xlabel('Time')
plt.ylabel('Stock value')
plt.legend()
plt.show()
