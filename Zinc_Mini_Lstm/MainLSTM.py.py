import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

dataset_train=pd.read_excel('5 min dat.xlsx')
training_set = pd.DataFrame(index=range(0,len(dataset_train)),columns=['Date', 'Close'])#initialize for the for loop

for i in range(0,len(dataset_train)):
     training_set['Date'][i] = dataset_train['Date'][i]
     training_set['Close'][i] = dataset_train['Close'][i]
training_set.index=training_set['Date']
training_set=training_set.drop(['Date'],axis=1).values
training_set=training_set.values
#Feature scaling applying normalization
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
training_set_scaled=sc.fit_transform(training_set) 

#look at 60 previous stock prices to predict the the 1 next 
X_train=[]
Y_train=[]
for i in range(60, 11998):#11998 rows 
    X_train.append(training_set_scaled[i-60:i,0])#Selecting 0 to 59 stock prices for learning ie (60-60 to 60)
    Y_train.append(training_set_scaled[i, 0])#will contain the 60th stock price which will learnt from X_train 
X_train,Y_train=np.array(X_train),np.array(Y_train)#converting onto numpy arrays

#Reshaping or creating a indicator 
X_train=np.reshape(X_train,(11938,60,1))
#Creating a indicator which will be the 3rd dimension based on the(observation,total time steps) 

from keras.models import Sequential#Iniliatize 
from keras.layers import Dense#Add the output layer
from keras.layers import LSTM#Add LSTM layer
from keras.layers import Dropout#Add some droput regulariztion to avoid overfitting 

regressor=Sequential()

regressor.add(LSTM(units= 50 ,return_sequences=True,input_shape=(60,1)))
#(number of nuerons,return the current output sequence to the next layer,
#input shape which will be accepted by the layer)
regressor.add(Dropout(0.2))#20% nuerosn will be dropped during eating iteration 

regressor.add(LSTM(units= 50 ,return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units= 50 ,return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units= 50))#last LSTM layer default parameter is false 
regressor.add(Dropout(0.2))

#Output Layer
regressor.add(Dense(units=1))#number of dimensions in the output layer 

#Compiling the RNN 
regressor.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])

#Fitting the nueral network to the dataset 
regressor.fit(X_train,Y_train,epochs=20,batch_size=20)

#Choosing the test data for comparing 
dataset_test=pd.read_excel('5 min data test.xlsx')
testing_set = pd.DataFrame(index=range(0,len(dataset_test)),columns=['Date', 'Close'])#initialize for the for loop
for i in range(0,len(dataset_test)):
     testing_set['Date'][i] = dataset_test['Date/Time'][i]
     testing_set['Close'][i] = dataset_test['Close'][i]
testing_set.index=testing_set['Date']
testing_set=testing_set.drop(['Date'],axis=1).values



dataset_all=pd.concat((dataset_train['Close'],dataset_test['Close']),axis=0)
#Concatinating both the open coloums of the  datasets along the vertical axis 

inputs=dataset_all[len(dataset_all)-len(dataset_test)-60:].values

inputs=inputs.reshape(-1,+1)

inputs=sc.transform(inputs)


X_test=[]
for i in range(60, 4907):#test set contains only 4847 finacial days   
    X_test.append(inputs[i-60:i,0])#Selecting 0 to 59 stock prices for learning ie (60-60 to 60)
X_test=np.array(X_test)
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted_price=regressor.predict(X_test)
predicted_price=sc.inverse_transform(predicted_price)

plt.plot(testing_set,color='red',label='The Original')
plt.plot(predicted_price,color='blue',label='The Predicted')
plt.title('Stock price prediction')
plt.xlabel('Time')
plt.ylabel('Stock value')
plt.legend()
plt.show()
