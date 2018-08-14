import numpy as np
import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier#Implementation of the scikit-learn classifier API for Keras.
from keras.wrappers.scikit_learn import KerasRegressor#for the use ofregressor API in kEras
from sklearn.preprocessing import StandardScaler


import time
from datetime import timedelta#combination of date and time 

dataset_train=pd.read_csv("train.csv")
dataset_test=pd.read_csv("test.csv")

#Filling the missing values 
def data(df):
   df=df.drop(["Name","Ticket","Cabin"],axis=1)
   df['Age']=df['Age'].fillna(value=df['Age'].mean())
   df['Fare']=df['Fare'].fillna(value=df['Fare'].mean())
   df['Embarked'] = df['Embarked'].fillna(value=df['Embarked'].value_counts().idxmax())#value_counts returns the frequencies of letters repeated in a single coloumn
   #returns the index of the row in the coloumn which has the maximum value 
   
  
   #Mapping male and female 
   df['Sex']=df['Sex'].map({'female':0,'male':1}).astype(int)#astype converts it into interger
   
   one_hot_vector=pd.get_dummies(df['Embarked'],prefix='Embarked')#encoding categorical variables into lists imbied in a dictionary,prefix is used for dictonary mapping to coloumn names 
   df=df.drop('Embarked',axis=1)
   df=df.join(one_hot_vector)
   
   return df

dataset_train=data(dataset_train)


#Splitting the dataset
X=dataset_train.drop(['Survived'],axis=1).values
scale=StandardScaler()
X=scale.fit_transform(X)
Y=dataset_train['Survived'].values

def create_model(optimizer="adam",init="uniform"):#init says how the random weights are initialized
    model=Sequential()
    model.add(Dense(20,input_dim=X.shape[1],activation="relu"))#no of nuerons,X.shape[1] selects all the coloumns
    model.add(Dense(8,activation="relu"))
    model.add(Dense(4,activation="relu"))
    model.add(Dense(1,activation="sigmoid"))
    
    #Compiling the model 
    model.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=["accuracy"])
    return model

estimator=KerasClassifier(build_fn=create_model,epochs=200,batch_size=5,verbose=1,init='glorot_uniform',optimizer='rmsprop')
estimator.fit(X,Y)

dataset_test=data(dataset_test)
X_test=dataset_test.values.astype(float)
X_test=scale.fit_transform(X_test)

#who will survive  or die 
prediction=estimator.predict(X_test)


#save Predictions 
submission = pd.DataFrame({
    'PassengerId': dataset_test.index,
    'Survived': prediction[:,0],
})
    
submission.sort_values('PassengerId', inplace=True) 
