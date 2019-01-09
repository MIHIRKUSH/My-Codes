import talib 
import numpy as np
import pandas as pd
import datetime
import time
data=pd.read_excel('zm lm 1 min jan 15.xlsx')
data['Close']=data['Zinc_Close']-data['Lead_Close']
data['Open']=data['Zinc_Open']-data['Lead_Open']
data['High']=data['Zinc_High']-data['Lead_High']
data['Low']=data['Zinc_Low']-data['Lead_Low']

macd=talib.MACD(data['Close'].values,fastperiod=24, slowperiod=52, signalperiod=18)
data['macd']=list(macd[0])
data['rsi']=talib.RSI(data['Close'].values,timeperiod=50)




signal = ''
data['Signal']=''
buy_price=''
short_price=''
data['Profit/loss']=''
data['Date Shift'] = data['Date/Time'].shift(-1)#Shifts rows value upwards



for index,row in data.iterrows():
    #print(row['Close'])
    if ( len(signal) == 0):
        if(row['macd']<0 and row['rsi']< 50):
            signal='Buy'
            data.loc[index,'Signal'] = 'Buy'
            print(signal)
            buy_price=row['Close']
        elif(row['macd']>0 and row['rsi']>50):
            signal='Short'
            data.loc[index,'Signal'] = 'Short'
            print(signal)
            short_price=row['Close']
    if(signal=='Buy'):
        if (row['macd']>0 or row['rsi']>50):
            signal='Sell'
            print(signal)
            data.loc[index,'Signal'] = 'Sell'
            data.loc[index,'Profit/loss']= row['Close']-buy_price
            signal=''
    if(signal=='Short'):
        if (row['macd']<0 or row['rsi']<50):
            signal='Cover'
            print(signal)
            data.loc[index,'Signal'] = 'Cover'
            data.loc[index,'Profit/loss']= short_price-row['Close']
            signal=''
            
    days= ( row['Date Shift']-row['Date/Time']).days
    if (days>0):
        if (signal=='Buy'):
            signal='Sell'
            print(signal)
            data.loc[index,'Signal'] = 'Sell'
            data.loc[index,'Profit/loss']= row['Close']-buy_price
            signal=''
        if(signal=='Short'):
            signal='Cover'
            print(signal)
            data.loc[index,'Signal'] = 'Cover'
            data.loc[index,'Profit/loss']= short_price-row['Close']
            signal=''  
    #print(signal)
    
    
data['Profit/loss'] = pd.to_numeric(data['Profit/loss'])
data['Profit/loss'].sum()
data['Signal'].value_counts()#Number of trades 
    


