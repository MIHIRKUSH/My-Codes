import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import talib
#Importing data
df = pd.read_excel('Zinc_Full_Script.xlsx')
df.index=df['Date/Time']
df=df['Close']

from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = df.rolling(12).mean()
    rolstd = df.rolling(12).std()

    #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
     
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    
test_stationarity(df)
plt.plot(df)

decompose_df=seasonal_decompose(df,freq=12)
plt.plot(decompose.resid)
plt.plot(decompose.seasonal)
plt.plot(decompose.trend)

#Moving Averages
moving_avg = df.rolling(3).mean()
plt.plot(df)
plt.plot(moving_avg, color='red')
df_log_moving_avg_diff = df - moving_avg
df_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(df_log_moving_avg_diff)

decompose_mv=seasonal_decompose(df_log_moving_avg_diff,freq=12)
plt.plot(decompose.resid)
plt.plot(decompose.seasonal)
plt.plot(decompose.trend)

#Test statistics is much smaller than 5% critical values
#df is not stationary trend is there 

#EWMA
df=pd.DataFrame(df)
exponential=talib.EMA(np.array(df),timeperiod=10)#half life species decay time
exponential=pd.DataFrame(exponential) 
exponential=exponential.dropna()

decompose_ex=seasonal_decompose(exponential,freq=12)
plt.plot(decompose.resid)
plt.plot(decompose.seasonal)
plt.plot(decompose.trend)

#Differencing
diff_df=df-df.shift(1)
diff_df=diff_df.dropna()
plt.plot(diff_df)
decompose=seasonal_decompose(diff_df,freq=12)
plt.plot(decompose.resid)
plt.plot(decompose.seasonal)
plt.plot(decompose.trend)


from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(df, nlags=20)
lag_pacf = pacf(df, nlags=20, method='ols')
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(diff_df)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(diff_df)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(diff_df)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(diff_df)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()


mod = SARIMAX(df,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])

results.plot_diagnostics(figsize=(16, 8))
plt.show()

pred=results.get_prediction(start=df.index.get_loc(pd.to_datetime('2018-09-03 10:00:00')),dynamic=False)
pred_ci = pred.conf_int()
y_forecasted = pred.predicted_mean
y_truth =df['2018-09-03 10:00:00':]
mse = ((y_forecasted - y_truth) ** 2).mean()
from sklearn.metrics import r2_score
r2_score(y_truth, y_forecasted)

plt.plot(y_forecasted)
plt.plot(y_truth)