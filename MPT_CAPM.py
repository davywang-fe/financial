import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

raw_data=pd.read_csv('/Users/xuanjiwang/Desktop/return_monthly_real.csv')
raw_data.drop([len(raw_data)-1],inplace=True)
Covariance=144*np.array(raw_data.cov())
stockReturn1=12*np.array(raw_data.mean(axis=0)).T
#calculating covariance matrix and original return prediction

data_capm=pd.read_csv('/Users/xuanjiwang/Desktop/pred_capm_monthly.csv')
stockReturn2=12*np.array(data_capm.mean(axis=0)).T
#calculating covariance matrix and return prediction via 'capm'

data_3factors=pd.read_csv('/Users/xuanjiwang/Desktop/pred_three_factor_monthly.csv')
stockReturn3=12*np.array(data_3factors.mean(axis=0)).T
#calculating covariance matrix and return prediction via '3-factor'

data_lstm=pd.read_csv('/Users/xuanjiwang/Desktop/pred_lstm_monthly.csv')
stockReturn4=12*np.array(data_lstm.mean(axis=0)).T
#calculating covariance matrix and return prediction via '3-factor'

#plt.figure()
def MPT_ploting(stockReturn, Covariance, Label):
    one=np.ones((len(stockReturn),1))
    A=float(one.T@Covariance@stockReturn)
    B=float(stockReturn.T@Covariance@stockReturn)
    C=float(one.T@Covariance@one)
    D=B*C-A**2
    
    r=np.linspace(A/C+0.2, A/C-0.2, 5000)
    sigma=np.sqrt((1+(C**2)*np.square(r-A/C)/D)/C)
    
    plt.figure()    #if you want to draw separately, move this before def
    plt.plot(sigma,r,label=Label)
    plt.xlabel('sigma') #if you want to draw separately, move the three lines before plt.show()
    plt.ylabel('return')    #if you want to draw separately, move the three lines before plt.show()
    plt.legend()    #if you want to draw separately, move the three lines before plt.show()


MPT_ploting(stockReturn1, Covariance, 'real')
MPT_ploting(stockReturn2, Covariance, 'capm')
MPT_ploting(stockReturn3, Covariance, '3factors')
MPT_ploting(stockReturn4, Covariance, 'lstm')
#plt.xlabel('sigma')
#plt.ylabel('return')
#plt.legend()
plt.show()

#================#

def CAPM_ploting(stockReturn, Covariance):
    pass
