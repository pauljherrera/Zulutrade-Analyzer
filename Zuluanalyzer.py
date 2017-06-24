# -*- coding: utf-8 -*-
"""
Created on Mon May 16 09:46:02 2016

Zuluanalyzer.py

@author: Paúl Herrera

TODO: - aclarar el valor de cada estadístico (pips o $)
        - normalizar los lotes (que todo se haga con 0.01 lotes)
"""

from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from datetime import datetime as dt 
import os
from tqdm import tqdm



def import_csv(path, csv):
    
    csvpath = os.path.join(path,csv)
    df = pd.read_csv(csvpath, index_col='Date Close')
    df.index = df.index.to_datetime()
    df = df.drop(df.columns[[0, 1, 2, 3, 4, 10]], axis=1)
    df = df.sort_index(ascending=True)
    df = df.rename(columns = {'Profit ($)':'Returns'})
    df = df.rename(columns = {'Profit (€)':'Returns'})
    df = df.rename(columns = {'Profit (Pips)':'Pips'})

    return df
  
def stats(df):
    num_ops = len(df)
    df['CumRet'] = np.cumsum(df['Returns'])
    df['CumRet (Pips)'] = np.cumsum(df['Pips'])
    dollar_prof = round(df['CumRet'][-1], 2)
    pips_prof = df['CumRet (Pips)'][-1]
    winners = len(df.ix[df['Returns'] > 0])
    losers = len(df.ix[df['Returns'] < 0])
    win_pct = float(winners) / num_ops
    los_pct = float(losers) / num_ops
    win_mean = df.ix[df['Returns'] > 0].mean()['Returns']
    los_mean = df.ix[df['Returns'] < 0].mean()['Returns']    
    Ratio_PL = -(win_mean / los_mean)
    mean = df['Returns'].mean()
    std = df['Returns'].std()
    sharpe = mean/std
    SQN = sharpe * np.sqrt(num_ops)
    win_max = np.max(df['Returns'])
    los_max = np.min(df['Returns'])
    maxdd = round(np.max(np.maximum.accumulate(df['CumRet']) - df['CumRet'] ), 
                  2)
    calmar = dollar_prof / maxdd
    #Making a DataFrame with the statistics    
    statsnames = ['num_ops', 'months', 'dollar_prof', 'pips_prof', 'winners', 
                  'losers', 'win_pct', 'los_pct', 'win_mean', 'los_mean',
                  'Ratio_PL', 'mean', 'std', 'sharpe', 'SQN', 'monthly_sharpe',
                  'yearly_sharpe', 'win_max', 'los_max', 'maxdd', 'calmar']
    statsvalues = [num_ops, 0, dollar_prof, pips_prof, winners, losers, win_pct, 
             los_pct, win_mean, los_mean, Ratio_PL, mean, std, sharpe, SQN, 0,
             0, win_max, los_max, maxdd, calmar]
    statistics = pd.DataFrame(statsvalues)
    statistics.index = statsnames
    statistics.columns = ['Values']
    
    return statistics
    
def monthly_returns(df, statistics):
    """
    Monthly returns. 
    It has a very complicated loop. That loop creates a dictionary with a 
    Dataframe for every trading month. The tradingmonths are listed too.
    The loops creates a dataframe in a different way for december. It also
    drops the non-trading months
    """    
    retbymonth = {}
    tradingmonths = list()
    years = [2012,2013,2014,2015,2016]
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct',
              'Nov','Dec']  
    i = 0
    for y in years:
        for m, month in enumerate(months):
            if m == 11:
                retbymonth[i] = \
                    pd.DataFrame(df.loc[(df.index > dt(y,(m+1),1)) & 
                    (df.index < dt((y+1),1,1))]['Returns'])
            else:
                retbymonth[i] = \
                    pd.DataFrame(df.loc[(df.index > dt(y,(m+1),1)) & 
                    (df.index < dt(y,(m+2),1))]['Returns'])
            #Dropping non-trading months
            if retbymonth[i].empty == True:
                del retbymonth[i]
            else:
                tradingmonths.append('%s%s' %(month, y))
            i = i + 1
    #Renaming keys
    for i, key in enumerate(retbymonth.keys()):
        retbymonth[i] = retbymonth.pop(key) 
    
    #Monthly returns
    monthlyret = {}
    for m in xrange(len(tradingmonths)):
        monthlyret[m] = retbymonth[m].sum()
    monthlyret = pd.DataFrame(monthlyret).T 
    monthlyret['CumRet'] = monthlyret['Returns'].cumsum()
    monthlyret['Months'] = tradingmonths
    statistics['Values']['months'] = len(monthlyret)
    statistics['Values']['monthly_sharpe'] = (monthlyret['Returns'].mean() /
                                        monthlyret['Returns'].std())
    statistics['Values']['yearly_sharpe'] = (statistics['Values']['monthly_sharpe'] *
                                        np.sqrt(12))
                
    return monthlyret, statistics
    
def sharpe_stabilization(df):
    df['Sharpe_evol'] = (df.Returns.expanding(min_periods=1).mean() / 
                    df.Returns.expanding(min_periods=1).std())
        
    return df

def bootstrap(df, n=1000):
    plt.figure(4, figsize=(11.5, 7))
    plt.title('Bootstrap. %i samples' %n)
    max_btstp_dd = 0 
    max_btstp_mean = 0
    min_btstp_mean = 'inf'
    max_btstp_std = 0
    min_btstp_std = 'inf'
    max_btstp_ret = 0
    min_btstp_ret = 'inf'
    #External loop: repeats sampling    
    for j in xrange(n):
        x = np.zeros(len(df))
        #Makes a new sample
        for i in xrange(len(df)):
            rand = int(np.floor(np.random.rand() * len(df)))
            x[i] = round(df['Returns'][rand], 2)
        #statistics calculation
        x_cumsum = np.cumsum(x)
        plt.plot(x_cumsum)
        dd = round(np.max(np.maximum.accumulate(x_cumsum) - x_cumsum ), 2)
        mean = x.mean()
        std = x.std()
        #Updating statistics
        if dd > max_btstp_dd:
            max_btstp_dd = dd
        if mean > max_btstp_mean:
            max_btstp_mean = mean
        if (mean < min_btstp_mean):
            min_btstp_mean = mean
        if std > max_btstp_std:
            max_btstp_std = std
        if std < min_btstp_std:
            min_btstp_std = std
        if x_cumsum[-1] < min_btstp_ret:
            min_btstp_ret = x_cumsum[-1]
        if x_cumsum[-1] > max_btstp_ret:
            max_btstp_ret = x_cumsum[-1]
    max_btstp_sharpe = max_btstp_mean / max_btstp_std
    min_btstp_sharpe = min_btstp_mean / min_btstp_std
    #Creates statistics DataFrame    
    btstp_stats = pd.DataFrame([max_btstp_dd, min_btstp_ret, max_btstp_ret,
                                min_btstp_sharpe, max_btstp_sharpe])
    btstp_stats.index = ['max_btstp_dd', 'min_btstp_ret', 'max_btstp_ret',
                         'min_btstp_sharpe', 'max_btstp_sharpe']
    btstp_stats.columns = ['Values']

    
    return btstp_stats

def plot(df, monthlyret):
    #Equity curves
    plt.figure(1, figsize=(11.5,20))
    plt.subplot(511)
    plt.plot(df['CumRet'])
    plt.title('Equity Curve')
    plt.subplot(512)
    plt.plot(np.cumsum(df['Pips']))
    plt.title('Profit in Pips')
    plt.subplot(513)
    plt.plot(df['Returns'])
    plt.plot(df.Returns.rolling(window=100, center=False).mean())
    plt.plot(df.Returns.rolling(window=50, center=False).mean())
    #plt.plot(pd.rolling_mean(df['Returns'], window=100))
    plt.title('Behavior ($)')
    plt.subplot(514)
    plt.bar(xrange(len(df)), df['Pips'])
    plt.title('Behavior (Pips)')
    plt.subplot(515)
    plt.bar(xrange(len(df)), df['Highest Profit (Pips)'], color='green')
    plt.bar(xrange(len(df)), df['Worst Drawdown (Pips)'], color='red')
    plt.title('Pips + vs. -')
    
    #Monthly returns plots
    plt.figure(2, figsize=(11.5,9))
    plt.subplot(221)
    plt.title('Monthly Equity Curve')
    plt.plot(np.ravel(monthlyret['CumRet']))
    plt.subplot(222)
    plt.title('Monthly Histogram')
    plt.hist(np.ravel(monthlyret['Returns']), bins=7)
    plt.subplot(223)    
    plt.title('Monthly Returns / Biannual rolling mean and std')
    plt.plot(monthlyret.Returns.rolling(window=6, center=False).mean(),
                             label='Mean')
    plt.plot(monthlyret.Returns.rolling(window=6, center=False).std(),
                            label='Std')
    plt.plot(np.ravel(monthlyret['Returns']), label='Returns')
    plt.legend()
    plt.subplot(224)
    plt.title('Yearly Rolling Sharpe')
    plt.plot((monthlyret.Returns.rolling(window=12, center=False).mean() /
        monthlyret.Returns.rolling(window=12, center=False).std()))

    #Sharpe stabilization
    plt.figure(3, figsize=(11.5,6))
    plt.title('Sharpe Stabilization')
    plt.plot(abs(df['Sharpe_evol'][100:-1] - 
            df['Sharpe_evol'][100:-1].shift()), label='Delta')
    plt.plot(abs(df['Sharpe_evol'][100:-1] - 
            df['Sharpe_evol'][100:-1].shift()).expanding(min_periods=1).mean(), 
            label='Mu')
    plt.legend()
    

                 
if __name__ == '__main__':
    
    #Variables    
    path = "csvfiles/seg_prueba"
    
    #Functions
    files = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        files.extend(filenames)
        break    
    
    for csv in files:
        df = import_csv(path, csv)
        statistics = stats(df)
        monthlyret, statistics = monthly_returns(df, statistics)
        df = sharpe_stabilization(df)
        plot(df, monthlyret)
        btstp_stats = bootstrap(df, n=1000)
        
        #Prints
        print('\n \n %s \n' %csv)
        print('--------General Statistics--------\n')   
        print(statistics)
        print('\n-----Bootstrapped Statistics------\n')   
        print(btstp_stats)
        print('')
        plt.show()

    