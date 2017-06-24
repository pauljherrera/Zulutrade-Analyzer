# -*- coding: utf-8 -*-
"""
Created on Thu May 19 12:59:42 2016

@author: Paúl
"""
from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize as opt
import seaborn as sbn
from datetime import datetime as dt
import os 

from Zuluanalyzer import import_csv
from Zuluanalyzer import monthly_returns
from Zuluanalyzer import stats


def csv_path(path):
    csvs = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        csvs.extend(filenames)
        break    

    return csvs

def import_csvs(path, csvs, startdate):
    """
    Creates a dictionary with all the candidates for the portfolio named 
    'strategies'. It uses a list of csv files as an input. It also uses a
    start date so all the strategies have a return matrixm of the same lenght
    """
    strategies = {}
    for i, strat in enumerate(csvs):
        df = import_csv(path, csvs[i])
        #Leaving one year returns
        df = df.loc[(df.index >= startdate)]
        strategies[i] = df
    
    return strategies
        
def stats_monthlyrets(strategies):
    """
    Creates two dictionaries (statistis and monthlyreturtns) with the same 
    keys as the 'strategies' dictionary.
    """
    #Applying the Zuluanalysis stats function
    statistics = {}
    for i in strategies.keys():    
        statistics[i] = stats(strategies[i])
    #Applying the Zuluanalysis monthlyret function
    monthlyreturns = {}
    for i in strategies.keys():    
        (monthlyreturns[i], statistics[i]) = monthly_returns(strategies[i], 
                                                    statistics[i])
     
    return (statistics, monthlyreturns)

def correlation_matrix(monthlyreturns):
    #Getting monthly returns
    ret_mat = np.zeros(shape=(len(monthlyreturns[0]), len(monthlyreturns)))
    for i in xrange(len(monthlyreturns)):
        print(monthlyreturns[i])
        ret_mat[:,i] = monthlyreturns[i]['Returns']
    """
    #Getting excess returns
    er_mat = np.zeros_like(ret_mat)
    for i in xrange(len(ret_mat[0,:])):
        er_mat[:,i] = ret_mat[:,i] - np.mean(ret_mat[:,i])
    #Calculating matrices
    XtX = np.dot(er_mat.T, er_mat)
    varcov_mat = XtX / (len(er_mat))
    sigmas = np.zeros(len(XtX))
    for i in xrange(len(sigmas)):
        sigmas[i] = np.std(ret_mat[:,i])
    sigmast = np.array([sigmas]).T
    ddT = np.dot(sigmast, sigmast.T)
    corr_mat = varcov_mat / ddT
    """
    #Correlation matrix    
    corr_mat = pd.DataFrame(ret_mat).corr()    
    
    return corr_mat, ret_mat

def portfolio_optimization(ret_mat, minWeight, maxWeight):
    mus = np.zeros(len(ret_mat[0,:]))
    for i in xrange(len(mus)):
        mus[i] = np.mean(ret_mat[:,i])
    equal_w = np.zeros(len(mus))
    equal_w[:] = 1.0 / len(mus)
    #Creating the boundaries and constraints for optimization
    const = [{'type':'eq', 'fun': const_1}]
    bnds = list()
    for i in xrange(len(equal_w)):
        bnds.append((minWeight, maxWeight))
    #Optimizing
    opt_w = opt.minimize(sharpe_optimizable, equal_w, method='SLSQP',
                         bounds=bnds, constraints=const)
    optimal_w = opt_w.x
    iterations = opt_w.nit
    sharpe_ratio = sharpe(optimal_w)
    # Putting optimal_w in a Dataframe for visualization
    weights = pd.DataFrame()
    strats = []
    for i in xrange(len(csvs)): 
        strats.append(csvs[i][13:-22])
    weights['strategies'] = strats
    weights['weight'] = optimal_w.round(3)
    
    return equal_w, optimal_w, sharpe_ratio, iterations, weights
 
def sharpe(w):
    return (np.dot(w, ret_mat.T).mean() / np.dot(w, ret_mat.T).std()) * \
                np.sqrt(12)
   
def sharpe_optimizable(w):
    return -(np.dot(w, ret_mat.T).mean() / np.dot(w, ret_mat.T).std())
    
def const_1(x):
    return x.sum() -1
     
if __name__ == '__main__':
    
    #Variables    
    path = "csvfiles"
    startdate = dt(2015,10,1)
    minWeight = 0.025
    maxWeight = 0.6
    
    #Function calls
    csvs = csv_path(path)
    strategies = import_csvs(path, csvs, startdate)
    statistics, monthlyreturns = stats_monthlyrets(strategies)
    corr_mat, ret_mat = correlation_matrix(monthlyreturns)
    equal_w, optimal_w, sharpe_ratio, iterations, weights = \
        portfolio_optimization(ret_mat, minWeight, maxWeight)
    
    
    #Prints
    plt.figure(1, figsize=(11.5,5))
    plt.subplot(121)
    plt.plot(np.cumsum(np.dot(optimal_w, ret_mat.T)))
    plt.title('Optimal weights. Sharpe: %0.3f' %sharpe(optimal_w))
    plt.legend(loc=0)
    plt.subplot(122)
    plt.plot(np.cumsum(np.dot(equal_w, ret_mat.T)))
    plt.legend(loc=0)
    plt.title('Equal weights. Sharpe: %0.3f' %sharpe(equal_w))
    plt.show()
    
    print('\n Optimal weights:')
    print(weights)
    print('\n Correlation Matrix:')
    print(np.round(corr_mat, 2))
    
    plt.figure(2, figsize=(3,3))
    plt.matshow(corr_mat, cmap='RdBu', interpolation='none', vmin=-1,
                vmax=1)
    plt.grid(False)
    plt.title('Correlation heatmap')
    plt.show()

    
    

