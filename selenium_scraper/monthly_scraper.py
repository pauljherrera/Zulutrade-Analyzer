# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 14:11:45 2016

@author: forex
"""

from __future__ import division
from __future__ import print_function

import pandas as pd
from time import sleep

from selenium_handler import Scraper


# Functions
def get_last_list_number():
    i = 3   #The first two rows are not visible, so we start on 3.
    flag = False
    while flag==False:
        try:
            zulu.driver.find_element_by_xpath('//*[@id="performance-table"]/tbody[%s]/tr[1]/td[1]' %i)
            i += 1
        except:
            flag = True
            
    numTraders = i - 3
    
    return numTraders


if __name__ == "__main__":
    # User variables.
    user = 'miromanuel@gmail.com'
    password = 'miro1986'
    
    # Initializing
    zulu = Scraper('http://www.zulutrade.com')
    
    # Logging in
    zulu.driver.find_element_by_xpath('//*[@id="login-trigger"]').click()
    zulu.fill_input('//*[@id="login-username"]', user)
    zulu.fill_input('//*[@id="login-password"]', password)
    zulu.driver.find_element_by_xpath('//*[@id="btn-login"]').click()
    sleep(10)
    
    # Change language
    zulu.driver.find_element_by_xpath('//*[@id="selected-language"]').click()
    zulu.driver.find_element_by_xpath('//*[@id="available-languages"]/ol/li[1]/a').click()
    
    # Accesing to traders page
    zulu.driver.find_element_by_xpath('//*[@id="Form"]/div[2]/header/nav[2]/ul/li[1]/a').click()
    
    #Get list of banned and approved traders
    bannedTraders = pd.read_csv('banned_traders.csv', index_col=0)
    bannedTradersList = list(bannedTraders.name)
    
    approvedTraders = pd.read_csv('approved_traders.csv', index_col=0)
    approvedTradersList = list(approvedTraders.name)
  
    # Download approved traders csv  
    for i in xrange(len(approvedTradersList)):
        zulu.fill_input('//*[@id="performance-search-forex-provider-name"]', 
                    approvedTradersList[i])
        sleep(10)
        zulu.driver.find_element_by_xpath('//*[@id="main_ctl00_ProviderQuickSearch-popup"]/ul/li[3]').click()
        zulu.driver.find_element_by_xpath('//*[@id="trade-history-export"]').click()
        zulu.driver.find_element_by_xpath('/html/body/ul/li[3]').click()
        zulu.driver.back() 
        
    # Filtering traders
    zulu.driver.find_element_by_xpath('//*[@id="performance-search-advanced"]/span[1]').click()
    zulu.driver.find_element_by_xpath('//*[@id="Form"]/div[2]/section/div[1]/div[4]/div[2]/button').click()
    zulu.driver.find_element_by_xpath('/html/body/div[12]/ul/li[1]/label').click()
    sleep(10)
    zulu.driver.find_element_by_xpath('//*[@id="performance-advanced"]/ul/li[1]/label').click()
    zulu.fill_input('//*[@id="search-tradesfrom"]', '300')
    zulu.driver.find_element_by_xpath('//*[@id="search-advbtn"]').click()
    sleep(10)        
 
    # Getting the number of potential traders
    flag = False
    prevNumTraders = 0
    
    while flag==False:
        zulu.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        sleep(6)
        numTraders = get_last_list_number()
        if numTraders == prevNumTraders:
            flag = True
        else: prevNumTraders = numTraders
        
    # Open each trader and download the csv file.
    for i in xrange(numTraders):
        name =\
            zulu.driver.find_element_by_xpath('//*[@id="performance-table"]/tbody[%s]/tr[1]/td[3]/div[1]/a' %(i+3)).text
        # Checking if the selected trader isn't banned.
        if not name.lower() in bannedTradersList:
            numWeeks =\
                int(zulu.driver.find_element_by_xpath('//*[@id="performance-table"]/tbody[%s]/tr[1]/td[14]' %(i+3)).text)             
            if not name.lower() in approvedTradersList:
                # Checking if the selected trader has enough weeks.
                if numWeeks >= 47:
                    zulu.driver.find_element_by_xpath('//*[@id="performance-table"]/tbody[%s]/tr[1]/td[3]/div[1]/a' %(i+2)).location_once_scrolled_into_view 
                    zulu.driver.find_element_by_xpath('//*[@id="performance-table"]/tbody[%s]/tr[1]/td[3]/div[1]/a' %(i+3)).click()
                    sleep(2)
                    zulu.driver.switch_to_window(zulu.driver.window_handles[1])
                    zulu.driver.find_element_by_xpath('//*[@id="trade-history-export"]').click()
                    zulu.driver.find_element_by_xpath('/html/body/ul/li[3]').click()
                    zulu.driver.execute_script("window.close('');")
                    zulu.driver.switch_to_window(zulu.driver.window_handles[0])
    




