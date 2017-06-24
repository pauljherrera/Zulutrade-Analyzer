# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 14:50:08 2016

@author: forex
"""

from __future__ import division
from __future__ import print_function

from selenium import webdriver
from selenium.webdriver.common.keys import Keys


class Scraper():
    """
    """
    def __init__(self, website):
        self.driver = webdriver.Chrome()
        self.driver.get(website)
    
    def fill_input(self, xpath, text):
        inputToFill = self.driver.find_element_by_xpath(xpath)
        inputToFill.clear()
        inputToFill.send_keys(text)
        
        











"""
elem = driver.find_element_by_name("q")
elem.clear()
elem.send_keys("pycon")
elem.send_keys(Keys.RETURN)
assert "No results found." not in driver.page_source
driver.close()
"""

