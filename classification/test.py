#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 11:27:35 2019

@author: thanmaireddy
"""

test = 'chandini'
n = 3
if len(test) % n != 0:
    print("invalid")
    
part = len(test)/3
k = 0
for i in test:
    if k % part == 0:
        pass    #print ('\n')
    print (i)
    k += 1
    