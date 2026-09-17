#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 14:35:11 2026

@author: singh
"""
import QuantLib as ql

#Empty Array
emptyArr = ql.Array()
print("Empty Array: ",emptyArr)

#Array with size and Value
sizeValueArr = ql.Array([1, 2, 3, 4])
print("sizeValueArr: ",sizeValueArr)

sizeValueArrIncrmt = ql.Array(([1, 2, 3, 4, 5, 6, 7, 8]), 2)
print("sizeValueArrIncrmt: ",sizeValueArrIncrmt)