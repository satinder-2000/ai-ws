#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 17:47:50 2026

@author: singh
"""
import QuantLib as ql

date1 = ql.Date(15, ql.June, 2026)
print(date1)

date2 = date1 + ql.Period(6, ql.Months)
print(date2)

date3 = date1 + ql.Period(30, ql.Days)
print(date3)

print(date1.weekday()) # 2 (Monday=1 ... Friday=5)
print(date1.month())
print(date1.year())

diff = date2 - date1
print(f"Days between: {diff}") 

today = ql.Date.todaysDate()
print(today)