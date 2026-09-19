#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 08:34:57 2026

@author: singh
"""
import QuantLib as ql

cur = ql.EURCurrency()
money1 = ql.Money(cur, 100)
money2 = ql.Money(100, cur)
print("money1: ",  money1.value())
print("money2: ",  money2.value())

money3 = 100 * cur
print("money3: ",  money3.value())