#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 20:51:25 2026

@author: singh
"""
import QuantLib as ql

print("\n-----SimpleCashFlow----\n")
amount = 105
date = ql.Date(15, 6, 2020)
cf = ql.SimpleCashFlow(amount, date)
print("cf.amount():",cf.amount())
print("cf.date():",cf.date())

print("\n-----Redemption----\n")
amount = 100
date = ql.Date(15,6,2020)
redemption = ql.Redemption(amount, date)
print("redemption.amount():",redemption.amount())
print("redemption.date():",redemption.date())