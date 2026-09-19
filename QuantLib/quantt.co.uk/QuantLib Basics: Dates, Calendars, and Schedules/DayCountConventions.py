#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 18:31:42 2026

@author: singh
"""
import QuantLib as ql

dc_act365 = ql.Actual365Fixed()
dc_act360 = ql.ActualActual(ql.ActualActual.ISDA)
dc_30360 = ql.Thirty360(ql.Thirty360.BondBasis)

d1 = ql.Date(1, ql.March, 2026)
d2 = ql.Date(1, ql.September, 2026)

print(f"Actual/365: {dc_act365.yearFraction(d1, d2): 6f}")
print(f"Act/Act ISDA: {dc_act360.yearFraction(d1, d2): 6f}")
print(f"30/360: {dc_30360.yearFraction(d1, d2): 6f}")