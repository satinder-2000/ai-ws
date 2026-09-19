#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 20:07:05 2026

@author: singh
"""
import QuantLib as ql

issue_date = ql.Date(15, ql.March, 2026)
maturity_date = ql.Date(15, ql.March, 2031)
tenor = ql.Period(ql.Semiannual)
calendar = ql.UnitedKingdom()
convention = ql.ModifiedFollowing
termination_convention = ql.ModifiedFollowing
rule = ql.DateGeneration.Backward

schedule = ql.Schedule(
    issue_date,
    maturity_date,
    tenor,
    calendar,
    convention,
    termination_convention,
    rule,
    False  # end of month
)

print("Coupon dates:")
for i, date in enumerate(schedule):
    print(f"  {i}: {date}")