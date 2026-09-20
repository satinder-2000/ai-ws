#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 06:35:38 2026

@author: singh
"""
import QuantLib as ql

today = ql.Date(9, ql.April, 2026)
ql.Settings.instance().evaluationDate=today
calendar = ql.UnitedKingdom()
day_count = ql.Actual365Fixed()

# Deposit rates (short end: overnight to 6 months)
deposit_helpers = []
deposit_data = [
    (ql.Period(1, ql.Days),     0.0430), #overnight
    (ql.Period(1, ql.Weeks),    0.0432),
    (ql.Period(1, ql.Months),   0.0435),
    (ql.Period(3, ql.Months),   0.0440),
    (ql.Period(6, ql.Months),   0.0445),
]

for tenor, rate in deposit_data:
    helper = ql.DepositRateHelper(
        ql.QuoteHandle(ql.SimpleQuote(rate)),
        tenor,
        2, #settlement days
        calendar,
        ql.ModifiedFollowing,
        False,  #end of month
        day_count
    )
    deposit_helpers.append(helper)
    
    
# Swap rates (long end: 2 years to 30 years
swap_helpers = []
swap_data = [
    (ql.Period(2, ql.Years),  0.0420),
    (ql.Period(3, ql.Years),  0.0415),
    (ql.Period(5, ql.Years),  0.0405),
    (ql.Period(7, ql.Years),  0.0400),
    (ql.Period(10, ql.Years), 0.0395),
    (ql.Period(15, ql.Years), 0.0390),
    (ql.Period(20, ql.Years), 0.0388),
    (ql.Period(30, ql.Years), 0.0385),
]
for tenor, rate in swap_data:
    helper = ql.SwapRateHelper(
        ql.QuoteHandle(ql.SimpleQuote(rate)), 
        tenor,
        calendar,
        ql.Annual,
        ql.ModifiedFollowing,
        day_count,
        ql.Euribor6M() #floating leg rate
    )
    swap_helpers.append(helper)
    
# Combine all helpers and bootstrap the curve
helpers = deposit_helpers + swap_helpers
curve = ql.PiecewiseLogLinearDiscount(today, helpers, day_count)
curve.enableExtrapolation()

# Extract zero rates and discount factors at various tenors
print("Tenor    Zero Rate   Discount Factor")
print("-" * 42)
tenors = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
for t in tenors:
    zero = curve.zeroRate(t, ql.Compounded, ql.Annual).rate()
    df = curve.discount(t)
    print(f"{t:5.2f}y    {zero:.4%}    {df:.6f}")