#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 02:49:44 2026

@author: singh
"""
import QuantLib as ql

today = ql.Date(9, ql.April, 2026)
ql.Settings.instance().evaluationDate = today

# Option parameters
strike = 100.0
expiry = ql.Date(9, ql.April, 2027)
option_type = ql.Option.Put

#Marget data
spot = 95.0
vol = 0.25
rate = 0.045
div_yield = 0.02

# Build the BSM process
spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
rate_handle = ql.YieldTermStructureHandle(
    ql.FlatForward(today, rate, ql.Actual365Fixed())
)
div_handle = ql.YieldTermStructureHandle(
    ql.FlatForward(today, div_yield, ql.Actual365Fixed())
)
vol_handle = ql.BlackVolTermStructureHandle(
    ql.BlackConstantVol(today, ql.NullCalendar(),vol, ql.Actual365Fixed())
)

process = ql.BlackScholesMertonProcess(
    spot_handle, div_handle, rate_handle, vol_handle
)

# Define the American option
payoff = ql.PlainVanillaPayoff(option_type, strike)
exercise = ql.AmericanExercise(today, expiry)
american_option = ql.VanillaOption(payoff, exercise)

# Method 1: Binomial tree (Cox-Ross-Rubinstein)
steps = 500
american_option.setPricingEngine(
    ql.BinomialVanillaEngine(process,"crr", steps)
)
binomial_price= american_option.NPV()
print(f"Binomial CRR ({steps} steps): {binomial_price:.4f}")

# Method 2: Finite difference
american_option.setPricingEngine(
    ql.FdBlackScholesVanillaEngine(process, 200, 200)    
)
fd_price = american_option.NPV()
print(f"Finite difference           {fd_price:.4f}")

# Compare with the European price (lower bound for the American)
euro_payoff = ql.PlainVanillaPayoff(option_type, strike)
euro_exercise = ql.EuropeanExercise(expiry)
euro_option = ql.VanillaOption(euro_payoff, euro_exercise) 
euro_option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
euro_price = euro_option.NPV()
print(f"European price (lower bound): {euro_price:.4f}")
print(f"Early exercise premium:       {fd_price - euro_price:.4f}")