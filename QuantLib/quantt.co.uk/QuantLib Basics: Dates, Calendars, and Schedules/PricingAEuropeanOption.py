#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 20:22:29 2026

@author: singh
"""
print("We'll price a European call option using the analytic Black-Scholes engine")
import QuantLib as ql

# Set the evaluation date
today = ql.Date(9, ql.April, 2026)
ql.Settings.instance().evaluationDate = today

# Option parameters
option_type = ql.Option.Call
strike = 100.0
expiry = ql.Date(9, ql.October, 2026)

# Market data
spot_price = 105.0
volatility = 0.20 # 20% annualised vol
risk_free_rate = 0.045 # 4.5% risk-free rate
dividend_yield = 0.01  # 1% continuous dividend yield

# Build the market data handles
spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
vol_handle = ql.BlackVolTermStructureHandle(
    ql.BlackConstantVol(today, ql.NullCalendar(), volatility, ql.Actual365Fixed())
)
rate_handle = ql.YieldTermStructureHandle(
    ql.FlatForward(today, risk_free_rate, ql.Actual365Fixed())
)
dividend_handle = ql.YieldTermStructureHandle(
    ql.FlatForward(today, dividend_yield, ql.Actual365Fixed())
)


# Construct the Black-Scholes-Merton process
bsm_process = ql.BlackScholesMertonProcess(
    spot_handle, dividend_handle, rate_handle, vol_handle
)


# Define the option
payoff = ql.PlainVanillaPayoff(option_type, strike)
exercise = ql.EuropeanExercise(expiry)
option = ql.VanillaOption(payoff, exercise)

# Attach the pricing engine
option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))

# Results
print(f"Option price:   {option.NPV():.4f}")
print(f"Delta:          {option.delta():.4f}")
print(f"Gamma:          {option.gamma():.4f}")
print(f"Vega:           {option.vega():.4f}")
print(f"Theta:          {option.theta():.4f}")
print(f"Rho:            {option.rho():.4f}")
