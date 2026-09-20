#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 11:38:03 2026

@author: singh
"""
import QuantLib as ql

today = ql.Date(9, ql.April, 2026)
ql.Settings.instance().evaluationDate = today

#Setup
spot_quote = ql.SimpleQuote(100.0)
vol_quote = ql.SimpleQuote(0.20)
rate = 0.045
div_yield = 0.01

spot_handle = ql.QuoteHandle(spot_quote)
vol_handle = ql.BlackVolTermStructureHandle(
    ql.BlackConstantVol(today, ql.NullCalendar(), ql.QuoteHandle(vol_quote),ql.Actual365Fixed())
)
rate_handle = ql.YieldTermStructureHandle(
    ql.FlatForward(today, rate, ql.Actual365Fixed())
)
div_handle = ql.YieldTermStructureHandle(
    ql.FlatForward(today, div_yield, ql.Actual365Fixed())
)

process = ql.BlackScholesMertonProcess(
    spot_handle, div_handle,rate_handle, vol_handle
)

# Create and price the option
payoff = ql.PlainVanillaPayoff(ql.Option.Call, 100.0)
exercise = ql.EuropeanExercise(ql.Date(8, ql.October, 2026))
option = ql.VanillaOption(payoff, exercise)
option.setPricingEngine(ql.AnalyticEuropeanEngine(process))

# Analytic Greeks
print("=== Analytic Greeks ===")
print(f"Price:   {option.NPV():.4f}")
print(f"Delta:   {option.delta():.4f}")
print(f"Gamma:   {option.gamma():.6f}")
print(f"Vega:    {option.vega():.4f}")
print(f"Theta:   {option.theta():.4f}")
print(f"Rho:     {option.rho():.4f}")

# Bump-and-reprice for custom sensitivities
# Useful when analytic Greeks aren't available (e.g. exotic options)
bump = 0.01 # 1% relative bump

# Spot delta via finite difference
base_price = option.NPV()
spot_quote.setValue(100.0 * (1 + bump))
up_price = option.NPV()
spot_quote.setValue(100.0 * (1 - bump))
down_price = option.NPV()
spot_quote.setValue(100.0) #reset

fd_delta = (up_price - down_price) /(2 * 100.0 * bump)
print(f"\nFD Delta: {fd_delta:.4f}")

# Vega via vol bump
vol_quote.setValue(0.21)
up_vol_price = option.NPV()
vol_quote.setValue(0.19)
down_vol_price = option.NPV()
vol_quote.setValue(0.20) #reset

fd_vega = (up_vol_price - down_vol_price) / 0.02
print(f"FD Vega: {fd_vega:.4f}")