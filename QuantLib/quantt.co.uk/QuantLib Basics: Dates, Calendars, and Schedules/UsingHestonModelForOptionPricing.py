#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 22:51:12 2026

@author: singh
"""
import QuantLib as ql

today = ql.Date(9, ql.April,2026)
ql.Settings.instance().evaluationDate = today

#Market Data
spot = 100.0
rate = 0.045
div_yield = 0.01

spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
rate_handle = ql.YieldTermStructureHandle(
    ql.FlatForward(today,rate,ql.Actual365Fixed())
)
div_handle =  ql.YieldTermStructureHandle(
    ql.FlatForward(today,div_yield,ql.Actual365Fixed())
)

# Heston model parameters
v0 = 0.04       # initial variance (vol^2)
kappa = 2.0     # mean reversion speed
theta = 0.04    # long-run variance
sigma = 0.3     # vol of vol
rho = -0.7      # correlation between spot and vol

heston_process = ql.HestonProcess(
    rate_handle, div_handle, spot_handle, v0, kappa, theta, sigma, rho    
)
heston_model = ql.HestonModel(heston_process)
heston_engine = ql.AnalyticHestonEngine(heston_model)

# Price options at different strikes to see the volatility smile
strikes = [80, 85, 90, 95, 100, 105, 110, 115, 120]
expiry = ql.Date(9, ql.April, 2027)

print(f"{'Strike':>8} {'Price':>10} {'Implied Vol':.12}")
print("-" * 32)

for K in strikes:
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
    exercise = ql.EuropeanExercise(expiry)
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(heston_engine)
    
    price = option.NPV()
    # Back out the BS implied vol from the Heston price
    iv = option.impliedVolatility(
        price,
        ql.BlackScholesMertonProcess(
            spot_handle, div_handle, rate_handle,
            ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(today, ql.NullCalendar(), 0.2, ql.Actual365Fixed())
            )
        )
    )
    print(f"{K:>8.0f} {price:>10.4f} {iv:>11.2%}")