#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 09:17:19 2026

@author: singh
"""
import QuantLib as ql

today = ql.Date(9, ql.April, 2026)
ql.Settings.instance().evaluationDate = today

# Build a flat yield curve for simplicity
flat_rate = 0.04
curve = ql.FlatForward(today, flat_rate, ql.Actual365Fixed())
curve_handle = ql.YieldTermStructureHandle(curve)

#Bond parameters
face_value = 100.0
issue_date = ql.Date(15, ql.March, 2024)
maturity_date = ql.Date(15, ql.March, 2034)
coupon_rate = 0.045 # 4.5% annual coupon
settlement_days = 2
calendar = ql.UnitedKingdom()
day_count = ql.Actual365Fixed()

#Generate the coupon schedule
schedule = ql.Schedule(
    issue_date,
    maturity_date,
    ql.Period(ql.Semiannual),
    calendar,
    ql.ModifiedFollowing,
    ql.ModifiedFollowing,
    ql.DateGeneration.Backward,
    False
)

#create the bond
bond = ql.FixedRateBond(
    settlement_days,
    face_value,
    schedule,
    [coupon_rate],
    day_count
)

# Attach a discounting engine
engine = ql.DiscountingBondEngine(curve_handle)
bond.setPricingEngine(engine)

#Results
print(f"Clean price:    {bond.cleanPrice():.4f}")
print(f"Dirty price:    {bond.dirtyPrice():.4f}")
print(f"Accrued:        {bond.accruedAmount():.4f}")
print(f"Yield (semi0:   {bond.bondYield(day_count, ql.Compounded, ql.Semiannual):4f}")

#Cash flows
print("\nCash Flows:")
print(f"{'Date':<22} {'Amount': >10}")
print("-" * 32)
for cf in bond.cashflows():
    print(f"{cf.date()}  {cf.amount():>10.4f}")