#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 09:04:21 2026

@author: singh
"""
import QuantLib as ql

print("\n-----Forward Rate Agreement-----")
print("classql.ForwardRateAgreement(valueDate, maturityDate, position, strikeForward, notional, iborIndex, discountCurve=ql.YieldTermStructureHandle())\n")

startDate = ql.Date(30,6,2020)
ql.Settings.instance().evaluationDate = startDate

spotDates = [ql.Date(30,6,2020), ql.Date(31,12,2020),ql.Date(30,6,2021)]
spotRates = [0.05, 0.05, 0.05]

dayConvention = ql.Actual360()
calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
maturityDate = calendar.advance(startDate, ql.Period('3M'))

compounding = ql.Simple
compoundingFrequency = ql.Annual

spotCurve = ql.ZeroCurve(spotDates, spotRates,
                      dayConvention,calendar,
                      ql.Linear(), compounding,
                      compoundingFrequency)
spotCurve.enableExtrapolation()
spotCurveHandle = ql.YieldTermStructureHandle(spotCurve)

index = ql.USDLibor(ql.Period('3M'), spotCurveHandle)
index.addFixing(ql.Date(26, 6, 2020), 0.05)
notional =100000
rate = 0.06

fra = ql.ForwardRateAgreement(
    calendar.advance(startDate,ql.Period(3,ql.Months)),
    calendar.advance(startDate,ql.Period(6,ql.Months)),
    ql.Position.Long, 
    rate, 
    notional, 
    ql.Euribor6M(spotCurveHandle), spotCurveHandle)

print('NPV:', fra.NPV())