#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 18:21:29 2026

@author: singh
"""
import QuantLib as ql

today = ql.Date(30, 6, 2020)
ql.Settings.instance().evaluationDate = today

spotDates = [ql.Date(30, 6, 2020), ql.Date(31, 12, 2020), ql.Date(30, 6, 2021)]
spotRates = [0.05, 0.05, 0.05]

dayConvention = ql.Thirty360(ql.Thirty360.BondBasis)
calendar = ql.UnitedKingdom()

startDate = calendar.advance(today, ql.Period('3M'))
maturityDate = calendar.advance(startDate, ql.Period('3M'))

compounding = ql.Simple
compoundingFrequency = ql.Annual

spotCurve = ql.ZeroCurve(spotDates,spotRates,dayConvention,
                         calendar, ql.Linear(),
                         compounding, compoundingFrequency)
spotCurve.enableExtrapolation()
spotCurveHandle=ql.YieldTermStructureHandle(spotCurve)

index = ql.EURLibor(ql.Period('3M'), spotCurveHandle)
index.addFixing(ql.Date(26, 6, 2020), 0.05)
notional = 100000
rate = 0.06

fra = ql.ForwardRateAgreement(startDate, maturityDate, ql.Position.Long, 
                              0.01, notional, ql.EURLibor6M(spotCurveHandle),
                              spotCurveHandle)