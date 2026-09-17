#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 11:13:27 2026

@author: singh
"""

import QuantLib as ql

today = ql.Date(15, 6, 2020)

ql.Settings.instance().evaluationDate = today
print(ql.Settings.instance().evaluationDate)

settlementDays = 2

# Holiday calendar of united states
calendar =ql.UnitedStates(ql.UnitedStates.NYSE)

forwardRate = 0.05

"""Day Counter provides methods for determining the length of a time period according to given market convention,
both as a number of days and as a year fraction."""
dayCounter = ql.Actual360()

flatforwardTermStructure = ql.FlatForward(settlementDays, calendar, forwardRate, dayCounter)

flatforwardTermStructure.referenceDate()

print("Max Date: ", flatforwardTermStructure.maxDate())

#Changes evaluation date of calculation:

today = ql.Date(15, 6, 2020)
ql.Settings.instance().evaluationDate = today

effectiveDate = ql.Date(15, 6, 2020)
terminationDate = ql.Date(15, 6, 2022)

#create a schedule

schedule = ql.MakeSchedule(effectiveDate, terminationDate,
                           ql.Period(('6M')))

notional = [100.0]
rate = [0.05]
leg = ql.FixedRateLeg(schedule, dayCounter, notional, rate)

dayCounter = ql.Thirty360(ql.Thirty360.BondBasis)
rate = 0.03
compoundingType = ql.Compounded

frequency = ql.Annual
interestRate = ql.InterestRate(rate, dayCounter, compoundingType,frequency)

print("interestRate: ",interestRate)

discount_curve = ql.YieldTermStructureHandle(
    ql.FlatForward(0, ql.TARGET(), 0.03, ql.Actual360())
)

ql.Settings.instance().evaluationDate = ql.Date(15, 12, 2020)
print(ql.CashFlows.npv(leg, discount_curve, False))
