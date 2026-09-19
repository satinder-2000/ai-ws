#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 18:20:57 2026

@author: singh
"""
import QuantLib as ql

print("Calendars determine which days are business days and which are holidays. This matters for settlement dates, coupon payments, and exercise dates.")

uk_calendar = ql.UnitedKingdom()
us_calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)

date = ql.Date(25, ql.December, 2026)

print(uk_calendar.isBusinessDay(date))
print(uk_calendar.isHoliday(date))

# Advance by 5 business days from a given date
start = ql.Date(20, ql.December, 2026)
settlement = uk_calendar.advance(start, ql.Period(5, ql.Days))
print(settlement)  # Skips Christmas and Boxing Day

# Count business days between two dates
bdays = uk_calendar.businessDaysBetween(
    ql.Date(1, ql.January, 2026),
    ql.Date(31, ql.December, 2026)
)
print(f"Business days in 2026: {bdays}")