#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 17:22:17 2026

@author: singh
"""
import QuantLib as ql

print("\n---SimpleQuote---\n")

s = ql.SimpleQuote(0.01)
print("s.value():",s.value())
print("s.setValue(0.05)")
s.setValue(0.05)
print("s.isValid()", s.isValid())

print("\n---DerivedQuote---\n")
d1 = ql.SimpleQuote(0.06)
d2 = ql.DerivedQuote(ql.QuoteHandle(d1),lambda x: 10*x)
print("d2:",d2.value())


print("\n---CompositeQuote---\n")

c1 =ql.SimpleQuote(0.02)
c2 =ql.SimpleQuote(0.03)

def f(x,y):
    return x+y

c3 = ql.CompositeQuote(ql.QuoteHandle(c1), ql.QuoteHandle(c2), f)
print(c3.value())

c4 = ql.CompositeQuote(ql.QuoteHandle(c1), ql.QuoteHandle(c2), lambda x,y:x+y)
print(c4.value())

print("\n---DeltaVolQuote---\n")
print("A class for FX-style quotes where delta-maturity pairs are quoted in implied vol")
deltaType = ql.DeltaVolQuote.Fwd 
atmType = ql.DeltaVolQuote.AtmFwd

maturity = 1.0
volAtm, vol25DeltaCall, vol25DeltaPut = 0.08, 0.075, 0.095

atmDeltaQuote = ql.DeltaVolQuote(ql.QuoteHandle(ql.SimpleQuote(volAtm)), deltaType, maturity, atmType)
print("atmDeltaQuote:",atmDeltaQuote.value())
vol25DeltaPutQuote = ql.DeltaVolQuote(-0.25, ql.QuoteHandle(ql.SimpleQuote(vol25DeltaPut)), maturity, deltaType)
print("vol25DeltaPutQuote:",vol25DeltaPutQuote.value())
vol25DeltaCallQuote = ql.DeltaVolQuote(0.25, ql.QuoteHandle(ql.SimpleQuote(vol25DeltaCall)), maturity, deltaType)
print("vol25DeltaCallQuote:",vol25DeltaCallQuote.value())