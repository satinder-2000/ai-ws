#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 17:22:39 2026

@author: singh
"""
import QuantLib as ql

print("a null matrix: ", ql.Matrix())


print("a 2 x 2 matrix: ", ql.Matrix(2,2))

A = ql.Matrix(3,3)
A[0][0] =0.2
A[0][1] =8.4
A[0][2] =1.5
A[1][0] =0.6
A[1][1] =1.4
A[1][2] =7.3
A[2][0] =0.8
A[2][1] =4.4
A[2][2] =3.2

print("ql.Matrix(3,3):\n",A)