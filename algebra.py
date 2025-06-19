#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第1回演習問題
"""
import numpy as np

x = 100
print("x=", x)

########## 課題1(a)
print("Hello, World")

########## 課題1(b)i
A = np.array([[1,2,3,4,5],
              [6,7,8,9,10],
              [11,12,13,14,15],
              [16,17,18,19,20]],
             dtype = float)

b = np.array([[1],
              [0],
              [1],
              [0],
              [1]],
             dtype = float)

########## 課題1(b)ii
print("A = \n", A)
print("b = \n", b)

mul = A @ b
print("Ab = \n", mul)

########## 課題1(b)iii
row_sums = np.sum(A, axis=1)
col_sums = np.sum(A,axis=0)

print("Row sums：\n", row_sums)
print("Column sums:\n", col_sums)

########## 課題1(c) 例
a1 = [0] * 11

for i in range(1, 11):
    a1[i] = 2 * a1[i - 1] + 1
    print(f"a1[{i}] = {a1[i]}")

    
########## 課題1(c)
a2 = [0] * 11
a2[0] = 6

for i in range(1, 11):
    if a2[i - 1] % 2 == 0:
        a2[i] = a2[i - 1] // 2
    else:
        a2[i] = 3 * a2[i - 1] + 1
    print(f"a2[{i}] = {a2[i]}")
