#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/19/18 10:48 AM
# @Author  : gehen
# @File    : graph_matching.py
# @Software: PyCharm Community Edition

from scipy.spatial.distance import cosine,euclidean
import numpy as np

def cal_rank(matrix):
    col_num,row_num = matrix.shape
    col_sum = row_sum = 0
    # for i in range(col_num):
    #     for j in range(i+1,col_num):
    #         try:
    #             col_sum += cosine(matrix[:,i],matrix[:,j])
    #         except:
    #             pass
    for i in range(row_num):
        for j in range(i+1,row_num):
            try:
                row_sum += cosine(matrix[i,:],matrix[j,:])
            except:
                pass
    return np.mean(row_sum)

a = np.asarray([[1,0],[0,1]])
b = np.asarray([[1,0,1],[0,1,0],[1,0,1]])
c = np.asarray([[1,0,1,0],[0,1,0,0],[1,0,1,0],[0,1,0,0]])
# ua,sa,va = np.linalg.svd(a)
# ub,sb,vb = np.linalg.svd(b)
# print np.linalg.matrix_rank(c)
print np.linalg.norm(c,'nuc')