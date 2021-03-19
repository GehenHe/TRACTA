#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 08/21/18 9:12 AM
# @Author  : gehen
# @Site    : 
# @File    : rnmf.py
# @Software: PyCharm

import numpy as np
from time import time
import scipy.io as sio
np.set_printoptions(precision=4,suppress=True,linewidth=1000)

def factorize(S,view_list,K,item,Max_Iter=500000,check_point=100,thresh=5e-5,alpha=1):
    sort_list = sorted(range(len(view_list)), key=lambda k: -view_list[k])
    S_indexs = []
    S = np.mat(S)
    N = S.shape[0]

    # init A
    A = np.random.rand(N, K)
    # start = sum(view_list[0:sort_list[item]])
    # end = start + min(K,view_list[sort_list[item]])
    # S_index = range(start, end)
    # S_indexs.append(S_index)
    # A_index = range(min(K,view_list[sort_list[item]]))
    # A[S_index, :] = 0
    # A[:, A_index] = S[:, S_index]
    A = np.mat(A)

    # init I1, I2
    I1 = np.ones([K,1])
    I2 = np.ones([N,1])

    # optimize ||S-AA'||^2+alpha*||AI1-I2||^2
    check1 = np.linalg.norm(A * A.T, 2)
    check2 = np.linalg.norm(S - A*A.T, 2)
    exit_flag = 0
    for iter in range(1,Max_Iter):
        top = 4 * S* A + 2*alpha * I2 * I1.T + 1e-3
        bot = 4 * A * A.T* A + 2 * alpha * A * I1 * I1.T + 1e-3
        A = np.multiply(A, np.sqrt(top/bot))
        if iter % check_point == 0:
            if alpha>=3:
                alpha = alpha*(1.0/3)
            else:
                alpha = 1
            now_check1 = np.linalg.norm(A * A.T, 2)
            now_check2 = np.linalg.norm(S - A*A.T, 2)
            if abs(now_check1 - check1) <= thresh and abs(now_check2 - check2) <= thresh:
                exit_flag = 1
                break
            else:
                check1 = now_check1
                check2 = now_check2
    # print('Iteration is : {}, exit flag is : {}'.format(iter, exit_flag))
    return A

def assign_H(S,view_list,thresh=0.9):
    if sum(view_list)>0:
        [eig_value,_] = np.linalg.eig(S)
        K = np.sum(eig_value>=thresh)
        K = max(K,1)
        # print ('\t\t{}'.format(K))
        H = assign_kH(S,view_list,K)
        return H,K
    else:
        return np.zeros([0,0]),0


def assign_kH(S,view_list,K,sparsity_thresh=0.6,rank=10):
    N = S.shape[0]
    sum_view =  [sum(view_list[0:i]) for i in range(len(view_list)+1)]
    num = len(view_list)
    # assert K>=max(view_list),'k({}) is smaller than max view num {}'.format(K,max(view_list))

    H_list = []
    for item in range(rank):
        data_item = int(item%num)
        A = factorize(S, view_list, K,data_item)

        # assign H row-wise
        H = np.zeros([N,K])
        temp_A = A.copy()
        for i in range(N):
            j = np.argmax(temp_A[i,:])
            H[i,j] = 1
        H_list.append(np.mat(H))
    lost_list = [np.linalg.norm(S-H_item*H_item.T) for H_item in H_list]
    index = np.argmin(lost_list)
    optimal_H = H_list[index]
    # print index,lost_list
    # print('\n')
    # print(index)
    # for j in range(K):
    #     temp = np.where(optimal_H[:,j]==1)[0]
        # print (temp+1)
    return optimal_H

if __name__ == '__main__':
    S14 =  np.asarray([[1, 0 ,0 ,0.7, 0.4, 0.3, 0.8, 0.4 ,0.5 ,0.3, 0.6 ,0.5, 0.4 ,0.2],
       [0 ,1 ,0, 0.3, 0.4, 0.6, 0.2, 0.5, 0.3, 0.6, 0.1, 0.2, 0.4, 0.],
       [0, 0 ,1 ,0.4, 0.6, 0.3, 0.2 ,0.4, 0.1 ,0.3, 0.4, 0.1, 0.3, 0.3],
       [0.7, 0.2, 0.3, 1 ,0 ,0, 0.8, 0.3, 0.2, 0.4, 0.6, 0.1, 0.2, 0.5],
       [0.3, 0.2, 0.4, 0 ,1, 0, 0.4, 0.8, 0.3, 0.1, 0.2, 0.7, 0.3, 0.5],
       [0.3, 0.6, 0.1, 0, 0, 1, 0.3, 0.5, 0.4, 0.2, 0.3, 0.1, 0.4, 0.8],
       [0.6, 0.3, 0.2, 0.7, 0.3, 0.1, 1, 0, 0, 0, 0.6, 0.3, 0.2, 0.4],
       [0.2, 0.2, 0.4, 0.1, 0.7, 0.3, 0 ,1, 0, 0, 0.4, 0.8, 0.6, 0.4],
       [0.1, 0.4 ,0.3, 0.5, 0.6, 0.4, 0, 0, 1, 0, 0.4, 0.5, 0.8, 0.3],
       [0.3, 0.2 ,0.5, 0.4 ,0.3, 0.4, 0, 0, 0, 1, 0.4, 0.3, 0.1, 0.2],
       [0.6, 0.3, 0.1, 0.6, 0.4, 0.3, 0.8, 0.1, 0.5, 0.3 ,1, 0 ,0, 0],
       [0.3, 0.2, 0.1, 0.3, 0.7, 0.4, 0.6, 0.3, 0.2, 0.1, 0, 1, 0 ,0],
       [0.3, 0.1, 0.4, 0.3, 0.2, 0.3 ,0.6, 0.3, 0.8, 0.3, 0 ,0 ,1 ,0],
       [0.3, 0.7, 0.3, 0.5, 0.3 ,0.9, 0.2, 0.3, 0.1 ,0.6, 0 ,0, 0, 1]])
    S = (S14+S14.T)/2
    # sigma = 20
    # data = sio.loadmat('1741.mat')
    # D = data['array']
    # D  = D[0:18,0:18]
    # S = init_S(D,view_list)

    view_list = [3,3,4,4]
    start = time()
    # S = init_S(D, view_list)
    # H = assign_kH(S,view_list,k)
    # A = factorize(S,view_list,6,2)

    H = assign_H(S,view_list)
    k = H[0].shape[1]
    print (H[0])
    for j in range(k):
        print (np.where(H[0][:,j]==1)[0]+1)

    # H,k = assign_H(D,view_list)
    # print ('cost {}s'.format(time()-start))
    # print ('k = {}'.format(k))