#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/6 11:21
# @Author  : Gehen
# @Site    : 
# @File    : nmf_k.py
# @Software: PyCharm Community Edition

import numpy as np
from time import time
import scipy.io as sio
np.set_printoptions(precision=4,suppress=True,linewidth=1000)

def init_S(D,view_list,sigma=40):
    S = np.exp(-np.power(D,2)/(2*pow(sigma,2)))
    S = (S+S.T)/2
    for idx,view in enumerate(view_list):
        start = sum(view_list[0:idx])
        end = start+view
        S[start:end,start:end] = np.eye(view)
    return S

def factorize(S,view_list,K,k,item,Max_Iter=50000,check_point=100,thresh=5e-5,beta=1):
    sort_list = sorted(range(len(view_list)), key=lambda k: -view_list[k])
    S_indexs = []
    S = np.mat(S)

    # init A
    A = np.random.rand(K, k)
    start = sum(view_list[0:sort_list[item]])
    end = start + view_list[sort_list[item]]
    S_index = range(start, end)
    S_indexs.append(S_index)
    A_index = range(view_list[sort_list[item]])
    A[S_index, :] = 0
    A[:, A_index] = S[:, S_index]
    A = np.mat(A)

    # init W
    W = np.mat(np.eye(k, dtype=np.float16))

    # optimize S = A*W*A'
    check1 = np.linalg.norm(A * W * A.T, 2)
    check2 = np.linalg.norm(S - A * W * A.T, 2)
    exit_flag = 0
    for iter in range(1, Max_Iter + 1):
        w_top = A.T * S * A + 1e-3
        w_bot = A.T * A * W * A.T * A + 1e-3
        W = np.multiply(W, w_top / w_bot)

        a_top = S * A * W + 1e-3
        a_bot = A * W * A.T * A * W + 1e-3
        A = np.multiply(A, 1 - beta + beta * (a_top / a_bot))

        if iter % check_point == 0:
            now_check1 = np.linalg.norm(A * W * A.T, 2)
            now_check2 = np.linalg.norm(S - A * W * A.T, 2)
            if abs(now_check1 - check1) <= thresh and abs(now_check2 - check2) <= thresh:
                exit_flag = 1
                break
            else:
                check1 = now_check1
                check2 = now_check2
    # print('Iteration is : {}, exit flag is : {}'.format(iter, exit_flag))
    return A

def block_constrain(H,i,j,sum_view):
    H[i, :] = 0
    temp_sum = sum_view + [i]
    temp_sum.sort()
    target_index = len(temp_sum) - 1 - temp_sum[::-1].index(i)
    # assert (target_index < 1, 'target index is negative')
    block_range = range(temp_sum[target_index - 1], temp_sum[target_index + 1])
    H[block_range, j] = 0
    H[i, j] = 1
    return H

def assign_H(D,view_list,thresh=0.95):
    if sum(view_list)>0:
        S = init_S(D,view_list)
        # max_view = max(view_list)
        [eig_value,_] = np.linalg.eig(S)
        k = np.sum(eig_value>=thresh)
        k = max(k,max(view_list))
        H = assign_kH(S,view_list,k)
        return H,k
    else:
        return np.zeros(0),0

    # for k in range(max_view,2*max_view):
    #     start = time()
    #     H = assign_kH(S,view_list,k)
    #     print ('cost {}s'.format(time()-start))
    #     trace = np.trace(H.T*S*H)
    #     pass


def assign_kH(S,view_list,k,sparsity_thresh=0.6,rank=20):
    sum_view =  [sum(view_list[0:i]) for i in range(len(view_list)+1)]
    K = sum(view_list)
    num = view_list.count(max(view_list))
    assert K>=k,'Detection number K({}) is smaller than real ID number k({})'.format(K,k)
    assert k>=max(view_list),'k({}) is smaller than max view num {}'.format(k,max(view_list))

    H_list = []
    for item in range(rank):
        data_item = int(item%num)
        A = factorize(S, view_list, K, k,data_item)

        # assign H column-wise
        H = -np.ones([K,k])
        temp_A = A.copy()
        for j in range(k):
            temp_index = np.where(H[:,j]==-1)
            i = np.argmax(temp_A[temp_index,j])
            i = temp_index[0][i]
            H = block_constrain(H,i, j,sum_view)

        # assign H row-wise
        temp_A = (A / np.sum(A, 1)).copy()
        pow_list = [pow(np.max(temp_A[i, :]), 2) / pow(np.linalg.norm(temp_A[i, :]), 1) for i in range(K)]
        for i in range(K):
            if pow_list[i] >= sparsity_thresh:
                j = np.argmax(temp_A[i, :])
                if H[i, j] != -1:
                    continue
                H = block_constrain(H,i, j,sum_view)

        # re-assign the non-sparsity row
        temp_H = H.copy()
        temp_id = np.asarray([np.where(temp_H[:, j] == 1)[0] for j in range(k)])
        for i in range(K):
            re_assign_index = np.where(H[i, :] == -1)[0]
            if re_assign_index.shape[0] < 1:
                continue
            temp_indexs = temp_id[re_assign_index]
            temp_data = [np.min(S[i, temp_index]) for temp_index in temp_indexs]
            j = re_assign_index[np.argmax(temp_data)]
            H = block_constrain(H, i, j, sum_view)
        H_list.append(np.mat(H))
    trace_list = [np.trace(H_item.T*S*H_item) for H_item in H_list]
    index = np.argmax(trace_list)
    return H_list[index]

if __name__ == '__main__':
    S14 =  np.asarray([[1, 0 ,0 ,0.7, 0.4, 0.3, 0.8, 0.4 ,0.5 ,0.3, 0.6 ,0.5, 0.4 ,0.2],
       [0 ,1 ,0, 0.3, 0.4, 0.6, 0.2, 0.5, 0.3, 0.6, 0.1, 0.2, 0.4, 0.],
       [0, 0 ,1 ,0.4, 0.6, 0.3, 0.2 ,0.4, 0.8 ,0.3, 0.4, 0.1, 0.3, 0.3],
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
    # S = init_S(S, view_list)
    # H = assign_kH(S,view_list,k)
    # H,k = assign_H(S,view_list)
    print ('cost {}s'.format(time()-start))
    print ('k = {}'.format(k))
    # print (np.trace(H.T*S*H))
    # for j in range(k):
    #     print (np.where(H[:,j]==1)[0]+1)
