#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/1/18 11:09 AM
# @Author  : gehen
# @File    : homography_matrix.py
# @Software: PyCharm Community Edition

from self_config.self_config import self_cfg
import numpy as np
def terrace_H():
    H_1 = np.mat([[0.1060, 0.2962, 734.4005],  # terrace1-c0
                 [-0.0772, 0.0388, 667.4187],
                 [0.0002, 0.0000, 1]])
    H0 = np.mat([[-1.6688907435, -6.9502305710, 940.69592392565],  # terrace1-c0
                 [1.1984806153, -10.7495778320, 868.29873467315],
                 [0.0004069210, -0.0209324057, 0.42949125235]])

    H1 = np.mat([[0.6174778372, -0.4836875683, 147.00510919005],  # terrace2-c1
                 [0.5798503075, 3.8204849039, -386.096405131],
                 [0.0000000001, 0.0077222239, -0.01593391935]])

    H2 = np.mat([[-0.2717592338, 1.0286363982, -17.6643219215],  # terrace2-c2
                 [-0.1373600672, -0.3326731339, 161.0109069274],
                 [0.0000600052, 0.0030858398, -0.04195162855]])

    H3 = np.mat([[-0.3286861858, 0.1142963200, 130.25528281945],  # terrace2-c3
                 [0.1809954834, -0.2059386455, 125.0260427323],
                 [0.0000693641, 0.0040168154, -0.08284534995]])
    H = [H_1,H0, H1, H2]
    return H


def test_all():
    """
    产生各个身高对应的单应性矩阵
    tr:             相机外参:旋转矩阵
    k:              相机内参
    t:              相机外参:平移向量
    distort:        畸变系数 [k1,k2,k3,p1,p2]
    坐标系转换:       将世界坐标系统一到一个坐标系上
    :return:
    """
    H = {}
    # parameters for camera 111
    tr = np.mat([[0.8547,0.5039,0.1244],[-0.3268,0.3361,0.8833],[0.4033,-0.7957,0.4520]])
    k_11 = np.mat([[787.379880544512,-0.771575955278911,630.696616005492],[0,788.170017191021,347.869993514896],[0,0,1]])
    t = np.mat([2003.27764281001,738.959853408667,3113.27813131182]).T
    distort1 = np.mat([-0.4289, 0.2202, 0.00022492, 0.00041353, -0.0601])
    T_1to3 = np.mat([[-1, 0, 2290], [0, -1, 90], [0, 0, 1]])
    H1 = generate_all(k_11,tr,t,T_1to3)


    # parameters for camera 112
    tr = np.mat([[0.9261, -0.3582, 0.1186], [0.1134, 0.5638, 0.8181], [-0.3599, -0.7442, 0.5627]])   # camera 112
    k_12 = np.mat([[771.8073, -0.2406, 650.9571], [0, 771.5456, 362.0549], [0, 0, 1]])
    t = np.mat([-1046.3, 1101.2, 2910.8]).T
    distort2 = np.mat([-0.4289, 0.2202, 0.00022492, 0.00041353, -0.0601])
    T_2to3 = np.mat([[1, 0, 3242], [0, 1, 1525], [0, 0, 1]])
    H2 = generate_all(k_12,tr,t,T_2to3)


    # parameters for camera 113
    tr = np.mat([[-0.578384912213472, 0.813992422826668, 0.0537329405912692],
                 [-0.430686982345942, -0.360638338414673, 0.827314155628471],
                 [0.692805612391788, 0.455363947240309, 0.559163713944463]])# camera 113
    k_13 = np.mat([[752.097468966757,0.130073400861310,609.249312219661],[0,752.447972467573,379.619211022089],[0,0,1]])
    t = np.mat([382.564077439823,816.090711641218,2979.89865504004]).T
    T_3to3 = np.identity(3)
    distort3 = np.mat([-0.4289, 0.2202, 0.00022492, 0.00041353, -0.0601])
    H3 = generate_all(k_13,tr,t,T_3to3)

    H[1] = [H1,distort1,k_11]
    H[2] = [H2,distort2,k_12]
    H[3] = [H3,distort3,k_13]

    return H


'''
    generate homography matrix for different height plane
'''
# Parameters
#     ----------
#     k : np.mat
#         Instrinsic matrix
#     tr: np.mat
#         Rotation matrix
#     t : np.mat
#         Translation vector
#     coor_vec: np.mat
#         Translation vector for to map different world-coordinate into one world-coordinate
#     height_list : list
#         [0,150,155,160,165,170,175,180,185,190,195,200] for default
#
def generate_all(k,tr,t,coor_vec,height_list=self_cfg.TEST.height_list):
    """
    根据不同高度,产生对应的单应性矩阵
    :param k:               相机内参
    :param tr:              相机外参:旋转矩阵
    :param t:               相机外参:平移向量
    :param coor_vec:        坐标系转换向量
    :param height_list:     高度list
    :return: 各个高度对应的
    """
    H = {}
    for h in height_list:
        z = h*10
        z_height = np.mat([0, 0, -z, 1]).T
        RT = np.hstack((tr, t))
        t_height = RT * z_height
        RT_height = np.hstack((tr, t_height))
        RT_height = RT_height[:, [0, 1, 3]]
        H_t = k * RT_height
        H_t = H_t / H_t[2, 2]
        H_t = coor_vec*H_t.I
        H[h] = H_t
    return H

if __name__ == '__main__':
    H = test_all()
    pass
