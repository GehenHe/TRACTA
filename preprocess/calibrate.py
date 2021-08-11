#coding:utf-8
import cv2
import numpy as np
import glob

# def undistort_Points(input,inter_matrix,distor):
#     fx = inter_matrix[0, 0]
#     s = inter_matrix[0, 1]
#     cx = inter_matrix[2, 0]
#     fy = inter_matrix[1, 1]
#     cy = inter_matrix[1, 2]
#
#     rst = 1
#     return rst\
def img_to_world(points, k, distor, H):
    img_Points = cv2.undistortPoints(points, k, distor, P=k)
    input = np.mat([img_Points[0][0][0], img_Points[0][0][1], 1])
    world_points = H.I * input.T
    world_points = world_points/world_points[2]
    return world_points

# 113 camera params
# tr = np.mat([[-0.5768, 0.8154, 0.0484], [-0.4347, -0.3566, 0.8269], [0.6916, 0.4559, 0.5602]])
# k = np.mat([[769.0224, 0.0611, 611.1722], [0, 768.5477, 379.6660], [0, 0, 1]])
# t = np.mat([392.4461, 844.6573, 3032]).T
dist2 = np.mat([-0.4289, 0.2202, 0.00022492, 0.00041353, -0.0601]) # the best distortion

tr = np.mat([[-0.5772, 0.8152, 0.0479], [-0.4285, -0.3523, 0.8320], [0.6951, 0.4597, 0.5527]])
k = np.mat([[752.9527, 0.2847, 614.3600], [0, 754.1132, 378.0666], [0, 0, 1]])
t = np.mat([378.8820, 848.4238, 2964.7]).T
# dist2 = np.mat([-0.4220, 0.2159, 0.00077833, 0.00075903, -0.0515])

# 111 camera params
# tr = np.mat([[0.8588, 0.4934, 0.1380], [-0.3372, 0.3414, 0.8774], [0.3858, -0.8000, 0.4595]])
# k = np.mat([[759.9343, -0.0285, 632.8419], [0, 759.9635, 359.3356], [0, 0, 1]])
# t = np.mat([2002.9, 667.8282, 2962.9]).T
# dist2 = np.mat([-0.4005, 0.1717, 0.000091356, 0.00051993, -0.0357])
#  0 plane H
RT = np.hstack((tr, t))
RT1 = RT[:, [0, 1, 3]]
H = k * RT1
H = H/H[2, 2]
zz = np.mat([0, 0, 1]).T
mm = H * zz
# dist_image points to world points
dist_imgp = np.array([1182, 374], dtype=np.float32).reshape([1, 1, -1])
dst = img_to_world(dist_imgp, k, dist2, H)


RT = np.hstack((tr, t))
RT1 = RT[:, [0, 1, 3]]
H = k * RT1
H = H/H[2, 2]
z = np.mat([0, 0, -1750, 1]).T
m = RT * z
RT175 = np.hstack((tr, m))
RT175 = RT175[:, [0, 1, 3]]
HH = k * RT175
src = np.array([1230, 205], dtype=np.float32).reshape([1, 1, -1])
dst = cv2.undistortPoints(src, k, dist2, P=k)
d_input = np.mat([dst[0][0][0], dst[0][0][1], 1])
word_p = HH.I * d_input.T
word_p = word_p/word_p[2]
word_p2 = H.I * m
# matlab params
# mtx2 = np.mat([[1079.9, 0, 649.6852], [0, 1081.8, 358.4251], [0, 0, 1]]) #i=28 k=2 2
# dist2 = np.mat([-0.3796, 0.1427, 0, 0])

mtx2 = np.array([[1076.5, -0.2855, 657.6118], [0, 1077.7, 371.9572], [0, 0, 1]]) #i=28 k=3 4
dist2 = np.array([-0.3786, 0.1273, 0.00083051, 0.0016, 0.0285])
R = np.array([[-1033, -0.9932, -0.0531], [0.9135, -0.1159, 0.3900], [-0.3935, -0.0082, 0.9193]])
# mtx2 = np.mat([[839.7588, 209.3408, 757.9951], [0, 446.2765, 92.0305], [0, 0, 1]]) # i=3 k=2 3
# dist2 = np.mat([0.4225, -0.3819, -0.0022, 0.00048137, 0.1042])
# 去畸变
img2 = cv2.imread('/home/dinner/data/calibrate/A_camera/14.jpg')
# h,  w = img2.shape[:2]
# newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h)) # 自由比例参数d_
# src = np.array([890, 342], dtype=np.float32).reshape([1, 1, -1])
dst = cv2.undistort(img2, mtx2, dist2, None, mtx2)
# deal distortion
dst = cv2.undistortPoints(src, mtx2, dist2, P = mtx2)

ddst = cv2.undistort_Points(src, mtx2, dist2)
# 根据前面ROI区域裁剪图片
#x,y,w,h = roi
#dst = dst[y:y+h, x:x+w]
# cv2.imwrite('opencv_4.png',dst)


