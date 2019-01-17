import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc
import os
import queue

# 获得一个sobel算子，此处均采用3*3的算子
def get_sobel_kernel(x):
    sobel_kernel = np.zeros((3, 3), dtype=np.int16)
    sobel_kernel = np.array([
        [[-1, -2, -1],
         [0, 0, 0],
         [1, 2, 1]],    # Gx

        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]]    # Gy
    ])
    return sobel_kernel[x]


# 获得一个高斯滤波算子，可以选择算子的大小以及方差sigma的取值
# 参数：    sigma：方差
#           kernel_size:算子大小
def get_gaussian_kernel(sigma=3, kernel_size=3):
    # 初始化高斯核全0
    guassian_kernel = np.zeros((kernel_size, kernel_size))

    # 计算像素点的平方和距离
    def distance(x, y): return (x - int(kernel_size/2))**2 + \
        (y - int(kernel_size/2))**2

    # 计算高斯函数值
    def gaussian_func(x, y): return np.exp(-distance(x, y) /
                                           (2*sigma*sigma)) / (2*3.1416*sigma*sigma)
    # i，j位置使用高斯核进行卷积
    total = 0.
    for i in range(kernel_size):
        for j in range(kernel_size):
            guassian_kernel[i][j] = gaussian_func(i, j)
            total += guassian_kernel[i][j]

    # 高斯核归一化
    guassian_kernel = guassian_kernel/total

    return guassian_kernel


# 利用滤波算子在图像上滑动进行滤波
# 参数：    img：图片数组
#           kernel：滤波算子
def filter(img, kernel):
    kernel_size = kernel.shape[0]
    img_height = img.shape[0]
    img_width = img.shape[1]
    # 初始化滤波后的图片全0
    filtered_img = np.zeros(img.shape, dtype=np.int16)

    # offset偏移量
    offset = int(kernel_size/2)
    # 滑动算子，计算卷积后的取值
    for i in range(offset, img_height-offset):
        for j in range(offset, img_width-offset):
            total = 0
            for m in range(kernel_size):
                for n in range(kernel_size):
                    total += kernel[m][n]*img[i-offset+m][j-offset+n]
            filtered_img[i][j] = total

    return filtered_img

# 非极大值抑制
def suppression(Gx_Gy, theta):
    shape = theta.shape
    gN = np.zeros(shape, dtype=np.int16)
    for i in range(1, shape[0]-1):
        for j in range(1, shape[1]-1):
            if theta[i][j] == 0:
                left = Gx_Gy[i-1][j]
                right = Gx_Gy[i+1][j]
            elif theta[i][j] == 1:
                left = Gx_Gy[i][j-1]
                right = Gx_Gy[i][j+1]
            elif theta[i][j] == 45:
                left = Gx_Gy[i+1][j-1]
                right = Gx_Gy[i-1][j+1]
            else:
                left = Gx_Gy[i-1][j-1]
                right = Gx_Gy[i+1][j+1]

            if Gx_Gy[i][j] < max(left, right):
                gN[i][j] = 0
            else:
                gN[i][j] = Gx_Gy[i][j]

    return gN

# 广度优先搜索
def bfs(i, j, gN_L, gN_H):
    mark = np.zeros(gN_L.shape, dtype=np.int16)
    mark[i][j] = 1
    Q = queue.Queue()
    Q.put(np.array([i, j]))
    while(not Q.empty()):
        t = Q.get()
        if gN_H[t[0], t[1]] != 0:
            gN_H[i][j] = gN_L[i][j]
            break
        if gN_L[i-1][j-1] != 0 and mark[i-1][j-1] == 0:
            mark[i-1][j-1] = 1
            Q.put(np.array([i-1, j-1]))
        if gN_L[i-1][j] != 0 and mark[i-1][j] == 0:
            mark[i-1][j] = 1
            Q.put(np.array([i-1, j]))
        if gN_L[i-1][j+1] != 0 and mark[i-1][j+1] == 0:
            mark[i-1][j+1] = 1
            Q.put(np.array([i-1, j+1]))
        if gN_L[i][j-1] != 0 and mark[i][j-1] == 0:
            mark[i][j-1] = 1
            Q.put(np.array([i, j-1]))
        if gN_L[i][j+1] != 0 and mark[i][j+1] == 0:
            mark[i][j+1] = 1
            Q.put(np.array([i-1, j-1]))
        if gN_L[i+1][j-1] != 0 and mark[i+1][j-1] == 0:
            mark[i+1][j-1] = 1
            Q.put(np.array([i+1, j-1]))
        if gN_L[i+1][j] != 0 and mark[i+1][j] == 0:
            mark[i+1][j] = 1
            Q.put(np.array([i+1, j]))
        if gN_L[i+1][j+1] != 0 and mark[i+1][j+1] == 0:
            mark[i+1][j+1] = 1
            Q.put(np.array([i+1, j+1]))

        if gN_H[i][j] == gN_L[i][j]:
            target = np.where(mark == 1)
            for i, _ in enumerate(target[0]):
                gN_H[target[0][i], target[1][i]
                     ] = gN_L[target[0][i], target[1][i]]

# 边缘细化
def thinning(gN):
    x = [-1, -1, -1, 0]
    y = [1, 0, -1, -1]
    layers1 = np.zeros(gN.shape, dtype=np.int16)
    for i in range(1, gN.shape[0]-1):
        for j in range(1, gN.shape[1]-1):
            if gN[i][j] != 0:
                t = min([layers1[i + x[m]][j + y[m]] for m in range(4)])
                layers1[i][j] = t+1
    x = [1, 1, 1, 0]
    y = [-1, 0, 1, 1]
    layers2 = np.zeros(gN.shape, dtype=np.int16)
    for i in range(1, gN.shape[0]-1):
        i = gN.shape[0]-1-i
        for j in range(1, gN.shape[1]-1):
            j = gN.shape[1]-1-j
            if gN[i][j] != 0:
                t = min([layers1[i + x[m]][j + y[m]] for m in range(4)])
                layers2[i][j] = t+1
    x = [-1, -1, -1, 0]
    y = [1, 0, -1, 1]
    layers3 = np.zeros(gN.shape, dtype=np.int16)
    for i in range(1, gN.shape[0]-1):
        for j in range(1, gN.shape[1]-1):
            if gN[i][j] != 0:
                t = min([layers3[i + x[m]][j + y[m]] for m in range(4)])
                layers3[i][j] = t+1
    x = [1, 1, 1, 0]
    y = [1, 0, -1, -1]
    layers4 = np.zeros(gN.shape, dtype=np.int16)
    for i in range(1, gN.shape[0]-1):
        for j in range(1, gN.shape[1]-1):
            if gN[i][j] != 0:
                t = min([layers4[i + x[m]][j + y[m]] for m in range(4)])
                layers4[i][j] = t+1

    target = np.where(layers1 > layers2)
    for i, _ in enumerate(target[0]):
        layers1[target[0][i]][target[1][i]
                              ] = layers2[target[0][i]][target[1][i]]

    target = np.where(layers1 > layers3)
    for i, _ in enumerate(target[0]):
        layers1[target[0][i]][target[1][i]
                              ] = layers3[target[0][i]][target[1][i]]

    target = np.where(layers1 > layers4)
    for i, _ in enumerate(target[0]):
        layers1[target[0][i]][target[1][i]
                              ] = layers4[target[0][i]][target[1][i]]

    x = [-1, -1, -1, 0, 0, 1, 1, 1]
    y = [-1, 0, 1, -1, 1, -1, 0, 1]
    for i in range(1, gN.shape[0]-1):
        for j in range(1, gN.shape[1]-1):
            if gN[i][j] > 0 and (not layers1[i][j] >= max([layers1[i + x[m]][j + y[m]] for m in range(8)])):
                gN[i][j] = 0

    return gN

# 双阈值处理和连接分析
def double_threshold(gN, low=60, high=180):
    gN_H = gN.copy()
    gN_H[gN_H < high] = 0

    gN_L = gN.copy()
    gN_L[gN_L < low] = 0
    gN_L[gN_L >= high] = 0

    for i in range(1, gN.shape[0]-1):
        for j in range(1, gN.shape[1]-1):
            if gN_L[i][j] != 0 and gN_H[i][j] == 0:
                bfs(i, j, gN_L, gN_H)

    return gN_H


def main():
    # 设置文件目录
    path_dir = r'E:\F disk\GitHub\CS-Course-Notes\数字图像处理\边缘检测'
    # lenna图片路径
    path = os.path.join(path_dir, 'lenna.jpg')
    img = np.array(Image.open(path), dtype=np.int16)

    # 高斯滤波平滑
    guassian_kernel = get_gaussian_kernel(1, 7)
    filtered_img = filter(img, guassian_kernel)
    filtered_img[filtered_img > 255] = 255
    target = np.where(filtered_img == 0)
    for i, _ in enumerate(target[0]):
        filtered_img[target[0][i]][target[1][i]
                                   ] = img[target[0][i]][target[1][i]]
    #################################################
    # sobel边缘检测
    sobel_kernel_x = get_sobel_kernel(0)
    Gx = filter(filtered_img, sobel_kernel_x)
    sobel_kernel_y = get_sobel_kernel(1)
    Gy = filter(filtered_img, sobel_kernel_y)
    Gx_Gy = np.abs(Gx) + np.abs(Gy)

    # 非极大值抑制
    theta = np.arctan2(Gy, Gx)*180/np.pi
    theta[theta >= -90] = 1
    theta[theta >= -67.5] = 45
    theta[theta >= -22.5] = 0
    theta[theta >= 22.5] = -45
    theta[theta >= -67.5] = 1
    gN = suppression(Gx_Gy, theta)
    print(gN.max())

    # 双阈值处理和链接分析
    gN = double_threshold(gN, 65, 195)
    scipy.misc.toimage(gN).save(
        os.path.join(path_dir, 'double_threshold_65-195.jpg'))

    # 边缘细化
    gN = thinning(Gx_Gy)
    scipy.misc.toimage(gN).save(
        os.path.join(path_dir, 'canny.jpg'))

# display()


if __name__ == '__main__':
    main()
