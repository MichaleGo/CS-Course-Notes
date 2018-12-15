import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc
import os


# 获得一个拉普拉斯锐化算子，此处均采用3*3的算子
# 参数：    center：锐化算子中心的权重（4 或 8）
def get_laplas_kernel(center=4):
    laplas_kernel = np.zeros((3, 3), dtype=np.int32)
    d = {4: 0, 8: 1}
    laplas_kernel = np.array([
        [[0, -1, 0],
         [-1, 4, -1],
         [0, -1, 0]],

        [[-1, -1, -1],
         [-1, 8, -1],
         [-1, -1, -1]]
    ])
    return laplas_kernel[d[center]]


# 获得一个高斯滤波算子，可以选择算子的大小以及方差sigma的取值
# 参数：    sigma：方差
#           kernel_size:算子大小
def get_gaussian_kernel(sigma=3, kernel_size=3):
    # 初始化高斯核全0
    guassian_kernel = np.zeros((kernel_size, kernel_size))

    # 计算像素点的平方和距离
    distance = lambda x, y: (x - int(kernel_size/2))**2 + \
        (y - int(kernel_size/2))**2

    # 计算高斯函数值
    gaussian_func = lambda x, y: np.exp(-distance(x, y) /
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
    filtered_img = np.zeros(img.shape, dtype=np.int32)

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


def main():
    # 设置文件目录
    path_dir = r'E:\F disk\GitHub\CS-Course-Notes\数字图像处理\图像处理上机大作业'
    # lenna图片路径
    path = os.path.join(path_dir, 'lenna.jpg')
    img = np.array(Image.open(path), dtype=np.int32)

    #################################################
    # 拉普拉斯滤波
    laplas_kernel = get_laplas_kernel(4)
    filtered_img = filter(img, laplas_kernel) + img
    # 拉普拉斯灰度滤波会存在滤波后灰度值>255的情况，冲突处理有两个办法
    # 方法一：采取归一化的方法，但是会在平滑的部分失真
    # filtered_img = (255.0 / (filtered_img.max()-filtered_img.min()) * (filtered_img - filtered_img.min())).astype(np.uint8)
    # 方法二：如下
    filtered_img[filtered_img > 255] = 255
    filtered_img[filtered_img < 0] = 0
    scipy.misc.toimage(filtered_img).save(
        os.path.join(path_dir, 'laplas_4_lenna.jpg'))

    #################################################
    # 高斯滤波
    # img = np.array(Image.open(os.path.join(path_dir, 'lenna_with_gaussian_noise.png')))
    guassian_kernel = get_gaussian_kernel(3, 5)
    filtered_img = filter(img, guassian_kernel)
    filtered_img[filtered_img > 255] = 255
    scipy.misc.toimage(filtered_img).save(
        os.path.join(path_dir, 'gaussian_3_5_lenna.jpg'))

#   对比实验结果
def display():
    path_dir = r'E:\F disk\GitHub\CS-Course-Notes\数字图像处理\图像处理上机大作业'
    ####################################################
    # 拉普拉斯滤波后图像与原图像的比较
    plt.subplot(1, 3, 1)
    img = plt.imread(os.path.join(path_dir, 'lenna.jpg'))
    plt.imshow(img, cmap='gray')
    plt.title('origin')

    plt.subplot(1, 3, 2)
    img = plt.imread(os.path.join(path_dir, 'laplas_4_lenna.jpg'))
    plt.imshow(img, cmap='gray')
    plt.title('laplas kernel_center=4')

    plt.subplot(1, 3, 3)
    img = plt.imread(os.path.join(path_dir, 'laplas_8_lenna.jpg'))
    plt.imshow(img, cmap='gray')
    plt.title('laplas kernel_center=8')

    plt.show()
    ####################################################
    # 高斯核方差sigma相同时，不同大小高斯核滤波后图像与原图像的比较
    plt.subplot(1, 3, 1)
    img = plt.imread(os.path.join(path_dir, 'lenna.jpg'))
    plt.imshow(img, cmap='gray')
    plt.title('origin')

    plt.subplot(1, 3, 2)
    img = plt.imread(os.path.join(path_dir, 'gaussian_3_3_lenna.jpg'))
    plt.imshow(img, cmap='gray')
    plt.title('sigma=3, kernel_size=3')

    plt.subplot(1, 3, 3)
    img = plt.imread(os.path.join(path_dir, 'gaussian_3_5_lenna.jpg'))
    plt.imshow(img, cmap='gray')
    plt.title('sigma=3, kernel_size=5')

    plt.show()
    ####################################################
    # 高斯核size相同时，不同sigma高斯核滤波后图像与原图像的比较
    plt.subplot(1, 3, 1)
    img = plt.imread(os.path.join(path_dir, 'lenna.jpg'))
    plt.imshow(img, cmap='gray')
    plt.title('origin')

    plt.subplot(1, 3, 2)
    img = plt.imread(os.path.join(path_dir, 'gaussian_1_5_lenna.jpg'))
    plt.imshow(img, cmap='gray')
    plt.title('sigma=1, kernel_size=5')

    plt.subplot(1, 3, 3)
    img = plt.imread(os.path.join(path_dir, 'gaussian_3_5_lenna.jpg'))
    plt.imshow(img, cmap='gray')
    plt.title('sigma=3, kernel_size=5')

    plt.show()


if __name__ == '__main__':
    main()
    display()
