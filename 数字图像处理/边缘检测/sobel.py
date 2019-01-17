import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc
import os


# 获得一个拉普拉斯锐化算子，此处均采用3*3的算子
# 参数：    center：锐化算子中心的权重（4 或 8）
def get_sobel_kernel(x):
    sobel_kernel = np.zeros((3, 3), dtype=np.int32)
    sobel_kernel = np.array([
        [[-1, -4, -1],
         [0, 0, 0],
         [1, 4, 1]],    # Gx    检测水平边沿

        [[-1, 0, 1],
         [-4, 0, 4],
         [-1, 0, 1]]    # Gy    检测竖直边沿
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
    path_dir = r'E:\F disk\GitHub\CS-Course-Notes\数字图像处理\边缘检测'
    # lenna图片路径
    path = os.path.join(path_dir, 'lenna.jpg')
    img = np.array(Image.open(path), dtype=np.int32)

    #################################################
    # sobel边缘检测
    sobel_kernel_x = get_sobel_kernel(0)
    Gx = filter(img, sobel_kernel_x)
    filtered_img = Gx + img
    filtered_img[filtered_img > 255] = 255
    filtered_img[filtered_img < 0] = 0
    scipy.misc.toimage(np.abs(Gx)).save(
        os.path.join(path_dir, 'Gx_4.jpg'))
    scipy.misc.toimage(filtered_img).save(
        os.path.join(path_dir, 'sobel_Gx.jpg'))

    sobel_kernel_y = get_sobel_kernel(1)
    Gy = filter(img, sobel_kernel_y)
    filtered_img = Gy + img
    filtered_img[filtered_img > 255] = 255
    filtered_img[filtered_img < 0] = 0
    scipy.misc.toimage(np.abs(Gy)).save(
        os.path.join(path_dir, 'Gy_4.jpg'))
    scipy.misc.toimage(filtered_img).save(
        os.path.join(path_dir, 'sobel_Gy.jpg'))

    scipy.misc.toimage(np.abs(Gy) + np.abs(Gx)).save(
        os.path.join(path_dir, 'Gx+Gy_4.jpg'))


#   对比实验结果
def display():
    path_dir = r'E:\F disk\GitHub\CS-Course-Notes\数字图像处理\边缘检测'
    ####################################################
    # 拉普拉斯滤波后图像与原图像的比较
    plt.subplot(1, 3, 1)
    img = plt.imread(os.path.join(path_dir, 'Gx.jpg'))
    plt.imshow(img, cmap='gray')
    plt.title('|Gx_2|')

    plt.subplot(1, 3, 2)
    img = plt.imread(os.path.join(path_dir, 'Gy.jpg'))
    plt.imshow(img, cmap='gray')
    plt.title('|Gy_2|')

    plt.subplot(1, 3, 3)
    img = plt.imread(os.path.join(path_dir, 'Gx+Gy.jpg'))
    plt.imshow(img, cmap='gray')
    plt.title('|Gx|+|Gy|_2')

    plt.show()
    # ####################################################
    # # 高斯核方差sigma相同时，不同大小高斯核滤波后图像与原图像的比较
    # plt.subplot(1, 3, 1)
    # img = plt.imread(os.path.join(path_dir, 'lenna.jpg'))
    # plt.imshow(img, cmap='gray')
    # plt.title('origin')

    # plt.subplot(1, 3, 2)
    # img = plt.imread(os.path.join(path_dir, 'gaussian_3_3_lenna.jpg'))
    # plt.imshow(img, cmap='gray')
    # plt.title('sigma=3, kernel_size=3')

    # plt.subplot(1, 3, 3)
    # img = plt.imread(os.path.join(path_dir, 'gaussian_3_5_lenna.jpg'))
    # plt.imshow(img, cmap='gray')
    # plt.title('sigma=3, kernel_size=5')

    # plt.show()
    # ####################################################
    # # 高斯核size相同时，不同sigma高斯核滤波后图像与原图像的比较
    # plt.subplot(1, 3, 1)
    # img = plt.imread(os.path.join(path_dir, 'lenna.jpg'))
    # plt.imshow(img, cmap='gray')
    # plt.title('origin')

    # plt.subplot(1, 3, 2)
    # img = plt.imread(os.path.join(path_dir, 'gaussian_1_5_lenna.jpg'))
    # plt.imshow(img, cmap='gray')
    # plt.title('sigma=1, kernel_size=5')

    # plt.subplot(1, 3, 3)
    # img = plt.imread(os.path.join(path_dir, 'gaussian_3_5_lenna.jpg'))
    # plt.imshow(img, cmap='gray')
    # plt.title('sigma=3, kernel_size=5')

    # plt.show()


display()

if __name__ == '__main__':
    main()
