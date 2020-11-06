import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import sph2cart
from scipy.linalg import lstsq
from numpy.linalg import pinv, norm


def dct(z_x, z_y, N):
    '''
    DCT变换

    Args:
        z_x (ndarray): 深度z的x方向偏导
        z_y (ndarray): 深度z的y方向偏导
        N (int): 图片的宽度(图片大小是N*N)

    Returns:
        list: 包含：
                C_1: 深度z的x方向偏导DCT变换的系数矩阵
                C_2: 深度z的y方向偏导DCT变换的系数矩阵
                D: DCT变换的基函数矩阵
                dD: D求导之后的结果
    '''
    t = np.arange(N).reshape(1, -1)
    D = np.sqrt(2 / N) * np.cos(t.T * (2 * t + 1) * np.pi / (2 * N))
    D[0, :] = np.sqrt(1 / N)

    dD = -np.sqrt(2 / N) * np.sin(t.T * (2 * t + 1) * np.pi /
                                  (2 * N)) * np.pi / N
    dD[0, :] = np.sqrt(1 / N)
    dD = dD * t.T
    dD = dD.T
    dD_i = pinv(dD)

    C_1 = D @ z_x.reshape([N, N]) @ dD_i.T
    C_2 = dD_i @ z_y.reshape([N, N]) @ D.T
    return [C_1, C_2, D, dD]


def estimateAlbedo(B, z):
    '''
    计算反射系数kd

    Args:
        B (ndarray): B的估计值B_star
        z (ndarray): 深度

    Returns:
        ndarray: 反射系数kd
    '''
    # input should both be (3,n)
    n = B.shape[1]
    kd = np.zeros([n, 1])
    for i in range(n):
        kd[i] = B[:, i] @ z[:, i] / norm(z[:, i])

    return kd.T


def rendering(dir):
    # z的尺度与x和y相同，大小等同于测试图像大小，位置与测试图像像素点一一对应
    # imgs为渲染结果，大小等同于测试图像大小，位置与测试图像像素点一一对应

    # 读取train.txt
    train_data_path = []
    with open(os.path.join(dir, 'train.txt'), 'r') as f:
        for line in f.readlines():
            train_data_path.append(line.strip('\n').split(','))

    # train_data转化为直角坐标
    train_s = []
    for data in train_data_path:
        train_s.append(sph2cart(int(data[1]), int(data[2]), 1))
    train_s = np.array(train_s)

    # 读取test.txt
    test_data = []
    with open(os.path.join(dir, 'test.txt'), 'r') as f:
        for line in f.readlines():
            test_data.append(line.strip('\n').split(','))

    # test_data转化为直角坐标
    test_s = []
    for data in test_data:
        test_s.append(sph2cart(int(data[1]), int(data[2]), 1))
    test_s = np.array(test_s)

    # 读取训练集图片
    train_images = []
    for i in range(len(train_data_path)):
        train_img = cv2.imread(
            os.path.join(dir, 'train', train_data_path[i][0] + '.bmp'), 0)
        train_img = train_img.flatten()
        train_images.append(train_img)
    train_data = np.zeros([len(train_images), len(train_images[0])])
    for i in range(0, len(train_images)):
        train_data[i, :] = train_images[i]

    # 数据预处理
    kd = np.mean(train_data, axis=0, keepdims=True)
    train_data[train_data > 250] = 250
    for i in range(train_data.shape[0]):
        tmp = train_data[None, i, :]
        train_data[None, i, :][tmp / kd < 0.1] = kd[tmp / kd < 0.1] * 0.1

    # 使用最小二乘法计算B
    B_star, _, _, _ = lstsq(train_s, train_data)

    # 计算zx_star, zy_star
    [zx_star, zy_star, _] = -B_star / B_star[2, :]

    [C_1, C_2, D, dD] = dct(zx_star, zy_star, 168)

    # 求P_x, P_y
    D_2 = D**2
    dD_2 = dD**2
    D_sum = np.sum(D_2, axis=0, keepdims=True)
    dD_sum = np.sum(dD_2, axis=0, keepdims=True)
    P_x = (D_sum.T * dD_sum)
    P_y = (dD_sum.T * D_sum)

    # 根据C_1, C_2优化得到系数矩阵C, 并以此求z, zx_bar, zy_bar
    C = (C_1 * P_x + C_2 * P_y) / (P_x + P_y + 1e-15)
    z = D.T @ C @ D
    zx_bar = (D.T @ C @ dD.T).reshape([1, -1])
    zy_bar = (dD @ C @ D).reshape([1, -1])

    zxy_bar = np.concatenate([zx_bar, zy_bar, -np.ones_like(zx_bar)])
    zxy_bar = zxy_bar / np.sqrt(1 + zx_bar**2 + zy_bar**2)

    # 计算最终的kd并以此计算B
    kd_new = estimateAlbedo(B_star, zxy_bar)
    B_bar = kd_new * zxy_bar

    # 绘制人脸深度彩虹图
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # X = np.arange(168)
    # Y = np.arange(168)
    # X, Y = np.meshgrid(X, Y)
    # ax.plot_surface(X, Y, z, rstride = 1, cstride = 1, cmap="rainbow")
    # plt.show()

    # 进行测试
    img_r = test_s @ B_bar
    # Half Lambert
    # img_r = kd_new * (test_s @ zxy_bar * 0.5 - 0.5)

    # 分布平移
    # for i in range(len(img_r)):
    #     drop_num = round(img_r.shape[1] * 0.01)
    #     img_max = np.min(heapq.nlargest(drop_num, img_r[i]))
    #     img_min = np.max(heapq.nsmallest(drop_num, img_r[i]))
    #     if img_max > 255 or img_min < -5:
    #         move = 127.5 - (img_max + img_min) / 2
    #         img_r[i] = img_r[i] + move

    # 阈值法调整到0-255
    img_r[img_r < 0] = 0
    img_r[img_r > 255] = 255

    # 生成测试图像
    imgs = []
    for img in img_r:
        imgs.append(np.reshape(img, [168, 168]).astype(np.uint8))

    return z, imgs
