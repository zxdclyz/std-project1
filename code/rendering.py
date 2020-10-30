import numpy as np
import os
import cv2
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.core.fromnumeric import reshape
from utils import sph2cart
from scipy.optimize import curve_fit
from scipy.linalg import lstsq


def dct(z_x, z_y, N):
    t = np.arange(N).reshape(1, -1)
    D = np.sqrt(2 / N) * np.cos(t.T * (2 * t + 1) * np.pi / (2 * N))
    D[0, :] = np.sqrt(1 / N)

    dD = - np.sqrt(2 / N) * np.sin(t.T * (2 * t + 1)
                                   * np.pi / (2 * N)) * np.pi / N
    dD[0, :] = np.sqrt(1 / N)
    dD = dD * t.T
    dD = dD.T
    dD_i = np.linalg.pinv(dD)

    C_1 = D @ z_x.reshape([N, N]) @ dD_i.T
    C_2 = dD_i @ z_y.reshape([N, N]) @ D.T
    return [C_1, C_2, D, dD, dD_i]


# def dist(C, C_1, C_2, P_x, P_y):
#     C_r = C.reshape(list(C_1.shape))
#     return np.sum((C_r - C_1) ** 2 * P_x + (C_r - C_2) ** 2 * P_y)


def rendering(dir):
    # z的尺度与x和y相同，大小等同于测试图像大小，位置与测试图像像素点一一对应
    # imgs为渲染结果，大小等同于测试图像大小，位置与测试图像像素点一一对应

    # 读取train.txt
    train_data = []
    with open(os.path.join(dir, 'train.txt'), 'r') as f:
        for line in f.readlines():
            train_data.append(line.strip('\n').split(','))

    # train_data转化为直角坐标
    train_s = []
    for data in train_data:
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
    for i in range(len(train_data)):
        train_img = cv2.imread(os.path.join(
            dir, 'train', train_data[i][0] + '.bmp'), 0)
        train_img = train_img.flatten()
        train_images.append(train_img)

    train_data = np.zeros([len(train_images), len(train_images[0])])
    for i in range(0, len(train_images)):
        train_data[i, :] = train_images[i]

    kd = np.mean(train_data, axis=0, keepdims=True)

    # 使用最小二乘法计算B
    B_star, _, _, _ = lstsq(train_s, train_data)

    t = B_star/kd
    [zx_star, zy_star, _] = -t/t[2, :]

    [C_1, C_2, D, dD, dD_i] = dct(zx_star, zy_star, 168)

    D_2 = D ** 2
    dD_2 = dD ** 2
    D_sum = np.sum(D_2, axis=0, keepdims=True)
    dD_sum = np.sum(dD_2, axis=0, keepdims=True)

    P_x = (D_sum.T * dD_sum)
    P_y = (dD_sum.T * D_sum)

    C = C_1 + (C_2 - C_1) * (P_y / (P_x + P_y + 1e-15))
    # C_ = C_1 + (C_2 - C_1) * (1 - 1 / (1 + np.sqrt(P_x / (P_y))))
    Z = D.T @ C @ D
    zx_bar = D.T @ C @ dD.T
    zy_bar = dD @ C @ D

    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(168)
    Y = np.arange(168)
    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(X, Y, Z, cmap="rainbow")
    plt.show()

    # z_img = np.round((Z - np.min(Z)) / np.max(Z - np.min(Z)) * 255)
    # cv2.imwrite("z.jpg", z_img.astype(np.uint8))
    # zx_img = np.round((zx_bar - np.min(zx_bar)) /
    #                   np.max(zx_bar - np.min(zx_bar)) * 255)
    # cv2.imwrite("zx.jpg", zx_img.astype(np.uint8))
    # zy_img = np.round((zy_bar - np.min(zy_bar)) /
    #                   np.max(zy_bar - np.min(zy_bar)) * 255)
    # cv2.imwrite("zy.jpg", zy_img.astype(np.uint8))

    # 进行测试
    img_r = test_s @ B_star

    # 生成测试图像
    imgs = []
    for img in img_r:
        imgs.append(np.reshape(img, [168, 168]).astype(np.uint8))

    z = np.zeros([168, 168])

    return Z, imgs


def main():
    pass


if __name__ == "__main__":
    main()
