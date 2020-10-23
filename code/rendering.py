import numpy as np
import os
import cv2
from utils import sph2cart
from scipy.optimize import curve_fit
from scipy.linalg import lstsq


def rendering(dir):
    #z的尺度与x和y相同，大小等同于测试图像大小，位置与测试图像像素点一一对应
    #imgs为渲染结果，大小等同于测试图像大小，位置与测试图像像素点一一对应

    # 读取train.txt
    train_data = []
    with open(os.path.join(dir, 'train.txt'), 'r') as f:
        for line in f.readlines():
            train_data.append(line.strip('\n').split(','))
    
    # train_data转化为直角坐标
    train_s = []
    for data in train_data:
        train_s.append(sph2cart(int(data[1]),int(data[2]),1))
    train_s = np.array(train_s)
    
    # 读取test.txt
    test_data = []
    with open(os.path.join(dir, 'test.txt'), 'r') as f:
        for line in f.readlines():
            test_data.append(line.strip('\n').split(','))

    # test_data转化为直角坐标
    test_s = []
    for data in test_data:
        test_s.append(sph2cart(int(data[1]),int(data[2]),1))
    test_s = np.array(test_s)
    
    # 读取训练集图片 
    train_images = []
    for i in range(len(train_data)):
        train_img = cv2.imread(os.path.join(dir,'train', train_data[i][0] + '.bmp'), 0)
        train_img = train_img.flatten()
        train_images.append(train_img)
    
    train_data = np.zeros([len(train_images),len(train_images[0])])
    for i in range(0,len(train_images)):
        train_data[i,:] = train_images[i]

    # 使用最小二乘法计算B
    b,_,_,_ = lstsq(train_s,train_data)

    # 进行测试
    img_r = test_s @ b

    # 生成测试图像
    imgs = []
    for img in img_r:
        imgs.append(np.reshape(img,[168,168]).astype(np.uint8))

    z = np.zeros([168,168])
    
    return z, imgs
