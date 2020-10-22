import numpy as np
import os
import cv2

def rendering(dir):
    #z的尺度与x和y相同，大小等同于测试图像大小，位置与测试图像像素点一一对应
    #imgs为渲染结果，大小等同于测试图像大小，位置与测试图像像素点一一对应

    # 读取train.txt
    train_data = []
    with open(os.path.join(dir, 'train.txt'), 'r') as f:
        for line in f.readlines():
            train_data.append(line.strip('\n').split(','))
    # 读取test.txt
    test_data = []
    with open(os.path.join(dir, 'test.txt'), 'r') as f:
        for line in f.readlines():
            test_data.append(line.strip('\n').split(','))
    # 读取训练集图片 
    train_images = []
    for i in range(len(train_data)):
        train_img = cv2.imread(os.path.join(dir,'train', train_data[i][0] + '.bmp'), 0)
        train_images.append(train_img)

    z = np.zeros([168,168])
    imgs=np.zeros([10, 168, 168]).astype(np.uint8)
    return z, imgs
