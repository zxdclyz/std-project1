import numpy as np

def rendering(dir):
    #z的尺度与x和y相同，大小等同于测试图像大小，位置与测试图像像素点一一对应
    #imgs为渲染结果，大小等同于测试图像大小，位置与测试图像像素点一一对应
    z = np.zeros([168,168])
    imgs=np.zeros([10, 168, 168]).astype(np.uint8)
    return z, imgs