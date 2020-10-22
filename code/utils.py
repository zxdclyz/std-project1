import numpy as np

def cart2sph(x,y,z):
    '''
    直角坐标转化为球坐标，但此处的[x,y,z]对应标准球坐标系中的[y,z,x]

    输入:
        * x,y,z 直角坐标
        
    输出:
        * azimuth, elevation, r 角度值的球坐标
    '''
    r2d = 180.0/np.pi
    azimuth = np.arctan2(x,z)*r2d
    elevation = np.arctan2(y,np.sqrt(z**2 + x**2))*r2d
    r = np.sqrt(z**2 + x**2 + y**2)
    return azimuth, elevation, r

def sph2cart(azimuth,elevation,r):
    '''
    球坐标转化为直角坐标，但此处的[x,y,z]对应标准球坐标系中的[y,z,x]

    输入:
        * azimuth, elevation, r 角度值的球坐标
        
    输出:
        * x,y,z 直角坐标
    '''
    d2r = np.pi/180.0
    z = r * np.cos(elevation*d2r) * np.cos(azimuth*d2r)
    x = r * np.cos(elevation*d2r) * np.sin(azimuth*d2r)
    y = r * np.sin(elevation*d2r)
    return x, y, z