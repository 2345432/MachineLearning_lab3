import os
from PIL import Image
from feature import NPDFeature
import pickle
import numpy as np
def get_path(path):
    '''返回目录中所有JPG图像的文件名列表'''
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

#获取原始所有JPG图像的路径，前500个为人脸，后500个为非人脸
face = get_path(u'C:\\Users\\Administrator\\Desktop\\ML2017-lab-03-master\\datasets\\original\\face')
nonface = get_path(u'C:\\Users\\Administrator\\Desktop\\ML2017-lab-03-master\\datasets\\original\\nonface')
pic=face+nonface

for i in range(len(pic)):
    #将原始图片转化为大小为24*24的灰度图并保存
    Image.open(pic[i]).convert('L').resize((24, 24)).save('C:\\Users\\Administrator\\Desktop\\pic\\' + str(i) + '.jpg')

#获取转为灰度图的所有图像文件的路径
pic=get_path('C:\\Users\\Administrator\\Desktop\\pic')
x=[]
y=[]
for i in range(len(pic)):
    #提取图像NPD特征
    feature=NPDFeature(np.array(Image.open(pic[i]))).extract()
    feature=feature[0:2000]
    x.append(feature)
    #前500个人脸图片对应y值设为1，后500个设为-1
    if(i<500):
        y.append(1)
    else:
        y.append(-1)

#使用pickle.dump保存提取数据
pickle.dump(x, open( "C:\\Users\\Administrator\\Desktop\\data\\savex.txt", "wb" ) )
pickle.dump(y, open( "C:\\Users\\Administrator\\Desktop\\data\\savey.txt", "wb" ) )