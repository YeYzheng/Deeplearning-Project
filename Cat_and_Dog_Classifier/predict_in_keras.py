#-*- coding:utf-8 -*-
# create at  on YeY
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import argparse

parser = argparse.ArgumentParser('Predict Image Params')
parser.add_argument('ImageFile',type = str,help = 'Image File Path')
parser.add_argument('--plot','-p',action = 'store_true',help = 'Plot Image ')
args  = parser.parse_args()
#模型文件地址(根据本地地址改变)
model_path = 'Model_save/fine_tune_Vgg16_model.h5'


def predict(img_path,plot):
    '''
    猫狗识别器
    Argument:
        img_path -- 图片文件地址
        plot -- 是否打印图片
    '''
    model = load_model(model_path)
    image = cv2.imread(img_path,flags=cv2.IMREAD_COLOR)
    image = cv2.resize(image,(244,244))
    img = np.reshape(image,(1,image.shape[0],image.shape[1],image.shape[2]))
    pred = np.argmax(model.predict(img))
    if pred == 0:
        pred_class = 'cat'
    elif pred == 1:
        pred_class = 'dog'
    if plot:
        plt.imshow(image)
        plt.title('This is a {}'.format(pred_class))
        plt.show()
    else:
        print('This is a {}'.format(pred_class))

if __name__ == '__main__':
    predict(args.ImageFile,args.plot)