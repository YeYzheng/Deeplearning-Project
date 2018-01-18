#-*- coding:utf-8 -*-
# create at  on YeY

import cv2
import numpy as np
import os
from random import shuffle

#数据集文件地址（根据本地地址改变）
train_dirpath = 'data/train/'
#训练图片高宽
IMG_SIZE = 224
#训练集大小
Train_sample = 3000
#测试集大小
Test_sample = 500

def image_label(image_name):
    '''
    根据图片文件名返回图片类别标签

    Argument:
        image_name -- 图片文件名

    Return:
        label -- 2D的图片类别标签

    '''
    word_name = image_name.split('.')[0]

    if word_name == 'cat':
        label = [1,0]
    elif word_name == 'dog':
        label = [0,1]
    return label

def create_image_traindata():
    '''
    构造训练集

    '''
    data_filename = os.listdir(train_dirpath)
    cat_train_filename = data_filename[0:12500]
    dog_train_filename = data_filename[12500:]
    train_data = []
    for index in range(int(Train_sample/2)):
        img_cat_filepath = os.path.join(train_dirpath,cat_train_filename[index])
        img_dog_filepath = os.path.join(train_dirpath,dog_train_filename[index])

        #cat
        cat_label = image_label(cat_train_filename[index])
        cat_image = cv2.imread(img_cat_filepath,flags=cv2.IMREAD_COLOR)
        cat_image = cv2.resize(cat_image,(IMG_SIZE,IMG_SIZE))

        #dog
        dog_label = image_label(dog_train_filename[index])
        dog_image = cv2.imread(img_dog_filepath, flags=cv2.IMREAD_COLOR)
        dog_image = cv2.resize(dog_image, (IMG_SIZE, IMG_SIZE))

        train_data.append([cat_image,cat_label])
        train_data.append([dog_image,dog_label])

    train_data = np.array(train_data)

    shuffle(train_data)

    np.save(open('data/cat_dog_traindata.npy','wb'), train_data)

def create_image_testdata():
    '''
    构造测试集
    :return:
    '''
    data_filename = os.listdir(train_dirpath)
    cat_test_filename = data_filename[3000:12500]
    dog_test_filename = data_filename[15500:]
    test_data = []
    for index in range(int(Test_sample/2)):
        img_cat_filepath = os.path.join(train_dirpath,cat_test_filename[index])
        img_dog_filepath = os.path.join(train_dirpath,dog_test_filename[index])

        #cat
        cat_label = image_label(cat_test_filename[index])
        cat_image = cv2.imread(img_cat_filepath,flags=cv2.IMREAD_COLOR)
        cat_image = cv2.resize(cat_image,(IMG_SIZE,IMG_SIZE))

        #dog
        dog_label = image_label(dog_test_filename[index])
        dog_image = cv2.imread(img_dog_filepath, flags=cv2.IMREAD_COLOR)
        dog_image = cv2.resize(dog_image, (IMG_SIZE, IMG_SIZE))

        test_data.append([cat_image,cat_label])
        test_data.append([dog_image,dog_label])

    test_data = np.array(test_data)


    np.save(open('data/cat_dog_testdata.npy','wb'), test_data)

if __name__ == '__main__':
    create_image_traindata()
    create_image_testdata()