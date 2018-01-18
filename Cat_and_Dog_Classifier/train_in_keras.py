#-*- coding:utf-8 -*-
# create at  on YeY
from keras import applications
from keras import optimizers
from keras.layers import Dense,GlobalAveragePooling2D
from keras.models import Model
import numpy as np
import cv2
import matplotlib.pyplot as plt


Vgg16_model_notop = 'Model_save/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

def fine_tune_VGG16():
    '''
    固定VGG16前18层的参数不变，并连接两层全连接层。

    Return:
        model -- 模型对象
    '''
    VGG16_model = applications.VGG16(include_top=False,weights=Vgg16_model_notop)

    for layer in VGG16_model.layers[:18]:
         layer.trainable = False

    x = VGG16_model.output

    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)

    model = Model(inputs=VGG16_model.input,outputs=predictions)

    model.compile(optimizer=optimizers.SGD(lr=1e-4,momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train():
    '''
    训练模型

    '''
    #load Train data
    train_data = np.load(open('data/cat_dog_traindata.npy','rb'))
    X_train = np.array([i[0] for i in train_data]).reshape(-1, 224, 224, 3)
    Y_train = np.array([i[1] for i in train_data])

    #Train Model
    print('Srart Train Model ....')
    model = fine_tune_VGG16()
    model.fit(X_train,Y_train,batch_size=32,epochs=10)

    #Save Model
    print('Start Save Model ....')
    model.save('Model_save/fine_tune_Vgg16_model.h5')

    #load Test data
    test_data = np.load(open('data/cat_dog_testdata.npy','rb'))
    X_test = np.array([i[0] for i in test_data]).reshape(-1, 224, 224, 3)
    Y_test = np.array([i[1] for i in test_data])

    # Accuracy
    # The test accuracy is 0.97
    # model = load_model('Model_save/fine_tune_Vgg16_model.h5')
    print('Srart Evaluate Model ....')
    acc = model.evaluate(X_test,Y_test)
    print('The Test Loss : {} Accuracy : {}'.format(acc[0],acc[1]))



if __name__ == '__main__':
    train()



