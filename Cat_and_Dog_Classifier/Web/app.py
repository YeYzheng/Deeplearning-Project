#-*- coding:utf-8 -*-
# create at  on YeY

import os
from flask import Flask, render_template
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
from keras.models import  load_model
import cv2
import numpy as np

#模型文件地址(根据本地地址改变)
cat_dog_classify_modelpath = 'E:/project_code/Deeplearning_Project/Cat_and_Dog_Classifier/Model_save/fine_tune_Vgg16_model.h5'

def predict(img_path):
    '''
    预测
    '''
    model = load_model(cat_dog_classify_modelpath)
    image = cv2.imread(img_path,flags=cv2.IMREAD_COLOR)
    image = cv2.resize(image,(244,244))
    img = np.reshape(image,(1,image.shape[0],image.shape[1],image.shape[2]))
    pred = np.argmax(model.predict(img))
    if pred == 0:
        return 'This is a cat'
    elif pred == 1:
        return 'This is a dog'


app = Flask(__name__)
app.config['SECRET_KEY'] = 'I have a dream'
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB


class UploadForm(FlaskForm):
    photo = FileField(validators=[
        FileAllowed(photos, u'只能上传图片！'),
        FileRequired(u'文件未选择！')])
    submit = SubmitField(u'上传')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = photos.url(filename)
        predict_str = predict(os.path.join('uploads',filename))
    else:
        file_url = None
        predict_str = None
    return render_template('index.html', form=form, file_url=file_url,predict_str = predict_str)


if __name__ == '__main__':
    if not os.path.exists('uploads/'):
        os.mkdir('uploads/')
    app.run()