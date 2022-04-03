import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
from gluoncv.utils.viz import get_color_pallete
from gluoncv.data.transforms.presets.segmentation import test_transform
import gluoncv
import time
from matplotlib import pyplot as plt
import os
import numpy as np
import glob
import matplotlib.image as mpimg

def load_model(model_name, label_file_name, ):
    ctx = mx.cpu(0) # using cpu
    model = gluoncv.model_zoo.get_model(model_name, pretrained=True, ctx = mx.cpu(0)) # load model from gluoncv
    with open(label_file_name, encoding = 'utf-8') as h:
        lines = h.readlines()
        labels = [label.rstrip() for label in lines]
    return model, labels

def run_segmentation(img, model, label_num = 150):
    # ctx = mx.gpu(0) # device setting
    ctx = mx.cpu(0) # using cpu
    img = test_transform(img, ctx) # change image(array) to tensor
    model_output = model.predict(img)
    predict = mx.nd.squeeze(mx.nd.argmax(model_output, 1)).asnumpy()  # transfer model output to ndarray 

    img_shape =predict.shape
    pixel_num = img_shape[0]*img_shape[1]
    label_output=np.zeros([label_num])
    for i in range(label_num):
        label_percentage = round( (np.count_nonzero(predict == i))/pixel_num, 3)
        label_output[i] = label_percentage
    
    return label_output

model_name = 'deeplab_resnest269_ade' # pretrain model name
label_file_name ='ade20k_class_label.txt' # label txt file name
img_name = 'C_N010_1_gc1.6_Hsu_2016.jpg'
img =  image.imread(img_name)
model, label_name = load_model(model_name, label_file_name)
label_output = run_segmentation(img, model)
print('label_name = ', label_name)
print(label_output)

