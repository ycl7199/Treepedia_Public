# This program is used to calculate the green view index based on the collecte metadata. The
# Object based images classification algorithm is used to classify the greenery from the GSV imgs
# in this code, the meanshift algorithm implemented by pymeanshift was used to segment image
# first, based on the segmented image, we further use the Otsu's method to find threshold from
# ExG image to extract the greenery pixels.

# For more details about the object based image classification algorithm
# check: Li et al., 2016, Who lives in greener neighborhoods? the distribution of street greenery and it association with residents' socioeconomic conditions in Hartford, Connectictu, USA

# This program implementing OTSU algorithm to chose the threshold automatically
# For more details about the OTSU algorithm and python implmentation
# cite: http://docs.opencv.org/trunk/doc/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html


# Copyright(C) Xiaojiang Li, Ian Seiferling, Marwa Abdulhai, Senseable City Lab, MIT 
# First version June 18, 2014

import time
from PIL import Image
import numpy as np
import requests
import sys
import traceback
import test
from urllib.parse import urlencode

import csv
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
    img = image.imread(img)
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
    print('lebel output = ', label_output)
    return label_output


# using 18 directions is too time consuming, therefore, here I only use 6 horizontal directions
# Each time the function will read a text, with 1000 records, and save the result as a single TXT
def GreenViewComputing_ogr_6Horizon(GSVinfoFolder, outTXTRoot, greenmonth, key_file, semsegPath):
    

    '''
    This function is used to download the GSV from the information provide
    by the gsv info txt, and save the result to a shapefile

    Required modules: numpy, requests, and PIL

        GSVinfoTxt: the input folder name of GSV info txt
        outTXTRoot: the output folder to store result green result in txt files
        greenmonth: a list of the green season, for example in Boston, greenmonth = ['05','06','07','08','09']
        key_file: the API keys in txt file, each key is one row, I prepared five keys, you can replace by your owne keys if you have Google Account
        semsegPath: the path to the cloned repository from GitHub https://github.com/CSAILVision/semantic-segmentation-pytorch.git

    last modified by Yuki Minami, 23 October 2020

    '''


    
    # read the Google Street View API key files, you can also replace these keys by your own
    lines = open(key_file,"r")
    keylist = []
    for line in lines:
        key = line.rstrip()
        keylist.append(key)
    lines.close()
    
    print ('The key list is:=============', keylist)
    
    # set a series of heading angle
    # headingArr = 360/6*np.array([0,1,2,3,4,5])
    headingArr = [0,1,2,3]
    # headingArr = ['U','D','F','B','L','R']
    # number of GSV images for Green View calculation, in my original Green View View paper, I used 18 images, in this case, 6 images at different horizontal directions should be good.
    numGSVImg = len(headingArr)*1.0
    pitch = 0
    
    # load model from URL
    model_name = 'deeplab_resnest269_ade' # pretrain model name
    label_file_name ='D:\\Treepedia\\Treepedia_Public-master_36\\Treepedia\\ade20k_class_label.txt' # label txt file name
    model, labels = load_model(model_name, label_file_name)
    print('model load success')
    csv_output = open('test.csv', 'a+', newline= '',encoding='utf-8-sig')
    csv_writer = csv.writer(csv_output)
    attributes = ['panoID', 'panoDate', 'longitude', 'latitude']
    attributes = attributes + labels
    csv_writer.writerow(attributes)

    # create a folder for GSV images and grenView Info
    if not os.path.exists(outTXTRoot):
        os.makedirs(outTXTRoot)
    # the input GSV info should be in a folder
    if not os.path.isdir(GSVinfoFolder):
        print('You should input a folder for GSV metadata')
        return
    else:
        allTxtFiles = os.listdir(GSVinfoFolder)
        for txtfile in allTxtFiles:
            if not txtfile.endswith('.txt'):
                continue
            
            txtfilename = os.path.join(GSVinfoFolder,txtfile)
            panoIDLst, panoDateLst, panoLonLst, panoLatLst = get_pano_lists_from_file(txtfilename, greenmonth)
            
            # the output text file to store the green view and pano info
            gvTxt = 'GV_'+os.path.basename(txtfile)
            GreenViewTxtFile = os.path.join(outTXTRoot,gvTxt)
            
            
            # check whether the file already generated, if yes, skip. Therefore, you can run several process at same time using this code.
            print("Processing", GreenViewTxtFile)
            if os.path.exists(GreenViewTxtFile):
                print("File already exists")
                continue
            
            # write the green view and pano info to txt            
            with open(GreenViewTxtFile,"w") as gvResTxt:
                
                for i in range(len(panoIDLst)):
                    panoDate = panoDateLst[i]
                    panoID = panoIDLst[i]
                    lat = panoLatLst[i]
                    lon = panoLonLst[i]
                    
                    # get a different key from the key list each time
                    idx = i % len(keylist)
                    key = keylist[idx]
                    
                    # calculate the green view index
                    greenPercent = 0.0
                    for heading in headingArr:
                        print("Heading is: ",heading)
                        
                        # using different keys for different process, each key can only request 25,000 imgs every 24 hours
                        URL = get_api_url(panoID, heading, pitch, key)
                        # let the code to pause by 1s, in order to not go over data limitation of Google quota
                        time.sleep(1)
                        
                        # classify the GSV images and calcuate the GVI
                        try:
                            # im = get_api_image(URL, panoID, heading) # get image from gcp server
                            im = get_local_image(panoID, 'svd360', heading) # get image from local
                            label_percent= run_segmentation(im, model)
                            data2csv = [panoID, panoDate, lon, lat] + list(label_percent)
                            greenPercent += label_percent[4] # greenView label
                            csv_writer.writerow(data2csv)
                            # greenPercent = greenPercent + percent


                        # if the GSV images are not download successfully or failed to run, then return a null value
                        except Exception as e:
                            #    print(e)
                            error_class = e.__class__.__name__ #取得錯誤類型
                            detail = e.args[0] #取得詳細內容
                            cl, exc, tb = sys.exc_info() #取得Call Stack
                            lastCallStack = traceback.extract_tb(tb)[-1] #取得Call Stack的最後一筆資料
                            fileName = lastCallStack[0] #取得發生的檔案名稱
                            lineNum = lastCallStack[1] #取得發生的行號
                            funcName = lastCallStack[2] #取得發生的函數名稱
                            errMsg = "File \"{}\", line {}, in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)
                            print(errMsg)
                            greenPercent = -1000
                            break

                    # calculate the green view index by averaging six percents from six images
                    # greenViewVal = greenPercent/numGSVImg
                    # greenViewVal = greenPercent/6 # ori
                    greenViewVal = greenPercent/4
                    print('The greenview(Tree): %s, pano: %s, (%s, %s)'%(greenViewVal, panoID, lat, lon))

                    # write the result and the pano info to the result txt file
                    lineTxt = 'panoID: %s panoDate: %s longitude: %s latitude: %s, greenview: %s\n'%(panoID, panoDate, lon, lat, greenViewVal)
                    gvResTxt.write(lineTxt)
    csv_output.close()


def get_api_url(panoID, heading, pitch, key):
    params = {
        "size": "400x400",
        "pano": panoID,
        "fov": 60,
        "heading": heading,
        "pitch": pitch,
        "sensor": "false",
        "key": key,
        "source": "outdoor"
    }
    URL = "http://maps.googleapis.com/maps/api/streetview?" + urlencode(params)
    return URL


def get_local_image(pano_id, folder_path, heading = None, with_heading = False):
    img_name = pano_id + '.jpg'
    if with_heading:
        dir_list = ['F','R','B','L']
        img_name = pano_id +'_'+ dir_list[heading] + '.jpg'
    img_path = os.path.join(folder_path, img_name)
    print(img_name,'load succeed')
    return img_path


def get_api_image(url, pano_id, heading):
    import urllib
    # response = requests.get(url, stream=True)
    # im = np.array(Image.open(response.raw))
    # print('url =', url)
    imgFd = urllib.request.urlopen(url)
    print(type(imgFd))
    print(imgFd)
    # im = np.array(Image.open(imgFd))
    im = Image.open(imgFd)
    img_path = 'GSV_image\\' + str(pano_id) + str(heading) + '.jpg'
    im.save(img_path)
    return img_path


def get_pano_lists_from_file(txtfilename, greenmonth):
    lines = open(txtfilename,"r")

    # create empty lists, to store the information of panos,and remove duplicates
    panoIDLst = []
    panoDateLst = []
    panoLonLst = []
    panoLatLst = []
    
    # loop all lines in the txt files
    for line in lines:
        metadata = line.split(" ")
        panoID = metadata[1]
        panoDate = metadata[3]
        month = panoDate[-2:]
        lon = metadata[5]
        lat = metadata[7][:-1]

        # in case, the longitude and latitude are invalide
        if len(lon)<3:
            continue
        
        # only use the months of green seasons
        if month not in greenmonth:
            continue
        if panoID in panoIDLst:
            continue
        else:
            panoIDLst.append(panoID)
            panoDateLst.append(panoDate)
            panoLonLst.append(lon)
            panoLatLst.append(lat)

    lines.close()

    return panoIDLst, panoDateLst, panoLonLst, panoLatLst    


# ------------------------------Main function-------------------------------
if __name__ == "__main__":
    
    import os,os.path
    
    os.chdir("shp_TW")
    root = os.getcwd()
    GSVinfoRoot = os.path.join(root, "metadata")
    outputTextPath = os.path.join(root, "greenViewRes")
    greenmonth = ['01','02','03','04','05','06','07','08','09','10','11','12']

    os.chdir("../")
    key_file = os.path.join(os.getcwd(), 'keys.txt')

    os.chdir('../')
    semsegPath = os.path.join(os.getcwd(), 'semantic-segmentation-pytorch')
    os.chdir(os.path.join(os.getcwd(), 'Treepedia_Public-master_36'))


    
    GreenViewComputing_ogr_6Horizon(GSVinfoRoot,outputTextPath, greenmonth, key_file, semsegPath)


