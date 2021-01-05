# import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from xml.dom import minidom
from datetime import datetime
import glob
import os
import sys
import time
# from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import tensorflow as tf
from collections import Counter


MODE = 'dual'  # single or dual

# get file path
project_path = '/home/perth/Desktop/personal_project/0.datasets/test'

# xml
right_main_xml = project_path + "/box/r/main.xml"
right_empty_xml = project_path + "/box/r/empty.xml"

left_main_xml = project_path + "/box/l/main.xml"
left_empty_xml = project_path + "/box/l/empty.xml"

# label
main_label_file = project_path + "/model_and_label/main_labels.txt"
binary_label_file = project_path + "/model_and_label/empty_labels.txt"

# model
main_model_path = project_path + "/model_and_label/main_model.h5"
empty_model_path = project_path + '/model_and_label/empty_model.h5'

###########################################################################
# get images
left_test_images = sorted(glob.glob(project_path + '/data/left/*.jpg'))
right_test_images = sorted(glob.glob(project_path + '/data/right/*.jpg'))

############################################################################

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     for gpu in gpus:
#         tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)])
#   except RuntimeError as e:
#     print(e)

def init_model(test_img, main_model, empty_model):
    img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    empty_model.predict(img)
    main_model.predict(img)


def crop_image(image, boxes, resize=None, save_path=None):

    images = list(map(lambda b : image[b[1]:b[3], b[0]:b[2]], boxes)) 

    if str(type(resize)) == "<class 'tuple'>":
        try:
            images = list(map(lambda i: preprocess_input(cv2.resize(i, dsize=resize, interpolation=cv2.INTER_LINEAR)), images))
        except Exception as e:
            print(str(e))
    return images

def get_boxes(label_path):
    xml_path = os.path.join(label_path)
    root_1 = minidom.parse(xml_path)
    bnd_1 = root_1.getElementsByTagName('bndbox')
    result = []
    for i in range(len(bnd_1)):
        xmin = int(bnd_1[i].childNodes[1].childNodes[0].nodeValue)
        ymin = int(bnd_1[i].childNodes[3].childNodes[0].nodeValue)
        xmax = int(bnd_1[i].childNodes[5].childNodes[0].nodeValue)
        ymax = int(bnd_1[i].childNodes[7].childNodes[0].nodeValue)
        result.append((xmin,ymin,xmax,ymax))
    
    return result


def merge_left_right(left_images, right_images):
    total_images = []
    for left_frame, right_frame in zip(left_test_images, right_test_images):
      
                left_frame = cv2.imread(left_frame)
                left_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB)
                right_frame = cv2.imread(right_frame)
                right_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB)

                total_images.append((left_frame, right_frame))
    return total_images

def make_batch(total_images, main_or_empty): 
    if main_or_empty == 'main':
        r_boxes = rm_boxes
        l_boxes = lm_boxes
    
    else:
        r_boxes = rem_boxes
        l_boxes = lem_boxes

    #total_images = merge_left_right(left_images = left_test_images, right_images = right_test_images)
    main_final = []
    for i in range(len(total_images)):
        right_main_crops = crop_image(total_images[i][1], r_boxes, (224, 224))
        left_main_crops = crop_image(total_images[i][0], l_boxes, (224, 224))
        main_merge = left_main_crops + right_main_crops
        main_final.extend(main_merge)  #main_final = main_final + main_merge
        print(len(main_final))
    main_final = np.array(main_final)
    return main_final

def make_single_batch(image_path):
    total_images = sorted(glob.glob(os.path.join(image_path,'*.jpg')))
    main_final = []
    for img in total_images:
        frame = cv2.imread(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = preprocess_input(cv2.resize(frame, dsize=(224,224), interpolation=cv2.INTER_LINEAR))
        #main_final = main_final + frame
        main_final.append(frame)
        print(len(main_final))
    main_final = np.array(main_final)
    return main_final

def inference(main_model, empty_model):
    #empty_pred = empty_model.predict(empty_final)
    main_pred = main_model.predict(main_final)
    empty_pred = empty_model.predict(empty_final)
    result = []
    for i in range(len(empty_pred)):
        em_res = EM_CLASS_NAMES[np.argmax(empty_pred[i])]
        if em_res == "empty":
            result.append(em_res)
        else:
            main_res = CLASS_NAMES[np.argmax(main_pred[i])]
            result.append(main_res)
    return result

def get_label(label_path):
    df = pd.read_csv(main_label_file, sep = ' ', index_col=False, header=None)
    class_name = df[0].tolist()
    class_name = sorted(class_name)
    return class_name

# get label name
EM_CLASS_NAMES = get_label(binary_label_file)
CLASS_NAMES = get_label(main_label_file)

# get box corr
rm_boxes = get_boxes(right_main_xml)
rem_boxes = get_boxes(right_empty_xml)
lm_boxes = get_boxes(left_main_xml)
lem_boxes = get_boxes(left_empty_xml)


# main_total_images = merge_left_right(left_images = left_test_images, right_images = right_test_images)
# main_final = make_batch(total_images = main_total_images)

if MODE == 'dual':

    # 모델 불러오기
    main_model = tf.keras.models.load_model(main_model_path)
    empty_model = tf.keras.models.load_model(empty_model_path)

    # 더미 데이터로 모델 한번 돌려놓기!
    init_model(np.zeros((224, 224, 3), np.uint8), main_model, empty_model)

    main_total_images = merge_left_right(left_images = left_test_images, right_images = right_test_images)

    main_final = make_batch(total_images = main_total_images, main_or_empty = 'main')
    empty_final = make_batch(total_images = main_total_images, main_or_empty = 'empty')
    
    os.system('clear')
    
    rst = inference(main_model = main_model, empty_model = empty_model)

    #print(rst)
    idx = 0
    while idx < len(rst):
        print(rst[idx : idx + 7])
        idx += 7

#### single mode #############################################
elif MODE == 'single':
    pog = 'watermelon'
    main_final = make_single_batch(f'{project_path}/data/single/{pog}')
    print(main_final.shape)
    # 모델 불러오기
    main_model = tf.keras.models.load_model(main_model_path)
    
    main_pred = main_model.predict(main_final)

    os.system('clear')
    rst = []
    for i in main_pred:
        rst.append(CLASS_NAMES[np.argmax(i)])

    print(Counter(rst))

############################################################





# cv2.imshow('ddd',main_final[0])
# cv2.imshow('ddda',cv2.cvtColor(main_final[0], cv2.COLOR_BGR2RGB))
# cv2.waitKey()
# cv2.destroyAllWindows()



