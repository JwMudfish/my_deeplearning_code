import tensorflow as tf
from tensorflow import keras
import pathlib
import random
import os
import datetime
import time
from efficientnet.tfkeras import EfficientNetB4, preprocess_input
import pandas as pd
from glob import glob
#import matplotlib.pyplot as plt
import cv2

from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D, ReLU, MaxPooling2D, Dense, BatchNormalization, GlobalAveragePooling2D

#from tensorflow.keras.utils import multi_gpu_model

from sklearn.model_selection import StratifiedKFold


# EfficientNetB0 	224
# EfficientNetB1 	240
# EfficientNetB2 	260
# EfficientNetB3 	300
# EfficientNetB4 	380
# EfficientNetB5 	456
# EfficientNetB6 	528
# EfficientNetB7 	600



BATCH_SIZE = 8
IMG_SIZE = 380
NUM_EPOCHS = 100
EARLY_STOP_PATIENCE = 4

valid_set_num = 4

img_dir = './landmark_dataset/train/'

dataset_name = 'landmark_efnb4_380_cv4_cm03_v3'
saved_path = './model/'


def basic_processing(img_list, is_training):
    #img_path = pathlib.Path(img_path)

    #images = list(img_path.glob('*/*'))
    images = img_list
    images = [str(path) for path in images]
    len_images = len(images)

    if is_training:
        random.shuffle(images)

    #labels = sorted(item.name for item in img_path.glob('*/') if item.is_dir())
    labels = sorted(label_list)
    
    labels_len = len(labels)
    labels = dict((name, index) for index, name in enumerate(labels))
    labels = [labels[pathlib.Path(path).parent.name] for path in images]
    labels = tf.keras.utils.to_categorical(labels, num_classes=labels_len, dtype='float32')

    return images, labels, len_images, labels_len

def preprocess_image(image):

    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])  #  antialias = True 
    image = preprocess_input(image)
    return image

# 이미지 path -> tensor
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)    
    return preprocess_image(image)

# tf dataset 만들기
def make_tf_dataset(images, labels):
    image_ds = tf.data.Dataset.from_tensor_slices(images)
    image_ds = image_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

    lable_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.float32))

    image_label_ds = tf.data.Dataset.zip((image_ds, lable_ds))

    return image_label_ds


def cutmix(images, labels, PROB = 0.3):
    imgs = []
    labs = []
    
    for i in range(BATCH_SIZE):
        APPLY = tf.cast(tf.random.uniform(()) <= PROB, tf.int32)
        idx = tf.random.uniform((), 0, BATCH_SIZE, tf.int32)
        
        W = IMG_SIZE
        H = IMG_SIZE
        lam = tf.random.uniform(())
        
        cut_ratio = tf.math.sqrt(1.-lam)
        cut_w = tf.cast(W * cut_ratio, tf.int32) * APPLY
        cut_h = tf.cast(H * cut_ratio, tf.int32) * APPLY
        
        cx = tf.random.uniform((), int(W/8), int(7/8*W), tf.int32)
        cy = tf.random.uniform((), int(H/8), int(7/8*H), tf.int32)
        
        xmin = tf.clip_by_value(cx - cut_w//2, 0, W)   # clip_by_value 값의 상한 하한 설정,
        ymin = tf.clip_by_value(cy - cut_h//2, 0, H)
        xmax = tf.clip_by_value(cx + cut_w//2, 0, W)   # clip_by_value 값의 상한 하한 설정,
        ymax = tf.clip_by_value(cy + cut_h//2, 0, H)
        
        mid_left = images[i, ymin:ymax, :xmin, :]
        mid_mid = images[idx, ymin:ymax, xmin:xmax, :]
        mid_right = images[i, ymin:ymax, xmax:, :]
        middle = tf.concat([mid_left, mid_mid, mid_right], axis=1)
        
        top = images[i, :ymin, :, :]
        bottom = images[i, ymax:, :, :]
        new_img = tf.concat([top, middle, bottom], axis = 0)
        imgs.append(new_img)
        
        alpha = tf.cast((cut_w*cut_h) / (W*H), tf.float32)
        label1 = labels[i]
        label2 = labels[idx]
        
        new_label = ((1-alpha) * label1 + alpha * label2)
        labs.append(new_label)
        
    new_imgs = tf.reshape(tf.stack(imgs), [-1, IMG_SIZE, IMG_SIZE, 3])
    new_labs = tf.reshape(tf.stack(labs), [-1, train_labels_len])
    
    return new_imgs, new_labs


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # 텐서플로가 첫 번째 GPU에 1GB 메모리만 할당하도록 제한
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10024)])
  except RuntimeError as e:
    # 프로그램 시작시에 가상 장치가 설정되어야만 합니다
    print(e)

# result = []
# idx = 0

label_list = [f for f in os.listdir(img_dir) if not f.startswith('.')]

# for label in label_list:
#     if label[-1] == ']':
#         label = label.replace('[', '[[]')
#         find_index = label.rfind(']')
#         label = label[:find_index] + '[]]' + label[find_index+1:]

#     file_list = glob(os.path.join('./landmark_dataset/train',label,'*'))
#     for file in file_list:
#         result.append([idx, label, file])
#         idx += 1
        
# img_df = pd.DataFrame(result, columns=['idx','label','image_path'])

# X = img_df[['idx','label']].values[:,0]
# y = img_df[['idx','label']].values[:,1:]

# img_df['fold'] = -1

# skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

# for i, (trn_idx, vld_idx) in enumerate(skf.split(X,y)):
#     img_df.loc[vld_idx, 'fold'] = i
    
# img_df.to_csv('dataset.csv', index = False)


img_df = pd.read_csv('./dataset.csv')

trn_fold = [i for i in range(10) if i not in [valid_set_num]]
vld_fold = [valid_set_num]

trn_idx = img_df.loc[img_df['fold'].isin(trn_fold)].index
vld_idx = img_df.loc[img_df['fold'].isin(vld_fold)].index


#######################################


AUTOTUNE = tf.data.experimental.AUTOTUNE
strategy = tf.distribute.experimental.CentralStorageStrategy()
#train_dataset_path = '../datasets/emart24/cls_train/train'
#valid_dataset_path = '../datasets/emart24/cls_train/validation'

#TRAIN_STEP_PER_EPOCH = tf.math.ceil(train_images_len / BATCH_SIZE).numpy()
#VALID_STEP_PER_EPOCH = tf.math.ceil(valid_images_len / BATCH_SIZE).numpy()

time = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M") + '_tf2'
weight_file_name = '{epoch:02d}.hdf5'
checkpoint_path = saved_path + dataset_name + '/' + time + '/' + weight_file_name


if not(os.path.isdir(saved_path + dataset_name + '/' + time)):
    os.makedirs(os.path.join(saved_path + dataset_name + '/' + time))
else:
    pass


trn_img_list = list(img_df.loc[img_df['idx'].isin(trn_idx)]['image_path'])
vld_img_list = list(img_df.loc[img_df['idx'].isin(vld_idx)]['image_path'])

train_images, train_labels, train_images_len, train_labels_len = basic_processing(trn_img_list, True)
valid_images, valid_labels, valid_images_len, valid_labels_len = basic_processing(vld_img_list, False)

TRAIN_STEP_PER_EPOCH = tf.math.ceil(train_images_len / BATCH_SIZE).numpy()
VALID_STEP_PER_EPOCH = tf.math.ceil(valid_images_len / BATCH_SIZE).numpy()

# 기본 Dataset 만들기
train_ds = make_tf_dataset(train_images, train_labels)
train_ds = train_ds.repeat().batch(BATCH_SIZE).map(cutmix).prefetch(tf.data.experimental.AUTOTUNE) # tf.data.experimental.AUTOTUNE


valid_ds = make_tf_dataset(valid_images, valid_labels)
valid_ds = valid_ds.repeat().batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)


base_model = EfficientNetB4(weights='imagenet', 
                            include_top=False,
                            input_shape=(IMG_SIZE, IMG_SIZE, 3))

def create_model():
    model = models.Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dense(train_labels_len, activation='softmax'))
    return model

model = create_model()
for layer in model.layers:
    layer.trainable = True


def build_lrfn(lr_start=0.00001, lr_max=0.00005, 
               lr_min=0.00001, lr_rampup_epochs=5, 
               lr_sustain_epochs=0, lr_exp_decay=.8):
    lr_max = lr_max * strategy.num_replicas_in_sync

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) *\
                 lr_exp_decay**(epoch - lr_rampup_epochs\
                                - lr_sustain_epochs) + lr_min
        return lr
    return lrfn

#optimizer = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#optimizer = tf.keras.optimizers.Adam(LR_INIT)


#model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),    # label_smoothing=0.1
              optimizer = 'adam',
              metrics=["accuracy"])


lr_shedule_fn = build_lrfn()
lr_callback = keras.callbacks.LearningRateScheduler(lr_shedule_fn, verbose=1)

cb_early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
cb_checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     monitor='val_loss',
                                                     save_best_only=True,
                                                     mode='auto')

logger = keras.callbacks.CSVLogger(f'./logs/{dataset_name}_log.csv', separator=',', append=False)


history = model.fit(train_ds,
                    epochs=NUM_EPOCHS,
                    steps_per_epoch=TRAIN_STEP_PER_EPOCH,
                    shuffle=False,
                    validation_data=valid_ds,
                    validation_steps=VALID_STEP_PER_EPOCH,
                    verbose=1,
                    callbacks=[lr_callback, cb_checkpointer, cb_early_stopper, logger])  #cb_checkpointer, cb_early_stopper

model.save(saved_path + dataset_name  + '/' + dataset_name + '.h5')