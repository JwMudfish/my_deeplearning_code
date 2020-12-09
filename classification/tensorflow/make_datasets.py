from sklearn.model_selection import StratifiedKFold
#from efficientnet.tfkeras import preprocess_input
import pandas as pd
import os
from glob import glob
import tensorflow as tf  #tf 2.3
from tensorflow import keras
import pathlib
import random
import numpy as np


class MakeDataSetInfo:

    def __init__(self, train_data_path, save_path, label_path, dataset_name):
        self.train_data_path = train_data_path
        self.save_path = save_path
        self.dataset_name = dataset_name
        self.label_path = label_path

    def make_folder(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def make_dataframe(self, label_save=False):
        print('make_dataframe')
        label_list = sorted([folder for folder in os.listdir(self.train_data_path) if not folder.startswith('.')])
        
        if label_save == True:
            label_txt = pd.DataFrame(label_list)
            self.make_folder(self.label_path)
            label_txt.to_csv(f'./{self.label_path}/{self.dataset_name}_label.txt', index=False, header=False)
            print(f'./{self.label_path}/{self.dataset_name}_label.txt  saved!!!')

        else:
            print('label.txt file is not saved.')
        
        idx = 0
        result = []
        
        for label in label_list:
            if label[-1] == ']':
                label = label.replace('[', '[[]')
                find_index = label.rfind(']')
                label = label[:find_index] + '[]]' + label[find_index+1:]

            file_list = glob(os.path.join(self.train_data_path , label, '*'))
            
            for file_path in file_list:
                result.append([idx, label, file_path])
                idx += 1
        img_df = pd.DataFrame(result, columns=['idx','label','image_path'])
        return img_df, label_list

    def cross_validation(self, n_splits, auto_save=True):
        img_df = self.make_dataframe(label_save=True)[0]
        print(len(img_df))

        X = img_df[['idx','label']].values[:,0]
        y = img_df[['idx','label']].values[:,1:]

        img_df['fold'] = -1

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

        for i, (trn_idx, vld_idx) in enumerate(skf.split(X,y)):
            img_df.loc[vld_idx, 'fold'] = i

        if auto_save==True:
            self.make_folder(self.save_path)
            img_df.to_csv(f'{self.save_path}/{self.dataset_name}_datainfo.csv', index = False)

        return img_df


class MakeDataSet:  
    def __init__(self, dataframe, dataset_name, train_data_path, df_info_path, batch_size, valid_set_num, img_size, is_cutmix=False, prob = 0.3, read_from_csv=False):
        self.dataframe = dataframe
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.valid_set_num = valid_set_num
        self.train_data_path = train_data_path
        self.img_size = img_size
        self.is_cutmix = is_cutmix
        self.prob = prob
        self.df_info_path = df_info_path
        self.read_from_csv = read_from_csv

    def make_label_len(self):
        labels = sorted([folder for folder in os.listdir(self.train_data_path) if not folder.startswith('.')])
        label_len = len(labels)
        return label_len

    def split_dataset(self):
        if self.read_from_csv == True:
            img_df = pd.read_csv(f'{self.df_info_path}/{self.dataset_name}_datainfo.csv')
        else:
            img_df = self.dataframe

        trn_fold = [i for i in range(10) if i not in [self.valid_set_num]]
        vld_fold = [self.valid_set_num]

        trn_idx = img_df.loc[img_df['fold'].isin(trn_fold)].index
        vld_idx = img_df.loc[img_df['fold'].isin(vld_fold)].index

        trn_img_list = list(img_df.loc[img_df['idx'].isin(trn_idx)]['image_path'])
        vld_img_list = list(img_df.loc[img_df['idx'].isin(vld_idx)]['image_path'])

        print('train image :  ', len(trn_img_list))
        print('valid image :  ', len(vld_img_list))

        return trn_img_list, vld_img_list


    def basic_processing(self, img_list, is_training):
        images = img_list
        images = [str(path) for path in images]
        len_images = len(images)

        if is_training:
            random.shuffle(images)

        #labels = sorted(item.name for item in img_path.glob('*/') if item.is_dir())
        labels = sorted([folder for folder in os.listdir(self.train_data_path) if not folder.startswith('.')])
        labels_len = len(labels)
        print('basic_processing 실행중... labels_len : ', labels_len)
        labels = dict((name, index) for index, name in enumerate(labels))
        labels = [labels[pathlib.Path(path).parent.name] for path in images]
        labels = tf.keras.utils.to_categorical(labels, num_classes=labels_len, dtype='float32')

        return images, labels, len_images, labels_len

    def preprocess_image(self, image):
        #from efficientnet.tfkeras import preprocess_input
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [self.img_size, self.img_size])  #  antialias = True 
        #image = preprocess_input(image)
        image = tf.keras.applications.efficientnet.preprocess_input(image)
        return image

    # 이미지 path -> tensor
    def load_and_preprocess_image(self, path):
        image = tf.io.read_file(path)    
        return self.preprocess_image(image)

    # tf dataset 만들기
    def make_tf_dataset(self, images, labels):
        print('make_tf_dataset 실행')
        AUTOTUNE = tf.data.experimental.AUTOTUNE

        image_ds = tf.data.Dataset.from_tensor_slices(images)
        image_ds = image_ds.map(self.load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

        lable_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.float32))

        image_label_ds = tf.data.Dataset.zip((image_ds, lable_ds))

        return image_label_ds


    # def cutmix(self, images, labels, PROB = 0.3):
    def cutmix(self, images, labels):
        print('cutmix 실행')
        imgs = []
        labs = []
        
        for i in range(self.batch_size):
            APPLY = tf.cast(tf.random.uniform(()) <= self.prob, tf.int32)
            idx = tf.random.uniform((), 0, self.batch_size, tf.int32)
            
            W = self.img_size
            H = self.img_size
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
            
        new_imgs = tf.reshape(tf.stack(imgs), [-1, self.img_size, self.img_size, 3])
        new_labs = tf.reshape(tf.stack(labs), [-1, self.make_label_len()])
        
        return new_imgs, new_labs

    def final_dataset(self):
        print('...')
        trn_img_list = self.split_dataset()[0]
        vld_img_list = self.split_dataset()[1]

        train_images, train_labels, train_images_len, train_labels_len = self.basic_processing(trn_img_list, True)
        valid_images, valid_labels, valid_images_len, valid_labels_len = self.basic_processing(vld_img_list, False)

        #TRAIN_STEP_PER_EPOCH = tf.math.ceil(train_images_len / self.batch_size).numpy()
        #VALID_STEP_PER_EPOCH = tf.math.ceil(valid_images_len / self.batch_size).numpy()

        AUTOTUNE = tf.data.experimental.AUTOTUNE  #############
        
        if self.is_cutmix == True:
            print('cutmix 적용')
            train_ds = self.make_tf_dataset(train_images, train_labels)
            train_ds = train_ds.repeat().batch(self.batch_size).map(self.cutmix).prefetch(AUTOTUNE) # tf.data.experimental.AUTOTUNE

            valid_ds = self.make_tf_dataset(valid_images, valid_labels)
            valid_ds = valid_ds.repeat().batch(self.batch_size).prefetch(AUTOTUNE)
        else:
            print('cutmix 비적용')
            train_ds = self.make_tf_dataset(train_images, train_labels)
            train_ds = train_ds.repeat().batch(self.batch_size).prefetch(AUTOTUNE) # tf.data.experimental.AUTOTUNE

            valid_ds = self.make_tf_dataset(valid_images, valid_labels)
            valid_ds = valid_ds.repeat().batch(self.batch_size).prefetch(AUTOTUNE)

        return train_ds, valid_ds

'''
a = MakeDataSetInfo(train_data_path=train_data_path, 
                    save_path=save_path,
                    label_path=label_path,
                    dataset_name=dataset_name)
#print(type(a.cross_validation(n_splits=10)))

b = MakeDataSet(dataframe = a.cross_validation(n_splits=10), 
                train_data_path = train_data_path, 
                batch_size = 16, 
                valid_set_num = 5,
                img_size = 224, 
                is_cutmix=True)

tr, vl = b.final_dataset()
'''

