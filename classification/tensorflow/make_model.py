import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D, ReLU, MaxPooling2D, Dense, BatchNormalization, GlobalAveragePooling2D
from keras_adabound import AdaBound


# EfficientNetB0 	224
# EfficientNetB1 	240
# EfficientNetB2 	260
# EfficientNetB3 	300
# EfficientNetB4 	380
# EfficientNetB5 	456
# EfficientNetB6 	528
# EfficientNetB7 	600


class MainModel:
    def __init__(self, model_name ,img_size, label_count, pretrained_weight='imagenet'):
        self.model_name = model_name
        self.img_size = img_size
        self.pretrained_weight = pretrained_weight   #'imagenet', 'noisy-student'
        self.label_count = label_count

    def make_backbone(self):
        if self.model_name == 'efn0':
            #from efficientnet.tfkeras import EfficientNetB0
            base_model = tf.keras.applications.EfficientNetB0(weights=self.pretrained_weight,
                                        include_top=False,
                                        input_shape=(self.img_size, self.img_size, 3))

        elif self.model_name == 'efn1':
            #from efficientnet.tfkeras import EfficientNetB1
            base_model = tf.keras.applications.EfficientNetB1(weights=self.pretrained_weight,
                                        include_top=False,
                                        input_shape=(self.img_size, self.img_size, 3))

        elif self.model_name == 'efn2':
            #from efficientnet.tfkeras import EfficientNetB2
            base_model = tf.keras.applications.EfficientNetB2(weights=self.pretrained_weight,
                                        include_top=False,
                                        input_shape=(self.img_size, self.img_size, 3))
    
        elif self.model_name == 'efn3':
            #from efficientnet.tfkeras import EfficientNetB3
            base_model = tf.keras.applications.EfficientNetB3(weights=self.pretrained_weight,
                                        include_top=False,
                                        input_shape=(self.img_size, self.img_size, 3))

        elif self.model_name == 'efn4':
            #from efficientnet.tfkeras import EfficientNetB4
            base_model = tf.keras.applications.EfficientNetB4(weights=self.pretrained_weight,
                                        include_top=False,
                                        input_shape=(self.img_size, self.img_size, 3))

        elif self.model_name == 'efn5':
            #from efficientnet.tfkeras import EfficientNetB5
            base_model = tf.keras.applications.EfficientNetB5(weights=self.pretrained_weight,
                                        include_top=False,
                                        input_shape=(self.img_size, self.img_size, 3))

        return base_model

    def make_model(self, opt):
        model = models.Sequential(name = self.model_name)
        model.add(self.make_backbone())
        
        if opt == 1:
            print('model option 1 !!!!')
            model.add(GlobalAveragePooling2D())
            model.add(Dense(256))
            model.add(BatchNormalization())
            model.add(ReLU())
            model.add(Dense(self.label_count, activation='softmax'))
        elif opt == 2:
            print('model option 2 !!!!')
            model.add(GlobalAveragePooling2D())
            model.add(Dense(self.label_count, activation='softmax'))
        
        
        for layer in model.layers:
            layer.trainable = True
            
        print(model.summary())
        return model