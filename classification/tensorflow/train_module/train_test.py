import os, cv2, datetime, pathlib, random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from aug_test import * 


class Train():
    def __init__(self, project_name, auged_path, img_size, epoch, batch_size):
        self.project_name = project_name
        self.auged_data_path = auged_path + f'/{project_name}'
        self.img_size = img_size
        self.epoch = epoch
        self.strategy = self.gpu_setup()
        self.batch_size = batch_size * self.strategy.num_replicas_in_sync
        self.train_path = self.auged_data_path + '/train'
        self.valid_path = self.auged_data_path + '/valid'
        self.label_cnt = len(os.listdir(self.train_path))
        #self.label_cnt = 2
        #print('label_cnt : ', self.label_cnt)
        self.AUTO = tf.data.experimental.AUTOTUNE

    def gpu_setup(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        
        if len(gpus) > 1:
            try:
                print('Activate Multi GPU')
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                strategty = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
            except RuntimeError as e:
                print(e)

        else:
            try:
                print('Activate Single GPU')
                tf.config.experimental.set_memory_growth(gpus[0], True)
                strategy = tf.distribute.experimental.CentralStorageStrategy()
            except RuntimeError as e:
                print(e)
        return strategy

    def preprocess_image(self, images, label=None):
        image = tf.io.read_file(images)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [self.img_size, self.img_size])
        image = tf.keras.applications.efficientnet.preprocess_input(image)

        if label is None:
            return image

        else:
            return image, label

 
    def make_tf_dataset(self, ds_path, is_train):
        ds_path = pathlib.Path(ds_path)

        images = list(ds_path.glob('*/*'))
        images = [str(path) for path in images]
        total_images = len(images)

        if is_train:
            random.shuffle(images)

        labels = sorted(item.name for item in ds_path.glob('*/') if item.is_dir())
        print('------------------',labels)
        classes = labels
        labels = dict((name, index) for index, name in enumerate(labels))
        labels = [labels[pathlib.Path(path).parent.name] for path in images]
        labels = tf.keras.utils.to_categorical(labels, num_classes=len(classes), dtype='float32')

        if is_train:
            dataset = (tf.data.Dataset
                        .from_tensor_slices((images, labels))
                        .map(self.preprocess_image, num_parallel_calls=self.AUTO)
                        .repeat()
                        .shuffle(512)
                        .batch(self.batch_size)
                        .prefetch(self.AUTO)
            )

        else:
            dataset = (tf.data.Dataset
                        .from_tensor_slices((images, labels))
                        .map(self.preprocess_image, num_parallel_calls=self.AUTO)
                        .repeat()
                        .batch(self.batch_size)
                        .prefetch(self.AUTO)
            )

        return dataset, total_images, classes

    def build_lrfn(self, lr_start=0.00001, lr_max=0.00005, 
                lr_min=0.00001, lr_rampup_epochs=5, 
                lr_sustain_epochs=0, lr_exp_decay=.8):
        lr_max = lr_max * self.strategy.num_replicas_in_sync

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

    def get_model(self):
        with self.strategy.scope():
            base_model = tf.keras.applications.EfficientNetB4(input_shape=(self.img_size, self.img_size, 3),
                                        weights="imagenet", # noisy-student
                                        include_top=False)
            for layer in base_model.layers:
                layer.trainable = True
                
            avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
            output = tf.keras.layers.Dense(self.label_cnt, activation="softmax")(avg)
            model = tf.keras.Model(inputs=base_model.input, outputs=output)

        model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
        model.summary()
        return model

    def model_train(self):
        train_dataset, train_total, train_classes = self.make_tf_dataset(self.train_path, True)
        valid_dataset, valid_total, valid_classes = self.make_tf_dataset(self.valid_path, False)

        TRAIN_STEPS_PER_EPOCH = int(tf.math.ceil(train_total/ self.batch_size).numpy())
        VALID_STEP_PER_EPOCH = int(tf.math.ceil(valid_total / self.batch_size).numpy())

        print(len(train_classes), len(valid_classes))

        # Learning Rate Scheduler setup
        lrfn = self.build_lrfn()
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)

        # Checkpoint callback setup
        SAVED_PATH = f'/home/perth/Desktop/personal_project/2.test/train_test/data/model/{self.project_name}'
        LOG_TIME = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M")
        WEIGHT_FNAME = '{epoch:02d}-{val_categorical_accuracy:.2f}.hdf5'
        checkpoint_path = f'/{SAVED_PATH}/{LOG_TIME}/{WEIGHT_FNAME}'

        if not(os.path.isdir(f'/{SAVED_PATH}/{LOG_TIME}')):
            os.makedirs(f'/{SAVED_PATH}/{LOG_TIME}')
            f = open(f'{SAVED_PATH}/{LOG_TIME}/main_labels.txt', 'w')

            for label in train_classes:
                f.write(f'{label}\n')
            
            f.close()

        checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        monitor='val_categorical_accuracy',
                                                        save_best_only=True,
                                                        mode='max')
        earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

        model = self.get_model()    
        history = model.fit(train_dataset,
                            epochs = self.epoch,
                            callbacks=[lr_schedule, checkpointer, earlystopper],
                            steps_per_epoch=TRAIN_STEPS_PER_EPOCH,
                            verbose=1,
                            validation_data=valid_dataset,
                            validation_steps=VALID_STEP_PER_EPOCH)

        model.save(f'{SAVED_PATH}/{LOG_TIME}/main_model.h5')
        model.save(f'{SAVED_PATH}/{LOG_TIME}/pb_model', save_format='tf')

        #display_training_curves(history)


# train = Train(project_name = 'cu',
#             auged_path = '/home/perth/Desktop/personal_project/2.test/train_test/data/output',
#             img_size = 224, 
#             epoch = 10, 
#             batch_size = 16)



# train.model_train()









