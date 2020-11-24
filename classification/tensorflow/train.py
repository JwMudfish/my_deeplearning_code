import tensorflow as tf
from tensorflow import keras
from make_datasets import MakeDataSetInfo, MakeDataSet
from make_model import MainModel
import os
import datetime
import time

##!@######################
# 변수설정
BATCH_SIZE = 8
IMG_SIZE = 224
NUM_EPOCHS = 1
EARLY_STOP_PATIENCE = 4
OPTIMIZER = 'adam'            # adam  #adabound
IS_CUTMIX = True

N_SPLITS = 10
VALID_SET_NUM = 3

DATASET_NAME = 'test111'
MODEL_NAME = 'efn0'

TRAIN_DATA_PATH = '../../../datasets/estern_wells/'
DATAINFO_SAVE_PATH = './dataset_info'
MODEL_SAVE_PATH = './model'
LABEL_PATH = './label'
LOG_PATH = './logs'


####################################################

strategy = tf.distribute.experimental.CentralStorageStrategy()

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

def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10024)])
  except RuntimeError as e:
    print(e)


time = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M") + '_tf2'
weight_file_name = '{epoch:02d}.hdf5'
checkpoint_path = MODEL_SAVE_PATH + DATASET_NAME + '/' + time + '/' + weight_file_name

if not(os.path.isdir(MODEL_SAVE_PATH + DATASET_NAME + '/' + time)):
    os.makedirs(os.path.join(MODEL_SAVE_PATH + DATASET_NAME + '/' + time))
else:
    pass

##################       DataSet   ########################################################

data_info = MakeDataSetInfo(train_data_path=TRAIN_DATA_PATH,
                            save_path=DATAINFO_SAVE_PATH,
                            label_path=LABEL_PATH,
                            dataset_name=DATASET_NAME)

data_df = data_info.cross_validation(n_splits=N_SPLITS)

dataset = MakeDataSet(dataframe=data_df,
                      train_data_path=TRAIN_DATA_PATH,
                      batch_size=BATCH_SIZE,
                      valid_set_num=VALID_SET_NUM,
                      img_size=IMG_SIZE,
                      is_cutmix=IS_CUTMIX)

train_ds, valid_ds = dataset.final_dataset()[:2]

##################         Model    #######################################
backbone = MainModel(model_name=MODEL_NAME,
                     pretrained_weight='imagenet',
                     img_size=IMG_SIZE,
                     label_count=dataset.make_label_len())

main_model = backbone.make_model(opt=1)

############################   CallBacks  ######################################
lr_shedule_fn = build_lrfn()

lr_callback = keras.callbacks.LearningRateScheduler(lr_shedule_fn, verbose=1)

cb_early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
cb_checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     monitor='val_loss',
                                                     save_best_only=True,
                                                     mode='auto')
make_folder(LOG_PATH)
logger = keras.callbacks.CSVLogger(f'./logs/{DATASET_NAME}_log.csv', separator=',', append=False)


TRAIN_STEP_PER_EPOCH = tf.math.ceil(len(dataset.split_dataset()[0]) / BATCH_SIZE).numpy()
VALID_STEP_PER_EPOCH = tf.math.ceil(len(dataset.split_dataset()[1]) / BATCH_SIZE).numpy()


if OPTIMIZER == 'adam':
    main_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),    # label_smoothing=0.1
                  optimizer ='adam',
                  metrics=["accuracy"])
    
    history = main_model.fit(train_ds,
                    epochs=NUM_EPOCHS,
                    steps_per_epoch=TRAIN_STEP_PER_EPOCH,
                    shuffle=False,
                    validation_data=valid_ds,
                    validation_steps=VALID_STEP_PER_EPOCH,
                    verbose=1,
                    callbacks=[lr_callback, cb_checkpointer, cb_early_stopper, logger])  #cb_checkpointer, cb_early_stopper


elif OPTIMIZER == 'adabound':
    main_model.compile(optimizer=AdaBound(lr=1e-3,final_lr=0.1), 
                  loss=tf.keras.losses.CategoricalCrossentropy())


    history = main_model.fit(train_ds,
                        epochs=NUM_EPOCHS,
                        steps_per_epoch=TRAIN_STEP_PER_EPOCH,
                        shuffle=False,
                        validation_data=valid_ds,
                        validation_steps=VALID_STEP_PER_EPOCH,
                        verbose=1,
                        callbacks=[cb_checkpointer, cb_early_stopper, logger])

main_model.save(MODEL_SAVE_PATH + '/' + DATASET_NAME  + '/' + DATASET_NAME + '.h5')