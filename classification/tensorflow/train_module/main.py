from aug_test import DataAugmentation
from train_test import Train

PROJECT_NAME = 'cu'
SEED_PATH = '/home/perth/Desktop/personal_project/2.test/train_test/data/seed_image'
AUGED_PATH = '/home/perth/Desktop/personal_project/2.test/train_test/data/output'

IMG_SIZE = 224
EPOCH = 100
BATCH_SIZE = 16

TEST_SIZE = 0.1
AUG_NUM = 500



# 데이터셋 네임
ds = DataAugmentation(project_name = PROJECT_NAME,
                    seed_path = SEED_PATH,
                    auged_path = AUGED_PATH)

train_images, valid_images, train_info, valid_info = ds.split_seed(test_size = TEST_SIZE)

ds.augmentation(images = train_images, is_train = True, info = train_info, aug_num = AUG_NUM)
ds.augmentation(images = valid_images, is_train = False, info = valid_info, aug_num = AUG_NUM)


train = Train(project_name = PROJECT_NAME,
            auged_path = AUGED_PATH,
            img_size = IMG_SIZE, 
            epoch = EPOCH, 
            batch_size = BATCH_SIZE)


train.model_train()
