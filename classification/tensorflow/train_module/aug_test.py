import sys, glob, cv2, os, math, time, pathlib, datetime, argparse
import albumentations as A
import numpy as np
from random import randrange, choice, sample
from tqdm import tqdm
from sklearn.model_selection import train_test_split


## PROJECT_NAME
class DataAugmentation:
    def __init__(self, project_name, seed_path, auged_path):
        #super().__init__()
        self.input_path = seed_path + f'/{project_name}'
        self.output_path = auged_path + f'/{project_name}'
        self.label_list = sorted(os.listdir(self.input_path + '/'))
        self.n_classes = len(self.label_list)

    # {'cider_250_can': 389, 'coca_250_can': 385}
    def make_info(self, images):
        info = {}
        cnt = [0] * len(self.label_list)

        for image in images:
            label = image.split('/')[-2]
            
            if label in self.label_list:
                cnt[self.label_list.index(label)] += 1

        for label, num in zip(self.label_list, cnt):
            info.update({label:num})

        return info


    def split_seed(self, test_size):
        ds_path = pathlib.Path(self.input_path)
        images = list(ds_path.glob('*/*'))
        images = [str(path) for path in images]

        train_images, valid_images = train_test_split(images, test_size = test_size)
        train_info = self.make_info(train_images)
        valid_info = self.make_info(valid_images)

        return sorted(train_images), sorted(valid_images), train_info, valid_info

    def augmentation(self, images, is_train, info, aug_num):
        if is_train:
            output_path = f'{self.output_path}/train'

        else:
            output_path = f'{self.output_path}/valid'
            aug_num = int(aug_num * 0.2)

        for label in self.label_list:
            if not os.path.isdir(f'{output_path}/{label}'):
                os.makedirs(f'{output_path}/{label}')

        for i in tqdm(range(len(images))):
            image = images[i]
            image_name = image.split('/')[-1]
            label = image.split('/')[-2]

            cnt = int(math.ceil(aug_num / info[label]))
            total_images = len(os.listdir(f'{output_path}/{label}'))
            if total_images <= aug_num:
                image = cv2.imread(image)
                transform = A.Resize(224, 224)
                augmented_image = transform(image = image)['image']
                cv2.imwrite(f'{output_path}/{label}/orig_{image_name}', augmented_image)

                for c in range(cnt):
                    transform = A.Compose([
                        A.Resize(224, 224, p=1),
                        A.HorizontalFlip(p=0.4),
                        A.VerticalFlip(p=0.3),
                        A.Blur(p=0.1),

                        A.OneOf([
                            A.RandomContrast(p=0.5, limit=(-0.5, 0.3)),
                            A.RandomBrightness(p=0.5, limit=(-0.2, 0.3))
                        ], p=0.5)
                    ])
                    augmented_image =transform(image=image)['image']
                    cv2.imwrite(f'{output_path}/{label}/aug{c}_{image_name}', augmented_image)

        return output_path 

## 데이터셋 네임
# ds = DataAugmentation(input_path = '/home/perth/Desktop/personal_project/2.test/train_test/data/seed_image',
#                         output_path = '/home/perth/Desktop/personal_project/2.test/train_test/data/output')

# train_images, valid_images, train_info, valid_info = ds.split_seed(test_size = 0.1)

# ds.augmentation(images = train_images, is_train = True, info = train_info, aug_num = 500)
# ds.augmentation(images = valid_images, is_train = False, info = valid_info, aug_num = 500)

