def aug_processing(data_set, output_path, aug_num, is_train):
    img_path = data_set['image_path'].sort_index()
    labels = data_set['label'].value_counts().sort_index()

    if is_train == True:
        output_path = output_path + '/train_test'

    else:
        output_path = output_path + '/valid_test'

    for path in img_path:
        file_name = path.split('/')[-1]
        print(file_name)
        file_name = file_name.split('.')[0]
        label = path.split('/')[-2]
        avg = int(math.ceil(aug_num / labels[label]))

        image = cv2.imread(path)

        if not(os.path.isdir(output_path + '/' + label)):
            os.makedirs(output_path + '/' + label)

        else:
            pass

        total_auged = len(glob.glob(output_path + '/' + label + '/*.jpg'))

        if total_auged <= aug_num:
            for i in range(avg):
                transform = A.Compose([
                    A.Resize(224, 224, p=1),
                    # A.Rotate(limit=(-360, 360), p=0.5, border_mode=1),

                    # A.OneOf([
                    #     A.Rotate(limit=(-360, 360), p=0.5, border_mode=1),
                    #     A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.5),
                    # ], p=1),

                    A.OneOf([
                        A.HorizontalFlip(p=0.3),
                        A.Blur(p=0.2),
                        A.VerticalFlip(p=0.3)
                    ], p=1),

                    A.OneOf([
                        A.RandomContrast(p=0.5, limit=(-0.5, 1)),
                        A.RandomBrightness(p=0.5, limit=(-0.2, 0.4)),
                    ], p=1)
                ])
                augmented_image = transform(image=image)['image']
                cv2.imwrite(output_path + '/' + label + '/' + file_name + '_' + str(i) + '_' + str(time.time()) + '.jpg', augmented_image)

        else:
            pass

    return output_path