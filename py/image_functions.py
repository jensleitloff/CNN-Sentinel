# -*- coding: utf-8 -*-
def preprocessing_image_rgb(x):
    # define mean and std values
    mean = [87.845, 96.965, 103.947]
    std = [23.657, 16.474, 13.793]
    # loop over image channels
    for idx, mean_value in enumerate(mean):
        x[..., idx] -= mean_value
        x[..., idx] /= std[idx]
    return x


def preprocessing_image_ms(x):
    # define mean and std values
    mean = [1353.036, 1116.468, 1041.475, 945.344, 1198.498, 2004.878,
            2376.699, 2303.738, 732.957, 12.092, 1818.820, 1116.271, 2602.579]
    std = [65.479, 154.008, 187.997, 278.508, 228.122, 356.598, 456.035,
           531.570, 98.947, 1.188, 378.993, 303.851, 503.181]
    # loop over image channels
    for idx, mean_value in enumerate(mean):
        x[..., idx] -= mean_value
        x[..., idx] /= std[idx]
    return x


def categorical_label_from_full_file_name(files, class_indices):
    from keras.utils import to_categorical
    import os
    # file basename without extension
    base_name = [os.path.splitext(os.path.basename(i))[0] for i in files]
    # class label from filename
    base_name = [i.split("_")[0] for i in base_name]
    # label to indices
    image_class = [class_indices[i] for i in base_name]
    # class indices to one-hot-label
    return to_categorical(image_class, num_classes=len(class_indices))


def simple_image_generator(files, class_indices, batch_size=32,
                           rotation_range=0, horizontal_flip=False,
                           vertical_flip=False):
    from skimage.io import imread
    from skimage.transform import rotate
    import numpy as np
    from random import sample, choice

    while True:
        # select batch_size number of samples without replacement
        batch_files = sample(files, batch_size)
        # get one_hot_label
        batch_Y = categorical_label_from_full_file_name(batch_files,
                                                        class_indices)
        # array for images
        batch_X = []
        # loop over images of the current batch
        for idx, input_path in enumerate(batch_files):
            image = np.array(imread(input_path), dtype=float)
            image = preprocessing_image_ms(image)
            # process image
            if horizontal_flip:
                # randomly flip image up/down
                if choice([True, False]):
                    image = np.flipud(image)
            if vertical_flip:
                # randomly flip image left/right
                if choice([True, False]):
                    image = np.fliplr(image)
            # rotate image by random angle between
            # -rotation_range <= angle < rotation_range
            if rotation_range is not 0:
                angle = np.random.uniform(low=-abs(rotation_range),
                                          high=abs(rotation_range))
                image = rotate(image, angle, mode='reflect',
                               order=1, preserve_range=True)
            # put all together
            batch_X += [image]
        # convert lists to np.array
        X = np.array(batch_X)
        Y = np.array(batch_Y)

        yield(X, Y)
