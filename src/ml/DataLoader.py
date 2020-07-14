import numpy as np
import random
import os
from glob import glob
from skimage.io import imread
import matplotlib.pyplot as plt

from src.ml.augmentation import normalize_batch, augment_and_normalize_batch


def get_file(file_pattern):
    filenames = glob(file_pattern)
    if len(filenames) != 0:
        return filenames[0]
    return ""


def get_random_range(begin, end):
    if end > begin:
        return np.random.randint(begin, end)
    else:
        return begin


def get_image_patch(image, patch_shape, corner_coordinate):
    patch = image[corner_coordinate[0]:corner_coordinate[0] + patch_shape[0],
                  corner_coordinate[1]:corner_coordinate[1] + patch_shape[1],
                  corner_coordinate[2]:corner_coordinate[2] + patch_shape[2]]
    return patch


def get_train_generator(data_loader, batch_size):
    while True:
        image, label = data_loader.get_random_training_stack()
        rand_rot = random.randint(0, 3)
        image = np.rot90(image, k=rand_rot, axes=(1, 2))  #  randomly rotate 90
        label = np.rot90(label, k=rand_rot, axes=(1, 2))
        S = image.shape
        region = (0, S[0], 0, S[1], 0, S[2])
        corner_coords = data_loader.get_random_batch_corner_coordinates(batch_size, region)
        image_patches = []
        label_patches = []
        for corner in corner_coords:
            image_patch = get_image_patch(image, data_loader.patch_shape, corner)
            image_patches.append(image_patch)
            label_patch = get_image_patch(label, data_loader.patch_shape, corner)
            label_patches.append(label_patch)

        image_patches = np.moveaxis(np.array(image_patches), 1, 3)  # tf --> (N, W, H, D)
        label_patches = np.moveaxis(np.array(label_patches), 1, 3)

        augmented_imgs, augmented_labels = augment_and_normalize_batch(image_patches, label_patches, data_loader.data_aug_params)

        yield augmented_imgs, augmented_labels


def get_validation_data(data_loader):
    validation_x, validation_y = data_loader.get_random_validation_stack()

    D, H, W = validation_x.shape
    region = (0, D, 0, H, 0, W)

    validation_x_patches = []
    validation_y_patches = []

    for i in range(4):
        corner_coords = data_loader.get_random_batch_corner_coordinates(4, region)
        for corner in corner_coords:
            x_patch = get_image_patch(validation_x, data_loader.patch_shape, corner)
            validation_x_patches.append(x_patch)
            y_patch = get_image_patch(validation_y, data_loader.patch_shape, corner)
            validation_y_patches.append(y_patch)

    validation_x_patches = np.moveaxis(np.array(validation_x_patches), 1, 3)
    validation_y_patches = np.moveaxis(np.array(validation_y_patches), 1, 3)

    validation_x_patches = normalize_batch(validation_x_patches)

    validation_data = (validation_x_patches, validation_y_patches)

    return validation_data


class DataLoader:
    def __init__(self, params, images_dir, labels_dir, save_dir, aggregation_method_used):
        self.data_aug_params = params['data_augmentation']

        model_params = params['model']
        self.patch_shape = model_params['patch_shape']
        self.batch_size = model_params['batch_size']
        self.n_layers = model_params['layers']
        self.start_ch = model_params['start_ch']

        data_params = params['data']
        self.validation_rois = data_params['validation_rois']
        self.holdout_rois = data_params['holdout_rois']
        self.training_rois = data_params['training_rois']

        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.save_dir = save_dir
        self.aggregation_method_used = aggregation_method_used

        self.border_width = params['border_width_nm']

        model_folder = f'{self.border_width}nm_{self.n_layers}L_{self.start_ch}ch_{self.patch_shape}_{self.aggregation_method_used}'
        self.checkpoint_folder = os.path.join(self.save_dir, model_folder)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.checkpoint_folder):
            os.makedirs(self.checkpoint_folder)
        if self.training_rois == "*":
            rois = [os.path.splitext(filename)[0] for filename in os.listdir(self.labels_dir)]
            self.training_rois = [roi for roi in rois
                                  if roi not in self.validation_rois
                                  and roi not in self.holdout_rois]

    def load_stack_pair(self, roi):
        image = imread(get_file(self.images_dir + roi + ".*"))
        label = np.where(imread(get_file(self.labels_dir + roi + ".*")) >= 0.5, 1, 0)   # binary conversion by round-off
        return [image, label]

    def get_name(self):
        return os.path.join(self.checkpoint_folder, 'model')

    def get_random_training_stack(self):
        return self.load_stack_pair(random.choice(self.training_rois))

    def get_random_validation_stack(self):
        return self.load_stack_pair(random.choice(self.validation_rois))

    def check_data(self):
        print("Training sets")
        for roi in self.training_rois:
            pair = self.load_stack_pair(roi)
            print(roi, pair[0].shape, pair[1].shape)

        print("Validation sets")
        for roi in self.validation_rois:
            pair = self.load_stack_pair(roi)
            print(roi, pair[0].shape, pair[1].shape)

    def get_random_batch_corner_coordinates(self, batch_size, region):
        """ @param region: (Z0, Z1, Y0, Y1, X0, X1) return coordniates in high/low range given
            @return: (batch_size, 3) stacks of random (Z, Y, X) coordinates
        """
        r = np.array([[get_random_range(region[0], region[1] - self.patch_shape[0]),
                       get_random_range(region[2], region[3] - self.patch_shape[1]),
                       get_random_range(region[4], region[5] - self.patch_shape[2])
                       ] for _ in range(batch_size)])
        return r

    def visualise_training_batch(self):
        data_gen = get_train_generator(self, 1)
        image_batch, label_batch = next(data_gen)
        image = image_batch[0][:, :, 0]
        label = label_batch[0][:, :, 0]
        plt.imshow(image, cmap='gray')
        plt.imshow(np.ma.masked_where(label == 0, label), vmin=0, vmax=1, cmap='cool', alpha=0.5)
        plt.show()
