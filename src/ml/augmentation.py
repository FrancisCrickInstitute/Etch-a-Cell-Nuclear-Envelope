import numpy as np
from scipy.ndimage.interpolation import affine_transform
from scipy.ndimage.filters import gaussian_filter


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def zoom_and_rotate_patch(patch, angle, zoom):
    theta = np.deg2rad(angle)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    transform_matrix = rotation_matrix

    zoom_matrix = np.array([[zoom, 0, 0],
                            [0, zoom, 0],
                            [0, 0, 1]])

    transform_matrix = np.dot(transform_matrix, zoom_matrix)

    d, h, w = patch.shape
    transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
    patch = np.rollaxis(patch, 0, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]

    channel_images = [
        affine_transform(
            x_channel,
            final_affine_matrix,
            final_offset,
            order=1,
            mode='reflect'
        ) for x_channel in patch
    ]

    patch = np.stack(channel_images, axis=0)
    patch = np.rollaxis(patch, 0, 1)

    return patch


def normalize_batch(batch):
    """ normalize X batch by subbing mean then dividing by std
        normalized over the 0 axis (patch-wise mean and std)
    """
    mean = batch.mean(axis=(1, 2, 3), keepdims=True)
    std = batch.std(axis=(1, 2, 3), keepdims=True)
    batch = (batch - mean) / (std + 0.0001)
    return batch


sometimes = lambda x: np.random.random() < x  # True x% of the time


def augment_and_normalize_batch(image_patches, label_patches, data_aug_params):
    augmented_image_patches = []
    augmented_label_patches = []

    ROTATION_RANGE = data_aug_params['rotation_range']
    ZOOM_RANGE = data_aug_params['zoom_range']
    CONTRAST_RANGE = data_aug_params['contrast_range']
    BRIGHTNESS_RANGE = data_aug_params['brightness_range']
    BLUR_RANGE = data_aug_params['blur_range']
    NOISE = data_aug_params['noise']

    for i in range(len(image_patches)):

        img_patch = image_patches[i]
        lbl_patch = label_patches[i]

        if sometimes(0.5):  # 50% of the time, rotate or zoom the image
            zoom = np.random.normal(loc=1.0, scale=ZOOM_RANGE)
            angle = np.random.normal(loc=0.0, scale=ROTATION_RANGE)
            img_patch = zoom_and_rotate_patch(img_patch, angle, zoom)
            lbl_patch = zoom_and_rotate_patch(lbl_patch, angle, zoom)

        if sometimes(0.25):  # 25% of the time, blur OR noise up the image
            if sometimes(0.5):
                sigma = np.random.normal(loc=0.0, scale=BLUR_RANGE)
                img_patch = gaussian_filter(img_patch, sigma=sigma)
                # don't blur the segmentation mask
            else:
                img_patch = img_patch + np.random.normal(loc=0.0, scale=NOISE, size=img_patch.shape)

        if sometimes(0.25):  # 25% of the time, change brightness or contrast
            if sometimes(0.5):
                img_patch = img_patch + np.random.normal(loc=0.0, scale=BRIGHTNESS_RANGE)
            else:
                img_patch = img_patch * np.random.normal(loc=1.0, scale=CONTRAST_RANGE)

        augmented_image_patches.append(img_patch)
        augmented_label_patches.append(lbl_patch)

    image_patches = np.stack(augmented_image_patches, axis=0)
    label_patches = np.stack(augmented_label_patches, axis=0)
    image_patches = normalize_batch(image_patches)

    return image_patches, label_patches
