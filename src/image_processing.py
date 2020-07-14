import os
import glob
import numpy as np
from skimage.io import imsave, imread
from tifffile import TiffFile
from tqdm import tqdm
from PIL import Image

from src.helpers import get_file


def get_dict(dict, key):
    if key in dict:
        return dict[key]
    return None


def get_tag(tags, key):
    tag = get_dict(tags, key)
    if tag:
        return tag.value
    return ""


def get_tag_resolution(filename):
    size_z = 1
    unit = ""
    with TiffFile(filename) as tif:
        ijtags = tif.imagej_metadata
        imtags = tif.pages[0].tags
        xtags = get_tag(imtags, 'XResolution')
        ytags = get_tag(imtags, 'YResolution')
        if len(xtags) > 0:
            xres = xtags[0] / xtags[1]
        else:
            xres = xtags[0]
        if len(ytags) > 0:
            yres = ytags[0] / ytags[1]
        else:
            yres = ytags[0]
        if ijtags:
            size_z = get_dict(ijtags, 'spacing')
            unit = get_dict(ijtags, 'unit')
    return xres, yres, size_z, unit


def get_tag_imagesize(filename):
    width = 0
    height = 0
    with TiffFile(filename) as tif:
        imtags = tif.pages[0].tags
        width = get_tag(imtags, 'ImageWidth')
        height = get_tag(imtags, 'ImageLength')
    return width, height


def save_image(filepath, image, resx=1, resy=1, size_z=1, res_unit="", compress=False):
    # https://pypi.org/project/tifffile/
    # https://scikit-image.org/docs/0.13.x/api/skimage.external.tifffile.html#imsave
    # https://stackoverflow.com/questions/20529187/what-is-the-best-way-to-save-image-metadata-alongside-a-tif
    # imageJ format: dimensions in TZCYXS order
    metadata = {}
    image2 = image
    if len(image.shape) > 2:
        # reshape into [slices, channels, y, x]
        s = list(image.shape)
        s.insert(-2, 1)
        image2 = image.reshape(tuple(s))
        if size_z != 1:
            metadata['spacing'] = size_z
    if res_unit:
        metadata['unit'] = res_unit
    if compress:
        imsave(filepath, image2, check_contrast=False, imagej=True, resolution=[resx, resy], metadata=metadata, compress=6)
    else:
        imsave(filepath, image2, check_contrast=False, imagej=True, resolution=[resx, resy], metadata=metadata)


def unstack_images(image_dir, imagestack_dir, output_extension=".tiff", add_prefix_z=True, z_index_offset=0, compress=False):
    nprocessed = 0

    for stack_filename in tqdm(os.listdir(imagestack_dir)):
        stack_filepath = os.path.join(imagestack_dir, stack_filename)
        if os.path.isfile(stack_filepath):
            filetitle, _ = os.path.splitext(stack_filename)
            image_array = imread(stack_filepath)
            resx, resy, size_z, res_unit = get_tag_resolution(stack_filepath)
            index = 0
            for image in image_array:
                image_filename = filetitle + "_"
                if add_prefix_z:
                    image_filename += "z"
                image_filename += f"{z_index_offset + index:04d}" + output_extension
                save_image(os.path.join(image_dir, image_filename), image, resx=resx, resy=resy, size_z=size_z, res_unit=res_unit,
                           compress=compress)
                index += 1
            nprocessed += 1

    return nprocessed


def get_stack_filenames(image_dir):
    stack_filenames = set()
    filenames = []
    input_extension = "tiff"

    for file in os.listdir(image_dir):
        if os.path.isfile(os.path.join(image_dir, file)):
            filename, ext = os.path.splitext(file)
            input_extension = ext
            filenames.append(filename)
            filebase = filename.rsplit('_', 1)[0]
            stack_filenames.add(filebase)
    return stack_filenames, filenames, input_extension


def get_filenames(filebase):
    image_range = []
    z_prefix = False
    filenames = sorted(glob.glob(filebase + "*"))
    if len(filenames):
        slicei = os.path.splitext(filenames[0])[0].rsplit('_', 1)[1]
        z_prefix = slicei.lower().startswith("z")
        if z_prefix:
            slicei = slicei[1:]
        mini = int(slicei)
        slicei = os.path.splitext(filenames[-1])[0].rsplit('_', 1)[1]
        if z_prefix:
            slicei = slicei[1:]
        maxi = int(slicei)
        image_range = [mini, maxi]

    return filenames, image_range, z_prefix


def stack_images(image_dir, imagestack_dir, size_z, overwrite=False, compress=False):
    stack_filenames, filenames, input_extension = get_stack_filenames(image_dir)
    image_range = []

    for stack_filename in tqdm(stack_filenames):
        if overwrite or not os.path.exists(os.path.join(imagestack_dir, stack_filename + '.tiff')):
            image_stack = []
            resx = 1
            resy = 1
            res_unit = ""

            filenames, image_range, z_prefix = get_filenames(os.path.join(image_dir, stack_filename))
            source_image = imread(filenames[0])

            for i in range(image_range[0], image_range[1] + 1):
                input_filename = stack_filename + '_'
                if z_prefix:
                    input_filename += 'z'
                input_filename += f"{i:04d}" + input_extension
                input_filepath = os.path.join(image_dir, input_filename)
                if os.path.exists(input_filepath):
                    source_image = imread(input_filepath)
                    resx, resy, _, res_unit = get_tag_resolution(input_filepath)
                else:
                    source_image = np.zeros_like(source_image)
                image_stack.append(source_image)

            np_stack = np.array(image_stack)
            if np_stack.dtype.kind == "O":
                print("Stack error", stack_filename, "Type:", np_stack.dtype, "Shape:", np_stack.shape)
            else:
                save_image(os.path.join(imagestack_dir, stack_filename + '.tiff'), np_stack, resx=resx, resy=resy, size_z=size_z,
                           res_unit=res_unit, compress=compress)

    return image_range


def scale_image(image, new_width, new_height):
    resample = Image.BICUBIC
    # resample = Image.LANCZOS  # LANCZOS looks a bit sharper than BICUBIC
    pil_image = Image.fromarray(image)
    # note pil resize format!: width, height
    if "16" in pil_image.mode:
        # for 16 bit mode PIL only seems to support NEAREST
        resample = Image.NEAREST
    scaled_pil_image = pil_image.resize((new_width, new_height), resample=resample)
    scaled_image = np.asarray(scaled_pil_image)
    return scaled_image


def to_binary(image):
    if image.dtype.kind == 'f':
        image2 = np.where(image >= 0.5, 1, 0).astype(np.float32)
    else:
        image2 = np.where(image >= 128, 255, 0).astype(np.uint8)
    return image2


def to_int(image):
    if image.dtype.kind == 'f':
        image2 = np.where(image >= 0.5, 255, 0).astype(np.uint8)
    else:
        image2 = np.where(image >= 128, 255, 0).astype(np.uint8)
    return image2


def scale_save_image(input_dir, output_dir, filename, scale_xy, scale_z, binary_format=False, compress=False):
    overwrite = True

    # TODO: if z scale is not 1, resample images in z direction

    input_filepath = get_file(os.path.join(input_dir, filename + ".*"))
    if input_filepath:
        file_extension = "." + input_filepath.rsplit(".")[-1]
        output_filepath = os.path.join(output_dir, filename + file_extension)
        if overwrite or not os.path.exists(output_filepath):
            raw_image = imread(input_filepath)
            resx, resy, size_z, res_unit = get_tag_resolution(input_filepath)
            resx *= scale_xy
            resy *= scale_xy
            height, width = raw_image.shape[-2:]
            new_width = int(width * scale_xy + 0.5)
            new_height = int(height * scale_xy + 0.5)
            if len(raw_image.shape) > 2:
                # image stack
                scaled_image = []
                for image_layer in raw_image:
                    scaled_image.append(scale_image(image_layer, new_width, new_height))
                scaled_image = np.array(scaled_image)
            else:
                scaled_image = scale_image(raw_image, new_width, new_height)
            if binary_format:
                scaled_image = to_binary(scaled_image)
            save_image(output_filepath, scaled_image, resx=resx, resy=resy, size_z=size_z, res_unit=res_unit, compress=compress)
