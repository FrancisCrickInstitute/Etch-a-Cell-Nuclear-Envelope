from glob import glob
import csv
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def set_pandas_config():
    pd.set_option('display.max_columns', 30)
    pd.set_option('display.max_rows', 2500)
    pd.set_option('display.width', 1000)
    # pd.set_option('max_colwidth', -1)


# processed csvs should be small and hence fit in memory
def load_processed_csv(path):
    csv.field_size_limit(2**30)
    return pd.read_csv(path,
                       engine='c',
                       error_bad_lines=False)


def pad(string):
    string = str(string)
    for i in range(max(0, 4 - len(string))):
        string = '0' + string
    return string


def print_annotations_per_slice(df):
    """
    Useful for finding slices with a decent number of annotations.
    """
    df = df.groupby(['slice z', 'ROI'])
    for idx, group in df:
        print(f'slice/ROI: {idx}, annotations: {len(group.index)}')


def get_all_files(file_pattern):
    return glob(file_pattern)


def get_file(file_pattern):
    filenames = get_all_files(file_pattern)
    if len(filenames) != 0:
        return filenames[0]
    return ""


def showimage(image):
    dpi = matplotlib.rcParams['figure.dpi']
    height, width = image.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(image)
    plt.show()


def dpum_to_sizenm(dpum):
    # pixels/micron -> pixel size in nm
    size_nm = 1000 / dpum
    return size_nm


def sizenm_to_dpum(size_nm):
    # pixel size in nm -> pixels/micron
    dpum = 1000 / size_nm
    return dpum
