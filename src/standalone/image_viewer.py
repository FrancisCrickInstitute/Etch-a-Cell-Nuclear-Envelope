"""
Compare two different image stacks, such as a reference EM image and annotation, with a GUI to
scroll through the slices.
"""
import os
import tkinter as tk

import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib as mpl
from src.helpers import *
import numpy as np
from skimage.io import imread


mpl.rcParams['figure.dpi'] = 320

label_intensity = 0.5


class ImageExtractor:
    def __init__(self, image_dir, label_dir, save_dir, roi, use_stacks, start=0):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.save_dir = save_dir
        self.roi = roi
        self.use_stacks = use_stacks
        self.index = start
        self.len = 0
        self.init()

    def next(self):
        if self.index + 1 < self.len:
            self.index += 1
        else:
            self.index = 0
        return self.index

    def prev(self):
        if self.index - 1 >= 0:
            self.index -= 1
        else:
            self.index = self.len - 1
        return self.index

    def init(self):
        print("Reading images...")
        if use_stacks:
            filepath = get_file(self.image_dir + self.roi + ".*")
            if filepath:
                self.image_stack = imread(filepath)
                self.len = len(self.image_stack)
            else:
                print("No source image found")
            filepath = get_file(self.label_dir + self.roi + ".*")
            if filepath:
                self.label_stack = imread(filepath) > 0.5
                self.len = len(self.label_stack)
            else:
                print("No label image found")
        else:
            filepath = self.image_dir + self.roi + '*'
            filenames = get_all_files(filepath)
            if len(filenames):
                for filename in filenames:
                    num = os.path.splitext(filename)[0].rsplit("_")[-1]
                    if num.startswith('z'):
                        num = int(num[1:])
                        if num > self.len:
                            self.len = num
            else:
                print("No source image found")

    def get_image0(self, image_dir):
        image = []
        filename = self.roi
        if not use_stacks:
            filename += f'_z{self.index:04d}'
        filepath = get_file(image_dir + filename + ".*")
        if filepath:
            image_array = imread(filepath)
            image = image_array

        return image

    def get_image(self):
        if use_stacks:
            return self.image_stack[self.index]
        else:
            return self.get_image0(self.image_dir)

    def get_label(self):
        if use_stacks:
            return self.label_stack[self.index]
        else:
            return self.get_image0(self.label_dir)

    def save(self):
        output_filepath = self.save_dir + self.roi + "_" + str(self.index) + ".tif"
        image = self.get_image() / 0xFF
        label = self.get_label()
        b = image + label * label_intensity
        g = image
        r = image + label * label_intensity
        b[b > 1] = 1
        g[g > 1] = 1
        r[r > 1] = 1
        rgbimage = cv2.merge(((b * 0xFF).astype(np.uint8),
                              (g * 0xFF).astype(np.uint8),
                              (r * 0xFF).astype(np.uint8)))
        cv2.imwrite(output_filepath, rgbimage)
        print("Image saved:", output_filepath)


class ImageViewer(tk.Frame):
    def __init__(self, extractor, master=None):
        super().__init__(master)
        self.master = master
        self.extractor = extractor
        self.fig = Figure(constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.locator = tk.Entry(self, justify='center', width=50)

        self.leftButton = tk.Button(self, padx=10, pady=10)
        self.rightButton = tk.Button(self, padx=10, pady=10)
        self.saveButton = tk.Button(self, padx=10, pady=10)
        self.quitButton = tk.Button(self, padx=10, pady=10)

        # set up widgets
        self.master.title('Annotation Viewer')
        self.pack(padx=10, pady=10)

        self.locator.pack(side=tk.TOP)

        self.leftButton['text'] = '<'
        self.leftButton['command'] = self.prev
        self.leftButton.pack(side=tk.LEFT)
        master.bind("<Left>", self.prev_key)

        self.rightButton['text'] = '>'
        self.rightButton['command'] = self.next
        self.rightButton.pack(side=tk.RIGHT)
        master.bind("<Right>", self.next_key)

        self.quitButton['text'] = 'Quit'
        self.quitButton['command'] = self.master.destroy
        self.quitButton.pack(side=tk.BOTTOM)
        master.bind("<Escape>", self.quit_key)

        self.saveButton['text'] = 'Save'
        self.saveButton['command'] = self.save
        self.saveButton.pack(side=tk.BOTTOM)

        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.update_locator()
        self.update_graph()

    def update_locator(self):
        self.locator.config(state='normal')
        self.locator.delete(0, tk.END)
        self.locator.insert(tk.END, 'Annotation: ' + str(self.extractor.index + 1) + ' of ' + str(self.extractor.len))
        self.locator.config(state='disabled')

    def update_graph(self):
        self.fig.clf()
        axes = self.fig.gca()

        image = self.extractor.get_image()
        image_set = (len(image) != 0)
        label = self.extractor.get_label()
        label_set = (len(label) != 0)

        if image_set:
            axes.imshow(image, vmin=0, vmax=255, cmap='gray')
        if label_set:
            axes.imshow(np.ma.masked_where(label == 0, label), vmin=0, vmax=1, cmap='cool', alpha=label_intensity)
        self.canvas.draw()

    def prev(self):
        self.extractor.prev()
        self.update_locator()
        self.update_graph()

    def next(self):
        self.extractor.next()
        self.update_locator()
        self.update_graph()

    def save(self):
        self.extractor.save()

    def quit_key(self, event):
        self.master.destroy()

    def prev_key(self, event):
        self.prev()

    def next_key(self, event):
        self.next()


def view_images(image_dir, label_dir, save_dir, roi, use_stacks, start_index=0):
    extractor = ImageExtractor(image_dir, label_dir, save_dir, roi, use_stacks, start_index)
    root = tk.Tk()
    viewer = ImageViewer(extractor, master=root)
    viewer.mainloop()


if __name__ == '__main__':
    resource_dir = "../../projects/nuclear/resources/"
    image_raw_dir = resource_dir + "images/raw/"
    label_raw_dir = resource_dir + "images/raw-labels/"
    save_dir = resource_dir + "images/"

    use_stacks = ('stack' in image_raw_dir)
    start_index = 150

    #training data set
    #roi = 'ROI_1416-1932-171'
    #roi = 'ROI_1608-912-1'
    #roi = 'ROI_1716-7800-517'
    #roi = 'ROI_3768-7248-143'

    #roi = 'ROI_1536-3456-213'
    #roi = 'ROI_1584-6996-1'
    #roi = 'ROI_2448-4704-271'
    #roi = 'ROI_2820-6780-468'
    #roi = 'ROI_2832-1692-1'
    #roi = 'ROI_3000-3264-393'
    #roi = 'ROI_3516-5712-314'
    #roi = 'ROI_3576-5232-35'
    #roi = 'ROI_3972-1956-438'
    #roi = 'ROI_4320-1260-95'

    # validation data set
    #roi = 'ROI_2052-5784-112'
    #roi = 'ROI_3588-3972-1'

    # test data set
    roi = "ROI_1656-6756-329"
    #roi = "ROI_3624-2712-201"
    #roi = "Run2_ROI00_16bit_all-2k"
    #roi = "Broken_NE_cropped"

    view_images(image_raw_dir, label_raw_dir, save_dir, roi, use_stacks, start_index)
