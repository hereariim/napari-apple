from cv2 import imread
from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
import cv2
from napari.types import ImageData
import os
import numpy as np
import skimage
from skimage.transform import resize
from napari.utils.notifications import show_info
import pathlib
import subprocess
import napari_apple.path as paths

def do_object_detection(
    layer) -> ImageData:
    
    path_image = str(layer).replace('\\','/')
    os.chdir(os.path.join(paths.get_models_dir(),"darknet"))
    subprocess.run(['./darknet','detect','cfg/yolov3.cfg','yolov3.weights',path_image])
    
    return skimage.io.imread(os.path.join(paths.get_models_dir(),"darknet/predictions.jpg"))[:,:,:3]

def image_select(path_image) -> ImageData:
    path_image = str(path_image).replace('\\','/')
    imag = skimage.io.imread(path_image)[:,:,:3]
    return imag

@magic_factory(call_button="Run",filename={"label": "Pick a file:"})
def do_image_select(
    filename=pathlib.Path.cwd()) -> ImageData:
    return image_select(filename)

@magic_factory(call_button="Detection")
def do_model(layer: ImageData) -> ImageData:
    show_info('Running !')
    return do_object_detection(filename)


