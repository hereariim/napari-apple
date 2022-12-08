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
import pandas as pd
# import matplotlib.pyplot as plt
from skimage.io import imread
# import matplotlib.patches as patches
from napari.utils.notifications import show_info

def do_object_detection(layer,path_darknet)-> ImageData:
    
    path_image = str(layer).replace('\\','/')
    os.chdir(path_darknet)
    path_yolov4 = paths.get_weight_file()
    print(path_image)
    print(path_yolov4)

    if os.path.exists("predictions.jpg"):
      print("predictions.jpg was removed")
      os.remove("predictions.jpg")

    subprocess.run(['./darknet','detect',paths.get_cfg_file(),paths.get_weight_file(),path_image])
    show_info("DETECTION done")
    return skimage.io.imread("predictions.jpg")[:,:,:3]

# ./darknet detect /home/irhs/Documents/Herearii/napari-apple/src/napari_apple/main_folder/yolov4-tiny-train.cfg /home/irhs/Documents/Herearii/napari-apple/src/napari_apple/weight-darknet/yolov4-tiny-train_best.weights /home/irhs/Downloads/Apple.jpg

@magic_factory(call_button="Run",filename={"label": "Pick a file:"})
def do_image_detection(
    filename=pathlib.Path.cwd(),path_darknet="Path darknet") -> ImageData:
    if os.path.isdir(path_darknet):
      return do_object_detection(filename,path_darknet)
    else:
      show_info("darknet not found")
    # path_darknet = "/home/irhs/Documents/Herearii/darknet"