from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
import typing
from napari.types import ImageData, ShapesData
import napari
import os
import numpy as np
import skimage
from skimage.transform import resize
from napari.utils.notifications import show_info
import pathlib
import subprocess
import napari_apple.path as paths
import pandas as pd
from skimage.io import imread
import re
from qtpy.QtWidgets import QTableWidget, QTableWidgetItem, QGridLayout, QPushButton, QFileDialog, QWidget, QListWidget
import sys

def coordonee(path):

    with open(path) as f:
        contents = f.readlines()
    f.close()
    left_x_list = []
    top_y_list = []
    width_list = []
    height_list = []
    A = []
    for ix in contents:
        if "(left_x:" in ix:
            s = re.sub('\s+', ' ', ix)
            s_coord = s[s.find("(")+1:s.find(")")]
            if s.split(":")[0]=="apple":
                liste_coordinate = s_coord.split(' ')
                b,a,d,c = int(liste_coordinate[1])+int(liste_coordinate[5]),int(liste_coordinate[3]),int(liste_coordinate[5]),int(liste_coordinate[7])
                A.append(np.array([[a,b],[a+c,b],[a+c,b-d],[a,b-d]]))
    A = np.array(A)
    return A
  
def do_object_detection(layer,path_darknet):
    
    path_image = str(layer).replace('\\','/')
    os.chdir(path_darknet)
    path_yolov4 = paths.get_weight_file()
    print(path_image)
    print(path_yolov4)
    print(paths.get_obj_data())
    f = open(paths.get_obj_data(), "w")
    f.write("classes=1\n")
    f.write("names="+paths.get_obj_names())
    f.close()

    # subprocess.run(['./darknet','detect',paths.get_cfg_file(),paths.get_weight_file(),path_image])
    if sys.platform=="linux":
        os.system('./darknet detector test '+paths.get_obj_data()+' '+paths.get_cfg_file()+' '+paths.get_weight_file()+' -ext_output '+path_image+' > '+paths.send_result())
    elif sys.platform=="win32":
        os.system('darknet.exe detector test '+paths.get_obj_data()+' '+paths.get_cfg_file()+' '+paths.get_weight_file()+' -ext_output '+path_image+' > '+paths.send_result())
    
    
    path=paths.send_result()
    bbox_rects = coordonee(path)
    show_info(f"{len(bbox_rects)} apple detected")
    text_parameters = {
    'size': 12,
    'color': 'green'}
    return [(skimage.io.imread(path_image)[:,:,:3],),(bbox_rects,{'face_color':'transparent','edge_width':5,'edge_color':'yellow'},'shapes')]

# ./darknet detect /home/irhs/Documents/Herearii/napari-apple/src/napari_apple/main_folder/yolov4-tiny-train.cfg /home/irhs/Documents/Herearii/napari-apple/src/napari_apple/weight-darknet/yolov4-tiny-train_best.weights /home/irhs/Downloads/Apple.jpg

@magic_factory(call_button="Run",filename={"label": "Pick a file:"})
def do_image_detection(filename=pathlib.Path.cwd(),Darknet="Path darknet") -> typing.List[napari.types.LayerDataTuple]:
    # /home/g-laris89/Documents/darknet
    print(Darknet,os.path.isdir(Darknet))
    if os.path.isdir(Darknet):
      return do_object_detection(filename,Darknet)
    else:
      show_info("darknet not found")
    # path_darknet = "/home/irhs/Documents/Herearii/darknet"
    # /home/g-laris89/Documents/darknet
    # /home/g-laris89/Documents/napari-apple/src/napari_apple/main_folder/yolov4-tiny-train.cfg

def change_u1(u1):
    new_u1 = np.zeros(u1.shape)
    new_u2 = np.zeros(u1.shape)
    for i in range(4):
        new_u1[i,0]=u1[i,1]
        new_u1[i,1]=u1[i,0]
    for i in range(3,-1,-1):
        new_u2[3-i]=new_u1[i]
    return new_u2

def get_data_coord(u1):
    x1 = u1[2,1]
    x2 = u1[0,1]
    y1 = u1[0,0]
    y2 = u1[1,0]
    width = x2-x1
    heigh = y2-y1
    cx = width/2
    cy = heigh/2
    return cx,cy,heigh,width


@magic_factory(call_button="Export",layout="vertical")
def save_as_zip(layer: ShapesData):
    save_button = QPushButton("Save")
    filename, _ = QFileDialog.getSaveFileName(save_button, "Save", ".", "csv")
    import pandas as pd
    CX = []
    CY = []
    HGH = []
    WDTH = []
    for current_un in layer:
        if current_un[0,0]==current_un[1,0]:
            current_un=change_u1(current_un)
        # print(current_un)
        cx,cy,heigh,width = get_data_coord(current_un)
        CX.append(np.round(cx,2))
        CY.append(np.round(cy,2))
        HGH.append(np.round(heigh,2))
        WDTH.append(np.round(width,2))
    
    df = pd.DataFrame({'center_x':CX,'center_y':CY,'heigh':HGH,'width':WDTH})
    df.to_csv(filename)  
    show_info('Export')
        
    
    