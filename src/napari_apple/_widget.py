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
import tensorflow as tf
from tqdm import tqdm
import cv2
from zipfile import ZipFile 
from napari import Viewer

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
  
def do_object_detection(layer,image_viewer): 
    model_New = tf.keras.models.load_model(paths.get_weight_file(),compile=False)

    _, IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS = list(model_New.input.shape) # Taille INPUT
    print(IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS)

    # image_path = "F:\PHENET\AN_2024\Manini\Detection\Day1-20103.jpg"
    classes_list = {0:"apple"}

    SHAPE_h_list = []
    SHAPE_w_list = []
    images_data_or = []
    results_mess = []
    results_coords = []
    results_color = []

    if len(layer.shape)==3:
        layers = np.array(layer)
        original_image = np.copy(layers)        
        # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        h_,w_,_ = original_image.shape
        SHAPE_h_list.append(h_)
        SHAPE_w_list.append(w_)
        images_data_or.append(original_image)
        #PREPROCESSING
        img_rsz = cv2.resize(original_image,(IMG_HEIGHT, IMG_WIDTH))
        img_rsz = img_rsz / 255
        images_data_rs = [img_rsz]
        images_data_rs = np.asarray(images_data_rs).astype(np.float32)
        i = 0
        if True:
            #PROCESSING
            ##DARKNET
            batch = tf.constant(images_data_rs)
            pred_bbox_ = model_New(batch)
                        
                        
            ##NON_MAX_SUPPRESSION
            boxes = pred_bbox_[:, :, 0:4]
            pred_conf = pred_bbox_[:, :, 4:]
                        
            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                            scores=tf.reshape(
                                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                            max_output_size_per_class=50,
                            max_total_size=50,
                            iou_threshold=0.45,
                            score_threshold=0.25
                        )

            pred_bboxes = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

            bbox_rect = []
                        
            image_h, image_w, _ = original_image.shape
            out_boxes, out_scores, out_classes, num_boxes = pred_bboxes
            for k in range(num_boxes[0]):
                if int(out_classes[0][k]) < 0 or int(out_classes[0][k]) > 1: continue
                coor = out_boxes[0][k]
                coor[0] = int(coor[0] * image_h) #y1
                coor[2] = int(coor[2] * image_h) #y2
                coor[1] = int(coor[1] * image_w) #x1
                coor[3] = int(coor[3] * image_w) #x2
                x1 = coor[1]
                x2 = coor[3]
                y1 = coor[0]
                y2 = coor[2]
                            
                # class BBOX
                class_ind = int(out_classes[0][k])
                bbox_mess = classes_list[class_ind]
                results_mess.append(bbox_mess)
                            
                # color BBOX
                class_ind = int(out_classes[0][k])
                bbox_color = [1,0,0]
                results_color.append(bbox_color)
                            
                # coords BBOX
                bbox_rect.append(np.array([[i,y1, x1], [i,y2, x1], [i,y2, x2], [i,y1, x2]])) # coords BBOX
            results_coords+=bbox_rect

        properties = {
                'label': results_mess,
            }
            
        text_parameters = {
                'string': '{label}',
                'size': 12,
                'color': 'red',
                'anchor': 'upper_left',
                'translation': [-3, 0]
            }

        
        text_parameters = {
        'size': 12,
        'color': 'green'}
        # return [(np.array([original_image[:,:,:3],original_image[:,:,:3]]),),(results_coords,{'face_color':'transparent','edge_width':5,'edge_color':'yellow'},'shapes')]

        return image_viewer.add_shapes(
        results_coords,
        shape_type='rectangle',
        edge_width=5,
        edge_color='yellow',
        face_color='transparent'
        )


    if len(layer.shape)==4:
        layers = np.array(layer)
        seq_image = np.copy(layers)
        max_im,_,_,_ = seq_image.shape
        bbox_seq = []
        for i in range(max_im):
            original_image = seq_image[i,...]
            h_,w_,_ = original_image.shape
            SHAPE_h_list.append(h_)
            SHAPE_w_list.append(w_)
            images_data_or.append(original_image)
            #PREPROCESSING
            img_rsz = cv2.resize(original_image,(IMG_HEIGHT, IMG_WIDTH))
            img_rsz = img_rsz / 255
            images_data_rs = [img_rsz]
            images_data_rs = np.asarray(images_data_rs).astype(np.float32)
            # i = 0
            if True:
                #PROCESSING
                ##DARKNET
                batch = tf.constant(images_data_rs)
                pred_bbox_ = model_New(batch)
                            
                            
                ##NON_MAX_SUPPRESSION
                boxes = pred_bbox_[:, :, 0:4]
                pred_conf = pred_bbox_[:, :, 4:]
                            
                boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                                scores=tf.reshape(
                                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                                max_output_size_per_class=50,
                                max_total_size=50,
                                iou_threshold=0.45,
                                score_threshold=0.25
                            )

                pred_bboxes = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

                bbox_rect = []
                            
                image_h, image_w, _ = original_image.shape
                out_boxes, out_scores, out_classes, num_boxes = pred_bboxes
                for k in range(num_boxes[0]):
                    if int(out_classes[0][k]) < 0 or int(out_classes[0][k]) > 1: continue
                    coor = out_boxes[0][k]
                    coor[0] = int(coor[0] * image_h) #y1
                    coor[2] = int(coor[2] * image_h) #y2
                    coor[1] = int(coor[1] * image_w) #x1
                    coor[3] = int(coor[3] * image_w) #x2
                    x1 = coor[1]
                    x2 = coor[3]
                    y1 = coor[0]
                    y2 = coor[2]
                                
                    # class BBOX
                    class_ind = int(out_classes[0][k])
                    bbox_mess = classes_list[class_ind]
                    results_mess.append(bbox_mess)
                                
                    # color BBOX
                    class_ind = int(out_classes[0][k])
                    bbox_color = [1,0,0]
                    results_color.append(bbox_color)
                                
                    # coords BBOX
                    bbox_rect.append(np.array([[i,y1, x1], [i,y2, x1], [i,y2, x2], [i,y1, x2]])) # coords BBOX
                results_coords+=bbox_rect
            bbox_seq.append(results_coords)
        properties = {
                'label': results_mess,
            }
            
        text_parameters = {
                'string': '{label}',
                'size': 12,
                'color': 'red',
                'anchor': 'upper_left',
                'translation': [-3, 0]
            }

        
        text_parameters = {
        'size': 12,
        'color': 'green'}
        #return [(np.array([original_image[:,:,:3],original_image[:,:,:3]]),),(results_coords,{'face_color':'transparent','edge_width':5,'edge_color':'yellow'},'shapes')]

        return image_viewer.add_shapes(
        results_coords,
        shape_type='rectangle',
        edge_width=5,
        edge_color='yellow',
        face_color='transparent'
        )

# @magic_factory(call_button="Run",filename={"label": "Pick a file:"})
@magic_factory(call_button="Run")
def do_image_detection(layer: ImageData,image_viewer: Viewer):
# def do_image_detection(filename=pathlib.Path.cwd()) -> typing.List[napari.types.LayerDataTuple]:
    return do_object_detection(layer,image_viewer)

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
def save_as_zip(layer_bbx: ShapesData, layer_RGB: ImageData):
    return save_shape(layer_bbx,layer_RGB)
        
           
def save_shape(layer_bbx,layer_RGB):
    save_button = QPushButton("Save as zip")
    filename, _ = QFileDialog.getSaveFileName(save_button, "Save as csv", ".", "csv")

    nbr_image = layer_RGB.shape 
    
    if len(nbr_image)==4:
        total_RGB = nbr_image[0]
        
        A = []
        for ix in layer_bbx: 
            vg = np.zeros((4,3),dtype='object')
            vg[:,:3] = ix
            A.append(vg)
        # bbx_array = np.array(A,dtype='int32')
        bbx_array = np.array(A)

        CX_ = []
        CY_ = []
        H_ = []
        W_ = []
        C_ = []

        for ix in tqdm(range(total_RGB),"Extracting"):
            #recherche des bbx de image courant
            bbx_current = bbx_array[bbx_array[:,:,0]==[ix,ix,ix,ix]]
            # bbx_current = bbx_array[bbx_array[:,0]==[ix,ix,ix,ix]] 
            n = bbx_current.shape[0]
            if n==0:
                CX_.append('')      
                CY_.append('')      
                H_.append('')      
                W_.append('')      
                C_.append('')       
            else:
                for i,j in zip(range(0,n,4),range(4,n+4,4)):
                    bbx_coord = bbx_current[i:j,:]  
                    _ , minr, minc = bbx_coord[0] # <-- adapter avec orientation du rectangle
                    _ , maxr, minc = bbx_coord[1]
                    _ , maxr, maxc = bbx_coord[2]
                    _ , minr, maxc = bbx_coord[3]
                    cx = (int(minc)+int(maxc))/2
                    cy = (int(minr)+int(maxr))/2
                    h = int(maxr)-int(minr)
                    w = int(maxc)-int(minc)
                    CX_.append(cx)
                    CY_.append(cy)
                    H_.append(h)
                    W_.append(w)
                    print(ix)
                    C_.append(str(ix))
        df = pd.DataFrame({'cx':CX_,'cy':CY_,'h':H_,'w':W_,'image':C_})
        df.to_csv(filename,index=False)
        
    elif len(nbr_image)==3:
        total_RGB = nbr_image[0] 
        bbx_array = np.array(layer_bbx,dtype='int32')

        #recherche des bbx de image courant
        n = bbx_array.shape[0]
        if n==0:
            CX_ = ['']
            CY_ = ['']
            H_ = ['']
            W_ = ['']        
        else:
            CX_ = []
            CY_ = []
            H_ = []
            W_ = []
            for ix in range(n):
                bbx_coord = bbx_array[ix]
                _ , minr, minc = bbx_coord[0]
                _ , maxr, minc = bbx_coord[1]
                _ , maxr, maxc = bbx_coord[2]
                _ , minr, maxc = bbx_coord[3]
                cx = (minc+maxc)/2
                cy = (minr+maxr)/2
                h = maxr-minr
                w = maxc-minc
                CX_.append(cx)
                CY_.append(cy)
                H_.append(h)
                W_.append(w)
        df = pd.DataFrame({'cx':CX_,'cy':CY_,'h':H_,'w':W_})
        df.to_csv(filename,index=False)

    show_info('Compressed file done')