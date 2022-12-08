import os.path
# from blossom import config

# CONF = config.get_conf_dict()
homedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# base_dir = CONF['general']['base_directory']
base_dir = "."

def get_base_dir():
    return os.path.abspath(os.path.join(homedir, base_dir))

def get_models_dir():
     return os.path.join(get_base_dir(),"napari_apple")

def get_weight_file():
     first = os.path.join(get_base_dir(),"napari_apple")
     second = os.path.join(first,"weight-darknet")
     return os.path.join(second,"yolov4-tiny-train_best.weights")

def get_obj_data():
     first = os.path.join(get_base_dir(),"napari_apple")
     second = os.path.join(first,"main_folder")
     return os.path.join(second,"obj.data")

def get_cfg_file():
     first = os.path.join(get_base_dir(),"napari_apple")
     second = os.path.join(first,"main_folder")
     return os.path.join(second,"yolov4-tiny-train.cfg")
      
