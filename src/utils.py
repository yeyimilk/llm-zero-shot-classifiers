import os
import pathlib
from datetime import datetime
import json
import numpy as np
import tensorflow as tf

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def make_sure_folder_exists(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    
def name_with_datetime(prefix='default'):
    now = datetime.now()
    return prefix + '_' + now.strftime("%Y%m%d_%H%M%S")

def make_path(name):
    name = name_with_datetime(name)
    abs_path = pathlib.Path(__file__).parent.resolve()
    
    sub_path = os.path.dirname(name)
    
    file_dir = f'{abs_path}/results/{sub_path}'
    make_sure_folder_exists(file_dir)
    return abs_path, name

def save_to_file(content, name, type='txt'):
    abs_path, name = make_path(name)
    
    if type == 'json':
        content = json.dumps(content, cls=NpEncoder)
    
    file_path = f'{abs_path}/results/{name}.{type}'
    with open(file_path, 'w') as f:
        f.write(content)
    
    return file_path

def setup_device():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # Use all available GPUs
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(len(gpus))])
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    else:
        # No GPU available, use CPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'