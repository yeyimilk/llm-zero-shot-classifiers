import os
import pathlib
from datetime import datetime
import json
import numpy as np

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

def make_path(name, with_t):
    if with_t:
        name = name_with_datetime(name)
    abs_path = pathlib.Path(__file__).parent.parent.resolve()
    
    sub_path = os.path.dirname(name)
    
    file_dir = f'{abs_path}/results/{sub_path}'
    make_sure_folder_exists(file_dir)
    return abs_path, name

def save_to_file(content, name, type='txt', with_t=False):
    abs_path, name = make_path(name, with_t)
    
    if type == 'json':
        content = json.dumps(content, cls=NpEncoder)
    
    file_path = f'{abs_path}/results/{name}.{type}'
    with open(file_path, 'w') as f:
        f.write(content)
    
    return file_path