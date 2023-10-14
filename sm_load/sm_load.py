import os, glob
import numpy as np
import datetime


def load(data_dir:str, i:int):
    file_path = glob.glob(os.path.join(data_dir, str(i).zfill(3))+'*')[0]
    with open(file_path) as f:
        first_line = f.readline()
        meta_data = [str.strip() for str in first_line.split('\t')]
    
    data = np.loadtxt(file_path, skiprows=1, usecols=range(1, len(meta_data)))
    data_dict = {}
    for i, key in enumerate(meta_data):
        if i == 0:
            times = np.loadtxt(file_path, dtype='str', skiprows=1, usecols=0)
            data_dict['Timestamp'] = np.array([(datetime.datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%f") - datetime.datetime(1970, 1, 1)).total_seconds() for time in times])
        else:
            data_dict[key] = data[:, i-1]
    return data_dict


def load2d(data_dir:str, ints:list):
    data_dict = {}
    for e, i in enumerate(ints):
        data_i = load(data_dir, i)
        if e == 0:
            for key, value in data_i.items():
                data_dict[key] = np.array([value])
        else:
            for key, value in data_i.items():
                data_dict[key] = np.vstack((data_dict[key], np.pad(value, (0, np.shape(data_dict[key])[1] - len(value)))))
    return data_dict
