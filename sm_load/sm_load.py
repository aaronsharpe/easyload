import os, glob
import numpy as np
import datetime

def _load_file(file_path:str):
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

def load(data_dir:str, i:int):
    file_path = glob.glob(os.path.join(data_dir, str(i).zfill(3))+'*')[0]
    data_dict = _load_file(file_path)
    return data_dict


def load2d(data_dir:str, ints:list):
    '''
    Loads all files specified in the ints list
    If -1 is passed as the last element in the list, it will find all files with the
    same trailing description as the first file number
    '''
    data_dict = {}
    if ints[-1] == -1:
        file_path = glob.glob(os.path.join(data_dir, str(ints[0]).zfill(3))+'*')[0]
        split_fname = file_path.split(str(ints[0]).zfill(3))
        files = glob.glob(os.path.join(data_dir,''+ '*' + split_fname[-1]))
    else:
        files = []
        for i in ints:
            files.append(glob.glob(os.path.join(data_dir, str(i).zfill(3))+'*')[0])


    for i, file in enumerate(files):
        data_i = _load_file(file)
        if i == 0:
            for key, value in data_i.items():
                data_dict[key] = np.array([value])
        else:
            for key, value in data_i.items():
                data_dict[key] = np.vstack((data_dict[key], np.pad(value, (0, np.shape(data_dict[key])[1] - len(value)))))
    return data_dict
