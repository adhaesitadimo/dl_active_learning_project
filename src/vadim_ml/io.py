
import os
import numpy as np
import pickle
import json
import inspect
from collections.abc import Iterable

def load_pickle(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except (EOFError, OSError) as e:
        raise FileNotFoundError(f'Failed to load object from {path}') from e

def dump_pickle(obj, path):
    try:
        with open(path, 'wb') as f:
            return pickle.dump(obj, f)
    except (EOFError, OSError) as e:
        raise Exception(f'Failed to write {path}') from e

def jsonify(obj):
    """Transform datatypes unsupported by JSON (sets, ndarrays)"""

    t = type(obj)

    if t == dict:
        return {jsonify(k):jsonify(v) for k,v in obj.items()}
    if t in [list, set]:
        return [jsonify(x) for x in obj]
    if t == np.ndarray:
        return [jsonify(x) for x in obj.tolist()]
    
    return obj

def unjsonify(obj):
    """Transform string keys back to numbers, if any"""

    t = type(obj)

    if t == dict:
        try:
            return {int(k):unjsonify(v) for k,v in obj.items()}
        except ValueError:
            return {k:unjsonify(v) for k,v in obj.items()}
    if t == list:
        return [unjsonify(x) for x in obj]

    return obj

def dump_json(obj, path):
    obj = jsonify(obj)

    try:
        with open(path, 'w') as f:
            return json.dump(obj, f)
    except (EOFError, OSError) as e:
        raise IOError(f'Failed to write {path}') from e

def load_json(path):
    try:
        with open(path, 'r') as f:
            return unjsonify(json.load(f))
    except (EOFError, OSError) as e:
        raise FileNotFoundError(f'Failed to load object from {path}') from e

def dump_txt(obj, path, encoding='utf-8'):
    try:
        with open(path, 'w', encoding=encoding) as f:
            return f.write(obj)
    except (EOFError, OSError) as e:
        raise IOError(f'Failed to write {path}') from e

def load_txt(path):
    try:
        with open(path, 'r') as f:
            return f.read()
    except (EOFError, OSError) as e:
        raise FileNotFoundError(f'Failed to load object from: {path}') from e

def drop_ext(filename):
    spl = filename.split('.')
    if len(spl) != 1:
        spl = spl[:-1]
    return '.'.join(spl)

class DirectoryDict():
    def __init__(self, *path):
        self.path = path
        self.filenames = os.listdir(os.path.join(*path))
 
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, key):
        for ext in ['.pickle', '.json', '.txt', '']:
            try:
                return load_file(*self.path, str(key) + ext)
            except FileNotFoundError:
                pass
        raise FileNotFoundError()

    def __iter__(self):
        return iter((drop_ext(filename), load_file(*self.path, filename)) 
                    for filename in self.filenames)

    def items(self):
        for filename in self.filenames:
            yield drop_ext(filename), load_file(*self.path, filename) 

    def keys(self):
        return [drop_ext(f) for f in self.filenames]

    def values(self):
        return [load_file(*self.path, filename) 
                for filename in self.filenames]

    def copy(self):
        return self # It's immutablem so why not

# Mutual recursion is never easy, always fun and sometimes worth it
load_dir = None
load_file = None

def load_dir(*path, filter_f=lambda x: True):
    return DirectoryDict(*path)

def load_file(*path):
    p = os.path.join(*path)
    p = os.path.abspath(p)

    if os.path.isdir(p):
        return load_dir(*path)
    elif path[-1][-5:] == '.json':
        return load_json(p)
    elif path[-1][-7:] == '.pickle':
        return load_pickle(p)
    else:
        # txt is the default for reading
        return load_txt(p)

def add_ext(path, ext):
    return path[:-1] + (path[-1] + ext,)

counter = 0

def dump_file(data, *path):
    path = tuple(str(x) for x in path)

    if path[:-1]:
        os.makedirs(os.path.join(*path[:-1]), exist_ok=True)
    p = os.path.join(*path)
    p = os.path.abspath(p)

    if path[-1][-5:] == '.json':
        return dump_json(data, p)
    elif path[-1][-7:] == '.pickle':
        return dump_pickle(data, p)
    elif path[-1][-4:] == '.txt':
        return dump_txt(data, p)
    elif inspect.isgenerator(data):
        for key, value in enumerate(data):
            dump_file(value, *(path + (key,)))
    elif isinstance(data, str):
        return dump_file(data, *path[:-1], path[-1] + '.txt')
    else:
        try:
            dict_data = dict(data)
            for key, value in dict_data.items():
                dump_file(value, *(path + (key,)))
        except (TypeError, ValueError):
            return dump_file(data, *path[:-1], path[-1] + '.pickle')
