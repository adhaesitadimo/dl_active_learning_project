import torch
import os
from vadim_ml.io import load_file, dump_file, add_ext

def generic_memoize(save=None, load=None):
    def memoizer(f):
        def memoized_f(*args):
            cached_response = load(args)

            if not cached_response:
                save(args, f(*args))
                cached_response = load(args)

            return cached_response

        return memoized_f
    return memoizer

memoize_caches = []

def flush_memoize_caches():
    for cache in memoize_caches:
        cache.clear()

def memoize(f):
    cache = {}
    memoize_caches.append(cache)
    def save(keys, value):
        cache[keys] = value
    def load(keys):
        return cache.get(keys)

    return generic_memoize(save=save, load=load)(f)

def disk_memoize(*path):
    def save(keys, value): 
        full_path = path + tuple(str(k) for k in keys)
        return dump_file(value, *full_path)
    def load(keys):
        full_path = path + tuple(str(k) for k in keys)
        val = None
        for ext in ['', '.pickle', '.json', '.txt']:
            try:
                val = load_file(*add_ext(full_path, ext))
            except OSError:
                pass
        return val

    return generic_memoize(save=save, load=load)

def torch_memoize(*path):
    def save(keys, model):
        full_path = path + tuple(str(k) for k in keys)
        file = os.path.join(*add_ext(full_path, '.torch'))
        return torch.save(model.cpu(), file)
    def load(keys):
        full_path = path + tuple(str(k) for k in keys)
        file = os.path.join(*add_ext(full_path, '.torch'))
        try:
            return torch.load(file)
        except FileNotFoundError:
            return
    return generic_memoize(save=save, load=load)

def lazy_initialization(initialize):
    doer = None

    def do_thing(*args):
        nonlocal doer
        
        if not doer:
            doer = initialize()

        return doer(*args)

    return do_thing