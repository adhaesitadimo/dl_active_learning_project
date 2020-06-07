import multiprocessing as mp
from collections.abc import Iterable


CUDA_DEVICES = mp.Queue()
WORKER_CUDA_DEVICE = None


def initialize_worker():
    global CUDA_DEVICES
    global WORKER_CUDA_DEVICE
    WORKER_CUDA_DEVICE = CUDA_DEVICES.get()
    print('Worker cuda device:', WORKER_CUDA_DEVICE)


def run_tasks(config, f_task):
    global CUDA_QUEUE
    
    if not isinstance(config.cuda_devices, Iterable):
        cuda_devices = [config.cuda_devices]
    else:
        cuda_devices = config.cuda_devices.split(',')
    
    print('Cuda devices:', cuda_devices)
    
    for cuda_device in cuda_devices:
        CUDA_DEVICES.put(cuda_device)
    
    if 'task_names' in config and config.task_names:
        tasks = [t for t in config.tasks if t.name in config.task_names.split(',')]
    else:
        tasks = config.tasks
        
    print('Tasks:', [t.name for t in config.tasks])
    
    pool = mp.Pool(len(cuda_devices), initializer=initialize_worker)
    pool.map(f_task, tasks)
    