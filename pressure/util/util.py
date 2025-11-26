import json
from pathlib import Path
import re
import psutil
import os
from math import floor
from glob import glob
import numpy as np
import torch
import random
from types import SimpleNamespace
import yaml

def extract_idx(folder_name):
    match = re.search(r'OM(\d+)$', folder_name)
    return int(match.group(1)) if match else None

def simplenamespace_to_dict(namespace):
    if isinstance(namespace, SimpleNamespace):
        return {k: simplenamespace_to_dict(v) for k, v in namespace.__dict__.items()}
    elif isinstance(namespace, list):
        return [simplenamespace_to_dict(item) for item in namespace]
    elif isinstance(namespace, dict):
        return {k: simplenamespace_to_dict(v) for k, v in namespace.items()}
    else:
        return namespace

def dict_to_simplenamespace(d):
    if isinstance(d, dict):
        for key, value in d.items():
            d[key] = dict_to_simplenamespace(value)
        return SimpleNamespace(**d)
    return d

def load_config(filepath):
    filepath = Path(filepath)
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)
    return dict_to_simplenamespace(data)

def cast_array(array, dtype):
    if isinstance(array, np.ndarray):
        return array.astype(dtype) 
    elif isinstance(array, torch.Tensor):
        return array.to(dtype)
    else:
        raise ValueError('Unsupported array type')  

import warnings
def load_model_checkpoint(path, model=None, optimizer=None, lr_scheduler=None, cfg=None):
    warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No checkpoint found at '{path}'")

    checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=True)
    if model is not None:
        model.load_state_dict(checkpoint['model'])

    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if lr_scheduler is not None and 'lr_scheduler' in checkpoint:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    
    epoch = checkpoint.get('epoch', -1)
    
    print(f"Loaded checkpoint '{path}' (epoch {epoch})")
    return epoch, model, optimizer, lr_scheduler

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

def numpy_to_python_native_types(obj):
    if isinstance(obj, dict):
        return {key: numpy_to_python_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python_native_types(element) for element in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    else:
        return obj

def natural_sort(l):
    """
    Sort the given list in the way that humans expect.
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def split_chunk_paths(chunk_dir, subject, train_val_split=0.9, shuffle=True):
    """Return lists of train, val, and test chunk file paths."""
    all_files = sorted(glob(os.path.join(chunk_dir, "subject_*.pkl")))
    all_files = [Path(f).as_posix() for f in all_files]
    train_val, test = [], []

    for f in all_files:
        if f"subject_{subject}_" in f:
            test.append(f)
        else:
            train_val.append(f)

    if shuffle:
        random.shuffle(train_val)
    split = int(len(train_val) * train_val_split)
    return train_val[:split], train_val[split:], sorted(test)

class MemoryMonitor:
    """Monitor memory usage during dataset operations"""
    def __init__(self):
        self.process = psutil.Process()
        self.start_memory = self.get_memory_mb()
        self.peak_memory = self.start_memory
        self.peak_gpu_memory = 0

    def get_memory_mb(self):
        return self.process.memory_info().rss / 1024 / 1024

    def get_gpu_memory_mb(self):
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0

    def log_memory(self, step_name=""):
        current = self.get_memory_mb()
        delta = current - self.start_memory
        self.peak_memory = max(self.peak_memory, current)
        gpu_mem = self.get_gpu_memory_mb()
        print(f"[{step_name:<20}] CPU: {current:8.1f} MB (Δ{delta:+.1f}) | "
              f"GPU: {gpu_mem:7.1f} MB | Peak CPU: {self.peak_memory:8.1f}")
        return current
    
def recommend_cache_size(avg_chunk_mb=10, fraction=0.15):
    """Estimate a safe number of chunks to cache."""
    avail_mb = psutil.virtual_memory().available / 1024 / 1024
    cache_budget = avail_mb * fraction
    cache_chunks = max(1, floor(cache_budget / avg_chunk_mb))
    print(f"Available: {avail_mb:.1f} MB | Target fraction: {fraction*100:.0f}% "
          f"| Budget: {cache_budget:.1f} MB → Recommended cache: {cache_chunks} chunks")
    return cache_chunks