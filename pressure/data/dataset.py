import json
from collections import OrderedDict
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from pressure.util import *
from pressure.util.util import extract_idx
from pressure.data.data_support import ContactMapConfig, PressureMapProcessor 
    

class PSUTMM100_Temporal_LOSO_Chunked(Dataset):
    """
    LOSO PSUTMM100 dataset with hierarchical shuffling and limited caching.
    """
    def __init__(self, chunk_dir, subject=1, split='train', chunk_size=5000, normalization='max',
                 files=None, shuffle=True, sequence_length=5, transform=None,
                 ordinary_movement=False, active_only=False, cfg=None, om_idx=None,
                 max_cached_chunks=5):
        self.chunk_dir = chunk_dir
        self.chunk_size = chunk_size
        self.normalization = normalization
        self.transform = transform
        self.shuffle = shuffle
        self.sequence_length = sequence_length
        self.subject = subject
        self.ordinary_movement = ordinary_movement
        self.split = split
        self.mode = cfg.default.mode
        self.active_only = active_only
        self.max_cached_chunks = max_cached_chunks

        # -------------------- Load metadata --------------------
        if ordinary_movement:
            if om_idx is None:
                om_idx = extract_idx(chunk_dir)
            metadata = json.load(open(f"{chunk_dir}/OM{om_idx}/OM_{om_idx}_metadata.json", 'r'))
            self.metadata = [item for item in metadata]
            self.chunk_files = [item['file_path'] for item in metadata]
        else:
            metadata = json.load(open(f"{chunk_dir}/subject_{subject}_metadata.json", 'r')) if split == 'test' \
                else [json.load(open(f"{chunk_dir}/subject_{i}_metadata.json", 'r')) for i in range(1, 11) if i != subject]

            if split != 'test':
                metadata = [item for sublist in metadata for item in sublist]

            for chunk_data in metadata:
                file_path_split = chunk_data['file_path'].split('/subject')
                metadata_chunk_dir = file_path_split[0]
                subject_num = file_path_split[1][1]
                if metadata_chunk_dir != chunk_dir:
                    raise NotImplementedError(
                        f"File path in subject {subject_num}'s metadata does not match parent directory."
                    )

            if files is not None:
                self.chunk_files = files
                self.metadata = [item for item in metadata if item['file_path'] in files]
            else:
                raise NotImplementedError

        # -------------------- Internal state --------------------
        self.loaded_files = set()
        self.chunk_cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        self.evictions = 0

        # -------------------- Contact / COM settings --------------------
        if cfg.data.gt_com and 'com' not in self.mode and split == 'test':
            self.ret_com = True
        elif cfg.data.gt_com and 'com' in self.mode:
            self.ret_com = True
        elif 'com' in self.mode:
            self.ret_com = True
        else:
            self.ret_com = False

        # -------------------- Shuffle setup --------------------
        self.sequence_starts = self.generate_sequence_starts()
        self._prepare_shuffling()

        if self.active_only:
            foot_mask = np.load('assets/foot_mask_nans.npy')  # (60,21,2)
            self.active_indices = np.where(foot_mask.flatten() == 1)[0]

        if 'contact' in self.mode:
            contact_config = ContactMapConfig(
                use_regions=cfg.data.use_regions,
                contact_threshold=cfg.data.contact_threshold,
                num_regions=cfg.data.num_regions,
                active_only=self.active_only,
                binary_contact=cfg.data.binary_contact
            )
            self.processor = PressureMapProcessor(contact_config)
            
    def _prepare_shuffling(self):
        """Implements hierarchical shuffling: shuffle chunks, then samples inside each chunk."""
        # Compute per-chunk sizes
        self.chunk_sizes = [m['chunk_size'] for m in self.metadata]
        self.chunk_offsets = np.cumsum([0] + self.chunk_sizes[:-1]).tolist()

        # Shuffle chunks and per-chunk sample order
        self.chunk_order = list(range(len(self.chunk_files)))
        if self.shuffle:
            random.shuffle(self.chunk_order)

        self.local_indices = []
        for size in self.chunk_sizes:
            idxs = list(range(size))
            if self.shuffle:
                random.shuffle(idxs)
            self.local_indices.append(idxs)

        # Map from global sample index to (chunk_idx, local_idx)
        self.sample_map = []
        for global_chunk_idx in self.chunk_order:
            base = self.chunk_offsets[global_chunk_idx]
            for local_idx in self.local_indices[global_chunk_idx]:
                self.sample_map.append((global_chunk_idx, local_idx + base))

    def __len__(self):
        return sum(item['chunk_size'] for item in self.metadata)

    def generate_sequence_starts(self):
        return list(range(self.__len__()))

    # ------------------------------------------------------------------
    # Chunk loading (LRU cache)
    # ------------------------------------------------------------------
    def load_chunk(self, chunk_idx):
        if chunk_idx in self.chunk_cache:
            self.cache_hits += 1
            self.chunk_cache.move_to_end(chunk_idx)
            return self.chunk_cache[chunk_idx]

        self.cache_misses += 1
        with open(self.chunk_files[chunk_idx], 'rb') as f:
            chunk = pickle.load(f)
        if self.normalization:
            chunk = self.normalize(chunk)

        self.chunk_cache[chunk_idx] = chunk
        if len(self.chunk_cache) > self.max_cached_chunks:
            self.chunk_cache.popitem(last=False)
            self.evictions += 1
        return chunk

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        half_seq = (self.sequence_length - 1) // 2
        # Get chunk and local index based on shuffled mapping
        if self.shuffle:
            chunk_idx, global_offset = self.sample_map[idx]
        else:
            chunk_idx = idx // self.chunk_size
            global_offset = idx

        chunk = self.load_chunk(chunk_idx)
        idx_in_chunk = global_offset - self.chunk_offsets[chunk_idx]

        # Build temporal sequence centered at idx_in_chunk
        frames, press, coms = [], [], []
        for offset in range(-half_seq, half_seq + 1):
            local_idx = np.clip(idx_in_chunk + offset, 0, len(chunk) - 1)
            sample = chunk[local_idx]

            joint, pressure, com = sample['joint'], sample['pressure'], sample['com']
            if self.transform:
                joint = self.transform(joint)
                pressure = self.transform(pressure)
                if com.ndim == 1:
                    com = np.expand_dims(com, axis=0)
                com = self.transform(com).squeeze(0)
                if pressure.dim() == 3:
                    pressure = pressure.permute(1, 2, 0)

            if self.active_only:
                pressure = pressure.reshape(-1)[self.active_indices]

            frames.append(joint)
            press.append(pressure)
            coms.append(com)

        # Stack sequence tensors
        joints = torch.stack(frames)
        press = torch.stack(press)
        coms = torch.stack(coms)
        mid = len(frames) // 2
        return self.create_return(joints, press[mid], coms[mid].squeeze(0), joints[mid].squeeze(0))

    def create_return(self, joints, pressure, com, middle_frame_joints):
        result = {'joint': joints, 'middle_frame_joints': middle_frame_joints}
        if 'pressure' in self.mode:
            result['pressure'] = pressure
        if 'contact' in self.mode:
            processed = self.processor.process_pressure_map(pressure)
            result['contact'] = processed['contact']
        if 'com' in self.mode or self.ret_com:
            result['com'] = com
        return result

    def get_cache_stats(self):
        total = self.cache_hits + self.cache_misses
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'evictions': self.evictions,
            'hit_rate': self.cache_hits / total if total > 0 else 0.0,
            'cache_size': len(self.chunk_cache)
        }
     
    def normalize(self, data):
        joint_data = np.array([sample[0] for sample in data])
        pressure_data = np.array([sample[1] for sample in data])
        pressure_data = np.nan_to_num(pressure_data)
        if self.normalization == 'z':
            mean = np.mean(pressure_data)
            std = np.std(pressure_data)
            np.subtract(pressure_data, mean, out=pressure_data)
            np.divide(pressure_data, std, out=pressure_data)
        elif self.normalization == 'max':
            np.divide(pressure_data, self.max_pressure, out=pressure_data)
        elif self.normalization == 'minmax':
            pressure_data  = (pressure_data - np.min(pressure_data)) / (np.max(pressure_data) - np.min(pressure_data))
            pressure_data[pressure_data == np.inf] = 0
            pressure_data = np.nan_to_num(pressure_data)

        normalized_chunk = list(zip(joint_data, pressure_data))
        return normalized_chunk 
     