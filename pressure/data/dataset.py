import json
from collections import OrderedDict
import pickle
import random
import re
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from pressure.util import *
from pressure.util.util import extract_idx
from pressure.data.data_support import ContactMapConfig, PressureMapProcessor 


class UnderPressureTemporalDataset(Dataset):
    """
    Temporal UnderPressure loader for a 30fps pose -> 30fps force baseline.

    The official UnderPressure preprocessing stores one sequence per ``.pth`` file
    with ``forces`` targets. Some local conversions also store precomputed
    ``positions``. This loader prefers precomputed positions, and falls back to
    the official UnderPressure FK code when the dataset root contains anim.py.
    """
    def __init__(self, root_dir, split='train', cfg=None, sequence_length=9,
                 train_val_split=0.9, shuffle=True, transform=None,
                 max_cached_sequences=8):
        self.root_dir = Path(root_dir)
        self.split = split
        self.cfg = cfg
        self.sequence_length = sequence_length
        self.shuffle = shuffle
        self.transform = transform
        self.max_cached_sequences = max_cached_sequences
        self.sequence_cache = OrderedDict()

        data_cfg = cfg.data
        self.pose_key = getattr(data_cfg, 'pose_key', 'positions')
        self.target_key = getattr(data_cfg, 'target_key', 'forces')
        self.source_fps = int(getattr(data_cfg, 'source_fps', 100))
        self.input_fps = int(getattr(data_cfg, 'input_fps', 30))
        self.target_fps = int(getattr(data_cfg, 'target_fps', 30))
        self.input_stride = self._fps_stride(self.source_fps, self.input_fps)
        self.target_stride = self._fps_stride(self.source_fps, self.target_fps)
        self.add_confidence = bool(getattr(data_cfg, 'add_confidence', True))
        self.normalize_pose = bool(getattr(data_cfg, 'normalize_pose', True))
        self.normalize_force_by_weight = bool(getattr(data_cfg, 'normalize_force_by_weight', False))

        self.files = self._select_files(train_val_split)
        if not self.files:
            raise FileNotFoundError(
                f"No UnderPressure sequence files found for split='{split}' under {self.root_dir}"
            )

        self.sequence_lengths = [self._sequence_length(path) for path in self.files]
        self.index = self._build_index()
        if self.shuffle:
            random.shuffle(self.index)

    @staticmethod
    def _fps_stride(source_fps, wanted_fps):
        return max(1.0, float(source_fps) / float(wanted_fps))

    def _all_sequence_files(self):
        extensions = ('*.pth', '*.pt', '*.pkl', '*.pickle', '*.npz', '*.npy')
        files = []
        for pattern in extensions:
            files.extend(self.root_dir.rglob(pattern))
        ignored = {'pretrained.tar', 'geo_insoles.pth', 'geo_insole_cells.pth'}
        return sorted([f for f in files if f.name not in ignored])

    def _subject_id(self, path):
        for part in path.parts:
            match = re.fullmatch(r'S\d+', part)
            if match:
                return match.group(0)
        match = re.search(r'(S\d+)', path.stem)
        return match.group(1) if match else None

    def _select_files(self, train_val_split):
        all_files = self._all_sequence_files()
        split_file = getattr(self.cfg.data, 'split_file', None)
        if split_file:
            with open(split_file, 'r') as f:
                splits = json.load(f)
            selected = set(splits[self.split])
            return [p for p in all_files if p.name in selected or p.as_posix() in selected]

        test_subjects = set(getattr(self.cfg.data, 'test_subjects', ['S8', 'S9', 'S10']))
        train_val = [p for p in all_files if self._subject_id(p) not in test_subjects]
        test = [p for p in all_files if self._subject_id(p) in test_subjects]

        rng = random.Random(getattr(self.cfg.default, 'seed', 0))
        rng.shuffle(train_val)
        split_idx = int(len(train_val) * train_val_split)
        if self.split == 'train':
            return sorted(train_val[:split_idx])
        if self.split == 'val':
            return sorted(train_val[split_idx:])
        if self.split == 'test':
            return sorted(test)
        raise ValueError(f"Unsupported split: {self.split}")

    def _load_file(self, path):
        suffix = path.suffix.lower()
        if suffix in {'.pth', '.pt'}:
            return torch.load(path, map_location='cpu', weights_only=False)
        if suffix in {'.pkl', '.pickle'}:
            with open(path, 'rb') as f:
                return pickle.load(f)
        if suffix == '.npz':
            return dict(np.load(path, allow_pickle=True))
        if suffix == '.npy':
            loaded = np.load(path, allow_pickle=True)
            return loaded.item() if loaded.shape == () else {self.pose_key: loaded}
        raise ValueError(f"Unsupported file type: {path}")

    def _sequence_length(self, path):
        item = self._load_file(path)
        if self.pose_key in item:
            return len(item[self.pose_key])
        if 'positions' in item:
            return len(item['positions'])
        if 'angles' in item:
            return len(item['angles'])
        if self.target_key in item:
            return len(item[self.target_key])
        raise KeyError(f"Could not infer sequence length from {path}")

    def _build_index(self):
        half = (self.sequence_length - 1) // 2
        radius = int(np.ceil(half * self.input_stride))
        index = []
        for seq_idx, nframes in enumerate(self.sequence_lengths):
            if nframes <= 0:
                continue
            centers = np.rint(np.arange(0, nframes, self.target_stride)).astype(int)
            centers = np.unique(np.clip(centers, 0, nframes - 1))
            if not getattr(self.cfg.data, 'pad_edges', True):
                centers = centers[(centers >= radius) & (centers < nframes - radius)]
            index.extend((seq_idx, center) for center in centers)
        return index

    def _load_sequence(self, seq_idx):
        if seq_idx in self.sequence_cache:
            self.sequence_cache.move_to_end(seq_idx)
            return self.sequence_cache[seq_idx]

        raw = self._load_file(self.files[seq_idx])
        sequence = self._prepare_sequence(raw, self.files[seq_idx])
        self.sequence_cache[seq_idx] = sequence
        if len(self.sequence_cache) > self.max_cached_sequences:
            self.sequence_cache.popitem(last=False)
        return sequence

    def _prepare_sequence(self, item, path):
        joints = self._extract_positions(item, path).float()
        target = torch.as_tensor(item[self.target_key]).float()

        if self.normalize_pose:
            joints = joints - joints[:, :1, :]
            scale = joints.flatten(1).std(dim=1).median().clamp_min(1e-6)
            joints = joints / scale

        if self.add_confidence and joints.shape[-1] == 3:
            conf = torch.ones(*joints.shape[:-1], 1, dtype=joints.dtype)
            joints = torch.cat([joints, conf], dim=-1)

        if self.normalize_force_by_weight and 'subject' in item:
            weight = torch.as_tensor(item['subject'].weight).float()
            target = target / weight.clamp_min(1e-6)

        return {
            'joint': torch.nan_to_num(joints),
            'pressure': torch.nan_to_num(target),
        }

    def _extract_positions(self, item, path):
        for key in (self.pose_key, 'positions', 'joints', 'joint'):
            if key in item:
                return torch.as_tensor(item[key])

        required = {'angles', 'skeleton', 'trajectory'}
        if required.issubset(item.keys()):
            repo_root = self._find_underpressure_repo(path)
            if repo_root is not None and str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            try:
                import anim
                from data import TOPOLOGY
                return anim.FK(item['angles'], item['skeleton'], item['trajectory'], TOPOLOGY)
            except Exception as exc:
                raise RuntimeError(
                    "This UnderPressure file stores angles/skeleton/trajectory but no positions. "
                    "Put the official UnderPressure repo root on cfg.data.underpressure_repo_path, "
                    "or precompute positions into each .pth file."
                ) from exc

        raise KeyError(f"No pose data found in {path}")

    def _find_underpressure_repo(self, path):
        configured = getattr(self.cfg.data, 'underpressure_repo_path', None)
        if configured:
            return Path(configured)
        for parent in [path.parent, *path.parents]:
            if (parent / 'anim.py').exists() and (parent / 'data.py').exists():
                return parent
        return None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        seq_idx, center = self.index[idx]
        sequence = self._load_sequence(seq_idx)
        joints = sequence['joint']
        pressure = sequence['pressure']

        half = (self.sequence_length - 1) // 2
        frame_indices = [
            int(np.clip(round(center + offset * self.input_stride), 0, len(joints) - 1))
            for offset in range(-half, half + 1)
        ]
        target_idx = int(np.clip(center, 0, len(pressure) - 1))
        joint_seq = joints[frame_indices]
        target = pressure[target_idx].reshape(-1)
        return {
            'joint': joint_seq,
            'middle_frame_joints': joint_seq[half],
            'pressure': target,
        }
    

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
     
