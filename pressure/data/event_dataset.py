import pickle
import random
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from pressure.data.data_support import ContactMapConfig, PressureMapProcessor


def get_event_indices(binary_signal):
    signal = np.asarray(binary_signal).astype(int)
    padded = np.concatenate(([0], signal, [0]))
    diff = np.diff(padded)
    onsets = np.where(diff == 1)[0]
    departures = np.where(diff == -1)[0]
    return onsets, departures


class PSUEventOffsetDataset(Dataset):
    """
    Event-window dataset for contact onset/departure offset regression.

    Each sample is a pose window around a GT contact event. The target is the
    event frame offset relative to the beginning of the window, normalized to
    [0, 1]. Event windows are jittered so the event is not always centered.
    """
    def __init__(
        self,
        chunk_files,
        cfg,
        split="train",
        window_frames=51,
        samples_per_event=1,
        min_event_offset=5,
        max_event_offset=None,
        max_cached_chunks=8,
        seed=0,
    ):
        self.chunk_files = list(chunk_files)
        self.cfg = cfg
        self.split = split
        self.window_frames = int(window_frames)
        self.samples_per_event = int(samples_per_event)
        self.min_event_offset = int(min_event_offset)
        self.max_event_offset = (
            int(max_event_offset)
            if max_event_offset is not None
            else self.window_frames - 1 - self.min_event_offset
        )
        self.max_cached_chunks = int(max_cached_chunks)
        self.chunk_cache = OrderedDict()
        self.rng = random.Random(seed)

        if self.window_frames < 3:
            raise ValueError("window_frames must be at least 3.")
        if self.min_event_offset < 0 or self.max_event_offset >= self.window_frames:
            raise ValueError("Event offset range must fit inside the window.")
        if self.min_event_offset > self.max_event_offset:
            raise ValueError("min_event_offset must be <= max_event_offset.")

        contact_config = ContactMapConfig(
            use_regions=cfg.data.use_regions,
            contact_threshold=cfg.data.contact_threshold,
            num_regions=cfg.data.num_regions,
            active_only=cfg.data.active_only,
            binary_contact=True,
        )
        self.processor = PressureMapProcessor(contact_config)
        self.active_only = bool(cfg.data.active_only)
        self.active_indices = contact_config.active_indices
        self.num_channels = (
            cfg.data.num_regions[0] * cfg.data.num_regions[1] * 2
            if cfg.data.use_regions
            else len(contact_config.active_indices)
        )

        self.events = self._build_events(seed)
        if not self.events:
            raise ValueError(f"No event windows found for split={split}.")

    def _load_chunk(self, chunk_idx):
        if chunk_idx in self.chunk_cache:
            self.chunk_cache.move_to_end(chunk_idx)
            return self.chunk_cache[chunk_idx]

        with open(self.chunk_files[chunk_idx], "rb") as f:
            chunk = pickle.load(f)

        self.chunk_cache[chunk_idx] = chunk
        if len(self.chunk_cache) > self.max_cached_chunks:
            self.chunk_cache.popitem(last=False)
        return chunk

    def _sample_fields(self, sample):
        if isinstance(sample, dict):
            return sample["joint"], sample["pressure"]
        if isinstance(sample, (list, tuple)) and len(sample) >= 2:
            return sample[0], sample[1]
        raise TypeError(f"Unsupported chunk sample type: {type(sample)}")

    def _contact_from_pressure(self, pressure):
        pressure = torch.as_tensor(pressure).float()
        if self.active_only:
            pressure = pressure.reshape(-1)[self.active_indices]
        processed = self.processor.process_pressure_map(pressure)
        return torch.as_tensor(processed["contact"]).float().cpu().numpy()

    def _chunk_contacts(self, chunk):
        contacts = []
        for sample in chunk:
            _, pressure = self._sample_fields(sample)
            contacts.append(self._contact_from_pressure(pressure))
        return np.stack(contacts).astype(np.float32)

    def _valid_offsets(self, event_frame, nframes):
        lo = max(self.min_event_offset, event_frame - (nframes - self.window_frames))
        hi = min(self.max_event_offset, event_frame)
        if lo > hi:
            return []
        return list(range(lo, hi + 1))

    def _build_events(self, seed):
        events = []
        rng = random.Random(seed)

        for chunk_idx in range(len(self.chunk_files)):
            chunk = self._load_chunk(chunk_idx)
            nframes = len(chunk)
            if nframes < self.window_frames:
                continue

            contacts = self._chunk_contacts(chunk)
            for channel in range(contacts.shape[1]):
                onsets, departures = get_event_indices(contacts[:, channel] >= 0.5)
                for event_type, event_frames in [(0, onsets), (1, departures)]:
                    for event_frame in event_frames:
                        valid_offsets = self._valid_offsets(int(event_frame), nframes)
                        if not valid_offsets:
                            continue
                        for _ in range(self.samples_per_event):
                            if self.split == "train":
                                offset = rng.choice(valid_offsets)
                            else:
                                offset = valid_offsets[len(valid_offsets) // 2]
                            events.append(
                                {
                                    "chunk_idx": chunk_idx,
                                    "event_frame": int(event_frame),
                                    "offset": int(offset),
                                    "channel": int(channel),
                                    "event_type": int(event_type),
                                }
                            )

        if self.split == "train":
            rng.shuffle(events)
        return events

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        event = self.events[idx]
        chunk = self._load_chunk(event["chunk_idx"])
        start = event["event_frame"] - event["offset"]
        end = start + self.window_frames

        joints = []
        for frame_idx in range(start, end):
            joint, _ = self._sample_fields(chunk[frame_idx])
            joints.append(torch.as_tensor(joint).float())
        joint_window = torch.stack(joints)

        target_offset = event["offset"] / float(self.window_frames - 1)
        return {
            "joint": joint_window,
            "offset": torch.tensor(target_offset, dtype=torch.float32),
            "event_type": torch.tensor(event["event_type"], dtype=torch.long),
            "channel": torch.tensor(event["channel"], dtype=torch.long),
        }
