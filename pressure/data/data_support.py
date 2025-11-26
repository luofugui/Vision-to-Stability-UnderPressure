import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from copy import deepcopy
import pickle
import os
import json
import numpy as np
import torch

def mirror_contact_regions(contact, num_regions):
    """
    Args:
        contact: numpy array of shape (N, R), R=2*num_regions (flattened contact)
        num_regions: number of regions per foot (e.g., 4 here)
        flip_left_right: whether to flip foot assignments
    """
    rows, cols = num_regions
    R = rows * cols
    if contact.shape[1] != 2 * R:
        raise ValueError(f"Expected contact shape (N, {2*R}) for num_regions={num_regions}, got {contact.shape}")

    left = contact[:, :R].reshape(-1, rows, cols)
    right = contact[:, R:].reshape(-1, rows, cols)

    # Horizontally mirror each foot
    left_flipped = np.flip(left, axis=2)
    right_flipped = np.flip(right, axis=2)

    # Swap left and right
    flipped = np.concatenate([right_flipped.reshape(-1, R), left_flipped.reshape(-1, R)], axis=1)
    return flipped


def visualize_contact_regions(cfg):
    region_map = np.full(cfg.foot_mask.shape, np.nan)

    # Fill in region indices from pixel_to_region
    for ch in [0, 1]:  # 0 = left, 1 = right
        valid_mask = ~np.isnan(cfg.foot_mask[..., ch])
        region_map[..., ch][valid_mask] = cfg.pixel_to_region[..., ch][valid_mask]

    num_regions = cfg.num_regions[0] * cfg.num_regions[1] * 2
    cmap = ListedColormap(plt.get_cmap("tab20").colors[:num_regions])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    titles = ['Left Foot', 'Right Foot']

    for ch, ax in enumerate(axes):
        img = region_map[..., ch]
        ax.imshow(img, cmap=cmap, vmin=0, vmax=num_regions - 1, interpolation='nearest')
        ax.set_title(titles[ch])
        ax.axis('off')

        # For each region, annotate its approximate center
        for region_id in range(num_regions):
            mask = img == region_id
            if np.any(mask):
                coords = np.column_stack(np.where(mask))
                center = coords.mean(axis=0)
                ax.text(center[1], center[0], str(region_id),
                        color='black', fontsize=8, ha='center', va='center',
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.2'))

    plt.suptitle("Region Map with Region IDs Overlaid")
    plt.tight_layout()
    plt.show()
    
class ContactMapConfig:
    def __init__(self, 
                 use_regions: bool = True,
                 contact_threshold: float = 0.03,
                 num_regions: tuple = (5, 5),
                 active_only: bool = False,
                 binary_contact: bool = True):
        self.use_regions = use_regions
        self.contact_threshold = contact_threshold
        self.num_regions = num_regions
        self.active_only = active_only
        self.binary_contact = binary_contact
        
        # Load foot mask and identify active pixels
        self.foot_mask = np.load('assets/foot_mask_nans.npy')  # Shape: (60, 21, 2)
        self.active_indices = np.where(~np.isnan(self.foot_mask.flatten()))[0]
        
        # Create mapping from full indices to active indices
        self.full_to_active = {full_idx: active_idx 
                              for active_idx, full_idx in enumerate(self.active_indices)}
        
        # Get original coordinates of active pixels
        self.active_coords = np.unravel_index(self.active_indices, (60, 21, 2))
        self.active_rows = self.active_coords[0]
        self.active_cols = self.active_coords[1]
        self.channel_idx = self.active_coords[2]  # channel index (0 or 1)
        
        # Now properly identify left and right foot pixels based on channel
        self.left_foot_mask = self.channel_idx == 0
        self.right_foot_mask = self.channel_idx == 1
        
        # Calculate output dimension
        if use_regions:
            self.output_dim = num_regions[0] * num_regions[1] * 2
            self._setup_region_mapping()
            self._create_region_lookup()
        else:
            self.output_dim = len(self.active_indices)

    def _setup_region_mapping(self):
        """Setup region assignments for active pixels."""
        regions_per_foot = self.num_regions[0] * self.num_regions[1]
        self.pixel_regions = np.zeros(len(self.active_indices), dtype=int)
        
        # Process each foot separately
        for foot_idx, foot_mask in enumerate([self.left_foot_mask, self.right_foot_mask]):
            if np.any(foot_mask):
                self._map_foot_regions(foot_mask, foot_idx, regions_per_foot)
    
    def _map_foot_regions(self, foot_mask, foot_idx, regions_per_foot):
        """Map regions for one foot's active pixels."""
        foot_rows = self.active_rows[foot_mask]
        foot_cols = self.active_cols[foot_mask]
        
        bounds = {
            'min_row': np.min(foot_rows),
            'max_row': np.max(foot_rows),
            'min_col': np.min(foot_cols),
            'max_col': np.max(foot_cols)
        }
        
        # Calculate region sizes
        row_size = (bounds['max_row'] - bounds['min_row']) / self.num_regions[0]
        col_size = (bounds['max_col'] - bounds['min_col']) / self.num_regions[1]
        
        # Assign regions to active pixels
        region_rows = np.floor((foot_rows - bounds['min_row']) / row_size).clip(0, self.num_regions[0] - 1)
        region_cols = np.floor((foot_cols - bounds['min_col']) / col_size).clip(0, self.num_regions[1] - 1)
        
        # Calculate region indices
        region_indices = (region_rows * self.num_regions[1] + region_cols).astype(int)
        region_indices += foot_idx * regions_per_foot
       
        # Store region assignments for active pixels only
        active_indices = np.where(foot_mask)[0]
        self.pixel_regions[active_indices] = region_indices
        
    def _create_region_lookup(self):
        """Create lookup tables for mapping between regions and pixels."""
        # Create mappings between regions and their active pixels
        self.region_to_pixels = {}
        self.pixel_to_region = np.zeros(np.prod(self.foot_mask.shape), dtype=int)
        self.pixel_to_region.fill(-1)  # Initialize with -1 for invalid pixels
        
        # Fill mappings for each region using active indices
        for region in range(self.output_dim):
            region_mask = (self.pixel_regions == region)
            # Map to positions in active-only array
            active_pixel_positions = np.where(region_mask)[0]
            self.region_to_pixels[region] = active_pixel_positions
            
            # Map to positions in full array for reconstruction
            full_pixel_indices = self.active_indices[region_mask]
            self.pixel_to_region[full_pixel_indices] = region
        
        # Reshape pixel_to_region map to match foot mask shape
        self.pixel_to_region = self.pixel_to_region.reshape(self.foot_mask.shape)

class PressureMapProcessor:
    def __init__(self, config: ContactMapConfig):
        self.config = config
        
    def create_contact_mask(self, pressure: torch.Tensor) -> torch.Tensor:
        """Creates contact mask from pressure values.
        Returns binary mask if binary_contact=True, otherwise returns sigmoid probabilities."""
        if self.config.binary_contact:
            return (pressure > self.config.contact_threshold).float()
        else:
            scale_factor = 1.0 / (self.config.contact_threshold * 0.5) 
            # Center around threshold and scale
            centered_pressure = (pressure - self.config.contact_threshold) * scale_factor
            contact_prob = torch.sigmoid(centered_pressure)
            # Add small epsilon to avoid numerical issues
            contact_prob = torch.clamp(contact_prob, min=1e-6, max=1-1e-6)
            return contact_prob
    
    def remove_extra_pixels(self, pressure: torch.Tensor) -> torch.Tensor:
        """Remove nan pixels from pressure map."""
        return pressure.reshape(-1)[self.config.active_indices]
    
    def process_pressure_map(self, pressure) -> dict:
        """
        Process pressure map to create contact map, ensuring proper region orientation.

        Args:
            pressure: Flattened array of active pixel pressures or full pressure map
            Can be either numpy array or torch tensor
        """
        # Convert input to torch tensor if it's a numpy array
        is_numpy = isinstance(pressure, np.ndarray)
        if is_numpy:
            pressure = torch.from_numpy(pressure)
        if not self.config.use_regions:
            contact = self.create_contact_mask(pressure)
            return {'pressure': pressure, 'contact': contact}

        # Convert full pressure map to active-only if needed
        if len(pressure.shape) > 1 and pressure.shape[-1] == 2:
            pressure = self.remove_extra_pixels(pressure)

        # Initialize regional contact map
        device = pressure.device if hasattr(pressure, 'device') else 'cpu'
        regions = torch.zeros(self.config.output_dim, device=device)
        regions_per_foot = self.config.num_regions[0] * self.config.num_regions[1]

        # Process each foot separately
        for foot_idx in [0, 1]:  # 0 = left foot, 1 = right foot
            foot_start = foot_idx * regions_per_foot
            foot_end = foot_start + regions_per_foot

            # Get indices for this foot
            foot_mask = torch.from_numpy(self.config.channel_idx == foot_idx)
            foot_pixels = pressure[foot_mask]

            # Process each region for this foot
            for region_idx in range(regions_per_foot):
                global_region_idx = foot_start + region_idx
                region_pixels_mask = torch.from_numpy(self.config.pixel_regions[foot_mask] == global_region_idx)

                if region_pixels_mask.any():
                    region_pressure = foot_pixels[region_pixels_mask]
                    max_pressure = region_pressure.max()

                    if self.config.binary_contact:
                        regions[global_region_idx] = (max_pressure > self.config.contact_threshold).float()
                    else:
                        scale_factor = 1.0 / (self.config.contact_threshold * 0.5)
                        centered_pressure = (max_pressure - self.config.contact_threshold) * scale_factor
                        regions[global_region_idx] = torch.sigmoid(centered_pressure)
                        regions[global_region_idx] = torch.clamp(regions[global_region_idx], min=1e-6, max=1-1e-6)

        # Convert back to numpy if input was numpy
        if is_numpy:
            pressure = pressure.numpy()
            regions = regions.numpy()

        return {
            'pressure': pressure,
            'contact': regions
        }
        
    def reconstruct_full_contact(self, contact_map, include_nans=True, flip_left_right=False):
        """
        Reconstructs full contact map from either active-only or regional contact map.
        Uses existing ContactMapConfig settings for consistent mapping.
        
        Args:
            contact_map: Array of shape (N, num_regions) or (N, num_active_pixels)
            include_nans: Whether to include NaN values in output
            
        Returns:
            Array of shape (N, 60, 21, 2)
        """
        if contact_map.ndim == 1:
            contact_map = contact_map.reshape(1, -1)
            
        num_frames = len(contact_map)
        full_shape = (num_frames, *self.config.foot_mask.shape)
     
        if self.config.use_regions:
            full_contact = np.zeros(full_shape)
            regions_per_foot = self.config.num_regions[0] * self.config.num_regions[1]
            
            # Process each foot separately
            for foot_idx in [0, 1]:  # 0 = left foot, 1 = right foot
                foot_start = foot_idx * regions_per_foot
                foot_end = foot_start + regions_per_foot
                
                # Get foot-specific regions and mask
                foot_regions = contact_map[:, foot_start:foot_end]
                foot_mask_slice = self.config.foot_mask[..., foot_idx]
                
                # Reshape regions into spatial layout (rows x cols)
                foot_regions = foot_regions.reshape(-1, self.config.num_regions[0], self.config.num_regions[1])
                
                # For right foot, we need to flip the regions horizontally 
                # to match pressure data orientation
                if foot_idx == 1:
                    foot_regions = np.flip(foot_regions, axis=2)
                
                # Map regional values back to pixels
                for i in range(num_frames):
                    frame = np.zeros_like(foot_mask_slice)
                    valid_mask = self.config.pixel_to_region[..., foot_idx] != -1
                    region_indices = self.config.pixel_to_region[valid_mask, foot_idx]
                    
                    # Convert global region indices to foot-relative indices
                    foot_relative_indices = region_indices - (foot_idx * regions_per_foot)
                    
                    # Ensure indices are valid
                    valid_region_mask = (foot_relative_indices >= 0) & (foot_relative_indices < regions_per_foot)
                    if not np.all(valid_region_mask):
                        print(f"Warning: Found invalid region indices for foot {foot_idx}")
                        print(f"Invalid indices: {foot_relative_indices[~valid_region_mask]}")
                    
                    frame[valid_mask] = foot_regions[i].flatten()[foot_relative_indices]
                    full_contact[i, ..., foot_idx] = frame
        else:
            # Handle active-only contact maps
            full_contact = np.zeros(full_shape)
            for i in range(num_frames):
                frame = np.zeros(np.prod(self.config.foot_mask.shape))
                frame[self.config.active_indices] = contact_map[i]
                full_contact[i] = frame.reshape(self.config.foot_mask.shape)
        
        if include_nans:
            # Broadcast NaN mask across all frames
            nan_mask = np.isnan(self.config.foot_mask)
            full_contact[..., nan_mask] = np.nan
            
        return full_contact


    def reconstruct_full_pressure(self, active_pressure, include_nans=True):
        """
        Reconstructs full pressure map from active-only pressure map using existing config.
        
        Args:
            active_pressure: Array of shape (N, num_active_pixels)
            include_nans: Whether to include NaN values in output
            
        Returns:
            Array of shape (N, 60, 21, 2)
        """
        if active_pressure.ndim == 1:
            active_pressure = active_pressure.reshape(1, -1)
            
        num_frames = len(active_pressure)
        full_shape = (num_frames, *self.config.foot_mask.shape)
        
        # Create empty array for full pressure maps
        full_pressure = np.zeros(full_shape)
        
        # Reshape active pressure to match full pressure format
        for i in range(num_frames):
            frame = np.zeros(np.prod(self.config.foot_mask.shape))
            frame[self.config.active_indices] = active_pressure[i]
            full_pressure[i] = frame.reshape(self.config.foot_mask.shape)
        
        if include_nans:
            # Broadcast NaN mask across all frames
            nan_mask = np.isnan(self.config.foot_mask)
            full_pressure[..., nan_mask] = np.nan
            
        return full_pressure

class DataNormalizer:
    def __init__(self, data_path, cfg=None):
        self.data_path = data_path
        self.is_distribution = False
        self.load_normalization_info()
        self.load_data_args()
        if cfg is not None:
            self.data_type = 'OM' if cfg.default.om else 'subject'
        
        if cfg is not None:
            pressure_config = ContactMapConfig(
                use_regions=cfg.data.use_regions,
                contact_threshold=cfg.data.contact_threshold,
                num_regions=cfg.data.num_regions,
                active_only=cfg.data.active_only,
                binary_contact=cfg.data.binary_contact
            )
            self.processor = PressureMapProcessor(pressure_config)
                

    def load_normalization_info(self):
        with open(os.path.join(self.data_path, 'normalization_info.pkl'), 'rb') as f:
            self.norm_info = pickle.load(f)

    def load_data_args(self):
        with open(os.path.join(self.data_path, 'args.json'), 'r') as f:
            self.data_args = json.load(f)
            self.is_distribution = self.data_args.get('make_pressure_distribution', False)

    def denormalize_joints(self, normalized_joints, data_id):
        """Denormalize joint data while handling both normalization types and rotation."""
        id = self.norm_info[f'{self.data_type}_ids'].index(data_id)
        norm_info = self.norm_info['normalization_info'][id]
        
        original_shape = normalized_joints.shape
        joint_data = normalized_joints[..., :-1]
        confidence = normalized_joints[..., -1:]
        
        # First undo normalization
        if norm_info['joint']['type'] == 'bone':
            denormalized = joint_data * norm_info['joint']['scale']
        else:
            denormalized = joint_data * norm_info['joint']['std'] + norm_info['joint']['mean']
        
        # Undo rotation if it was applied
        if 'rotation' in norm_info and norm_info['rotation']['applied']:
            inv_rotation = np.transpose(norm_info['rotation']['matrix'], (0, 2, 1))
            denormalized = np.einsum('bij,bkj->bki', inv_rotation, denormalized)
            
        # Recombine with confidence values
        denormalized = np.concatenate([denormalized, confidence], axis=-1)
        return denormalized.reshape(original_shape)

    def denormalize_com(self, normalized_com, data_id):
        """Denormalize CoM data while handling both normalization types and rotation."""
        id = self.norm_info[f'{self.data_type}_ids'].index(data_id)
        norm_info = self.norm_info['normalization_info'][id]
        
        original_shape = normalized_com.shape
        com_data = normalized_com[..., :-1]
        confidence = normalized_com[..., -1:]
        
        if norm_info['com']['type'] == 'zscore_gt' or norm_info['com']['type'] == 'zscore':
            com_mean = norm_info['com']['mean']
            com_std  = norm_info['com']['std']
            com_denorm = com_data * com_std + com_mean
        elif norm_info['com']['type'] == 'bone':
            scale = norm_info['com']['scale']
            com_denorm = com_data * scale
        # Undo rotation if it was applied
        if 'rotation' in norm_info and norm_info['rotation']['applied']:
            inv_rotation = np.transpose(norm_info['rotation']['matrix'], (0, 2, 1))
            denormalized = np.einsum('bij,bj->bi', inv_rotation, com_denorm)
        
        # Recombine with confidence values
        denormalized = np.concatenate([com_denorm, confidence], axis=-1)
        return denormalized.reshape(original_shape)
    
    def verify_consistency(self, cfg):
        """Verify that the data normalization matches the config settings."""
        if 'pressure' in cfg.default.mode:
            if cfg.loss.pressure_loss == 'kld':
                assert cfg.data.pressure_is_distribution, 'KLD loss requires pressure data to be distributions'
                assert self.data_args.get('make_pressure_distribution', False), 'Data was not created as distributions'
            else:
                # Only check normalization consistency for non-distribution data
                if not cfg.data.pressure_is_distribution:
                    assert self.data_args['max_norm'] == cfg.data.subject_wise_max_norm, \
                        'Inconsistent max normalization'
                    assert self.data_args['weight_norm'] == cfg.data.subject_wise_weight_norm, \
                        'Inconsistent weight normalization'
                                    
    def unnormalize_and_scale_to_kpa(self, normalized_pressure, data_id, cfg, reconstruct_full=True):
        """
        Convert normalized pressure back to kPa values.
        
        Args:
            normalized_pressure: Normalized pressure data
            subject: Subject ID
            cfg: Configuration object
            
        Returns:
            Pressure data in kPa
        """
        id = self.norm_info[f'{self.data_type}_ids'].index(data_id)
        norm_info = self.norm_info['normalization_info'][id]
      
        if self.is_distribution:
            original_sums = norm_info['pressure']['original_sums']
            # Reshape original_sums to match the batch dimension
            if len(normalized_pressure.shape) > 1:
                original_sums = original_sums.reshape(-1, *[1] * (len(normalized_pressure.shape) - 1))
            
            # First convert from distribution back to normalized pressure
            unnormalized = normalized_pressure * original_sums
            
            # Then apply weight scaling if it was originally applied
            if norm_info['pressure']['weight_normalized']:
                weight = norm_info['pressure']['weight']
                unnormalized *= weight
        else:
            # Original normalization logic
            max_pressures = deepcopy(self.norm_info['max_pressures'])
            weight = self.norm_info['weights'][id]
            scaling_factor = 1.0

            if cfg.data.subject_wise_max_norm:
                max_pressure = max_pressures[id]
                if isinstance(max_pressure, np.ndarray):
                    max_pressure = np.max(max_pressure)
                scaling_factor *= max_pressure
            else:
                del max_pressures[id]
                max_pressure = np.max(max_pressures)
                scaling_factor *= max_pressure

            if cfg.data.subject_wise_weight_norm:
                scaling_factor *= weight

            unnormalized = normalized_pressure * scaling_factor
            
        unnormalized[unnormalized < 0] = 0
        unnormalized = np.nan_to_num(unnormalized)
        
        if reconstruct_full:
            unnormalized = self.processor.reconstruct_full_pressure(unnormalized)
        
        return unnormalized
    
    def unnormalize_contact(self, contact_data, subject, cfg, reconstruct_full=True, apply_sigmoid=False):
        """
        Process contact data back to full format if needed.
        
        Args:
            contact_data: Normalized contact map (either regional or active-only)
            subject: Subject ID
            cfg: Configuration object
            reconstruct_full: Whether to reconstruct full contact map
            
        Returns:
            Contact data in original format
        """
        if apply_sigmoid:
            contact_data = 1/(1 + np.exp(-contact_data))
             
        
        if not reconstruct_full:
            return contact_data
        
        # Handle regional vs active-only contact maps
        return self.processor.reconstruct_full_contact(contact_data)
    
    def get_subject_weight(self, data_id):
        """Get the weight of a subject."""
        subject_id = self.norm_info[f'{self.data_type}_ids'].index(data_id)
        return self.norm_info['weights'][subject_id]

    def get_origin_index(self, subject):
        """Get the origin index used for centering data."""
        subject_id = self.norm_info['subject_ids'].index(subject)
        return self.norm_info['normalization_info'][subject_id]['origin']['idx']