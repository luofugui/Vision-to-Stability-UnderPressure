from abc import abstractmethod
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import io
from PIL import Image, ImageDraw, ImageFont
import functools
import cv2
import shutil

from pressure.data.data_support import PressureMapProcessor, ContactMapConfig, mirror_contact_regions
from pressure.util.video import *

### Utility functions for visualization ###
def get_bgr_colors_from_colormap(num_colors, colormap_name='hsv'):
    colormap = cm.get_cmap(colormap_name)
    colors = [colormap(i / num_colors)[:3] for i in range(num_colors)]
    colors_bgr = [(int(b * 255), int(g * 255), int(r * 255)) for r, g, b in colors]
    return colors_bgr

BODY_PARTS = [
        (0, 1), (1, 2), (1, 5),
        (2, 3), (3, 4),
        (5, 6), (6, 7),
        (1, 8), (8, 9), (9, 12),
        (9, 10), (10, 11),  (11, 23), (23, 22),
        (8, 12), (12, 13), (13, 14), (14, 21), (14, 19), (19, 20),
        # Face 
        (1, 0), (0, 15), (0, 16), (15, 17), (16, 18)
    ]

# Convert BGR colors to RGB for matplotlib
colors = get_bgr_colors_from_colormap(len(BODY_PARTS))
rgb_colors = [(r/255, g/255, b/255) for (b, g, r) in colors]

def fig_to_img(fig, output_path=None, dpi=100, target_size=None):
    """Convert matplotlib figure to image array or save to file.
    Returns:
        numpy array of image if output_path is None, otherwise None
    """
    if output_path:
        fig.savefig(output_path, bbox_inches='tight', transparent=True, dpi=dpi)
        plt.close(fig)
        return None
        
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, 
                transparent=True, dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    
    if target_size:
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        
    img_array = np.array(img.convert('RGBA'))
    buf.close()
    return img_array

def with_style(style_attr='plt_style'):  # Default to looking for self.plt_style
    """Decorator to apply matplotlib style context to visualization functions.
    
    Args:
        style_attr: Name of the instance attribute containing the style string
                   (defaults to 'plt_style' to match VisualizationManager)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            style = getattr(self, style_attr)  # Will get self.plt_style by default
            with plt.style.context(style):
                return func(self, *args, **kwargs)
        return wrapper
    return decorator

class BaseVisualizer:
    """Base class for all visualizers"""
    def __init__(self, save_dir, cfg, manager=None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.dpi = cfg.viz.dpi  # For validation/general use
        self.manager = manager
    
    @abstractmethod
    def visualize(self, data, auxiliary_data=None, return_fig=False, dpi=None):
        """Create visualization for a single sample"""
        pass

    def create_grid(self, predictions, targets, save_path, **kwargs):
        """Base grid creation logic for consistent sample selection across modalities"""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        num_samples = kwargs.get('num_samples', self.cfg.viz.collage.samples_per_viz)
        
        # Use provided indices or generate new ones
        if 'indices' in kwargs:
            indices = kwargs['indices']
        else:
            if len(predictions) < num_samples:
                num_samples = len(predictions)
            
            start_idx = kwargs.get('start_idx')
            if start_idx is None:
                start_idx = np.random.randint(0, len(predictions) - num_samples)
            indices = np.arange(start_idx, start_idx + num_samples)
            
        return indices
    
class PressureVisualizer(BaseVisualizer):
    def __init__(self, save_dir, cfg, manager=None, processor=None):
        super().__init__(save_dir, cfg, manager)
        self.cmap = cfg.viz.colormap
        self.view_angle = cfg.viz.figures.pressure.view_angle
        self.foot_mask_with_nans = np.load('assets/foot_mask_nans.npy')
        self.foot_mask = np.where(np.isnan(self.foot_mask_with_nans), 0, 1)
        self.default_silhouette = np.load('assets/default_silhouette.npy')
        self.figsize = (10,8)
        self.processor = processor

    def visualize(self, data, auxiliary_data=None, return_fig=False, dpi=None):
        """Create single pressure visualization.
        Uses existing implementation via _create_3d_visualization.
        """
        fig = self._create_3d_visualization(
            data=data,
            foot_mask=auxiliary_data.get('foot_mask', self.foot_mask) if auxiliary_data else self.foot_mask,
            title=auxiliary_data.get('title', None) if auxiliary_data else None
        )
        if dpi is None:
            dpi = self.dpi
        return fig if return_fig else fig_to_img(fig, dpi=dpi)
   
    def _create_3d_visualization(self, data, foot_mask=None, title=None, figsize=(10, 8)): 
        """Create a single 3D visualization with fixed figure size."""
        # Preprocess data
        processed_data = self._preprocess_pressure_data(data, foot_mask)
        
        silhouette = self.default_silhouette.copy()
            
        # Prepare silhouette
        silhouette[np.isnan(silhouette)] = 0
        silhouette = cv2.resize(silhouette.astype(float), 
                            (processed_data.shape[1], processed_data.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
        
        # Create visualization with fixed size
        fig = plt.figure(figsize=figsize, dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Create meshgrid and plot surface
        xx, yy = np.meshgrid(np.arange(processed_data.shape[1]), 
                            np.arange(processed_data.shape[0]))
        surf = ax.plot_surface(xx, yy, processed_data, rstride=1, cstride=1,
                            cmap=self.cmap, edgecolor='none', alpha=0.8)
        
        # Plot silhouette
        y_indices, x_indices = np.where(silhouette > 0)
        ax.scatter(x_indices, y_indices, np.zeros_like(x_indices), 
                c='black', s=.001)
        
        # Configure view
        ax.view_init(*self.view_angle)
        ax.set_yticklabels([])
        if title:
            ax.set_title(title)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
        
        return fig

    def _preprocess_pressure_data(self, data, foot_mask=None):
        """Preprocess pressure data for visualization."""
        # Reshape if necessary
        if data.shape != foot_mask.shape:
            data = self.processor.reconstruct_full_pressure(data)
            data = data.squeeze(0)
        
        # Apply foot mask if provided
        if foot_mask is not None:
            data = data * foot_mask
            
        # Concatenate and process
        data = np.concatenate([data[:, :, 0], data[:, :, 1]], axis=1)
        data = np.pad(data, (2, 2), 'constant', constant_values=np.nan)
        data = cv2.resize(data, (data.shape[1] * 10, data.shape[0] * 10))
        data = cv2.GaussianBlur(data, (3, 3), 0)
        data[data < 1e-3] = np.nan
        return data
    
    def create_grid(self, predictions, targets, save_path, **kwargs):
        indices = super().create_grid(predictions, targets, save_path, **kwargs)
        
        # Generate visualizations
        gt_images = [self.visualize(targets[i]) for i in indices]
        pred_images = [self.visualize(predictions[i]) for i in indices]
        
        # Create collage using manager's collage function
        self.manager.create_collage(gt_images, pred_images, save_path)
    
class ContactVisualizer(BaseVisualizer):
    def __init__(self, save_dir, cfg, manager=None, processor=None):
        super().__init__(save_dir, cfg, manager)
        self.num_regions = cfg.data.num_regions if cfg.data.use_regions else None
        self.foot_mask_with_nans = np.load('assets/foot_mask_nans.npy')
        self.foot_mask = np.where(np.isnan(self.foot_mask_with_nans), 0, 1)
        self.figsize = (4,8)
        self.processor = processor
        self.mirror = cfg.viz.figures.contact.mirror
        self.use_regions = cfg.data.use_regions

    def visualize(self, data, auxiliary_data, return_fig=False, dpi=None):
        """Create single contact visualization using existing implementation."""
        fig = self._create_contact_visualization(
            contact_map=data,
            num_regions=auxiliary_data.get('num_regions', self.num_regions) if self.use_regions else None,
            show_both_feet=True,
            show_colorbar=auxiliary_data.get('show_colorbar', True),
            is_ground_truth=auxiliary_data.get('is_ground_truth', False),
            )

        if dpi is None:
            dpi = self.dpi
        return fig if return_fig else fig_to_img(fig, dpi=dpi)

    def _create_contact_visualization(self, contact_map, num_regions=None, 
                                      show_both_feet=True, show_colorbar=True, is_ground_truth=False):
        """Create contact map visualization."""
        contact_map = self.processor.reconstruct_full_contact(contact_map)
        contact_map = contact_map.squeeze(0)
            
        # Get mask of valid pixels once
        valid_mask = ~np.isnan(self.foot_mask_with_nans)
       
        # Process values more efficiently
        if is_ground_truth:
            # Single operation: convert to bool then float, then apply mask
            result = (contact_map > 0.5).astype(float)
            result[~valid_mask] = np.nan
        else:
            # Clip values and apply mask in one go
            result = np.clip(contact_map, 0, 1)
            result[~valid_mask] = np.nan
            
        # Create figure
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=self.figsize, 
                             constrained_layout=True, sharex=True, sharey=True)
        titles = ['Left Foot', 'Right Foot']
        images = []
        
        for i, ax in enumerate(axes):
            if show_both_feet:
                data = contact_map[..., i]
            else:
                data = contact_map
           
            # Create heatmap with fixed value range
            im = ax.imshow(data, cmap='Blues', interpolation='nearest',
                        vmin=0, vmax=1)
            images.append(im)
            
            # Remove axis ticks
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add region grid if specified
            if num_regions is not None:
                self._add_region_grid(ax, data.shape, num_regions, channel_idx=i)
            
            ax.set_title(titles[i], fontsize=10)

        
        # Add colorbar if requested
        if show_colorbar:
            fig.colorbar(images[0], ax=axes, fraction=0.065, pad=0.04, 
                        ticks=[0, 0.5, 1])  # Fixed ticks for clearer reading
            # Make the colorbar labels smaller
            for cbar in fig.get_axes():
                cbar.yaxis.set_tick_params(labelsize=8)
                
        return fig

    def _add_region_grid(self, ax, shape, num_regions, channel_idx=0):
        """Add region grid lines and region ID annotations."""
        h, w = shape
        region_map = self.processor.config.pixel_to_region[..., channel_idx]

        # Draw gridlines
        x_points = np.linspace(0, w, num_regions[1] + 1)
        y_points = np.linspace(0, h, num_regions[0] + 1)

        for x in x_points[1:-1]:
            ax.axvline(x - 0.5, color='red', alpha=0.5, linewidth=1.5)
        for y in y_points[1:-1]:
            ax.axhline(y - 0.5, color='red', alpha=0.5, linewidth=1.5)

        # Region indexing
        regions_per_foot = num_regions[0] * num_regions[1]
        foot_offset = channel_idx * regions_per_foot

        for row_idx in range(num_regions[0]):
            for col_idx in range(num_regions[1]):
                region_id = row_idx * num_regions[1] + col_idx
                global_region = region_id + foot_offset

                # Find pixels with this region
                mask = region_map == global_region
                if np.any(mask):
                    coords = np.column_stack(np.where(mask))
                    center = coords.mean(axis=0)
                    ax.text(center[1], center[0], str(global_region),
                            color='black', fontsize=7, ha='center', va='center',
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, boxstyle='round,pad=0.2'))

    def create_grid(self, predictions, targets, save_path, **kwargs):
        """Create contact map grid visualization."""
        indices = super().create_grid(predictions, targets, save_path, **kwargs)
        num_regions = self.num_regions if self.use_regions else None
        
        # Generate individual visualizations
        gt_images = []
        pred_images = []
       
        for idx in indices:
            gt_viz = self.visualize(
                targets[idx],
                auxiliary_data={
                    'num_regions': num_regions,
                    'is_ground_truth': True,
                    'show_both_feet': self.cfg.viz.figures.contact.show_both_feet,
                    'show_colorbar': self.cfg.viz.figures.contact.show_colorbar
                }
            )
            
            pred_viz = self.visualize(
                predictions[idx],
                auxiliary_data={
                    'num_regions': num_regions,
                    'is_ground_truth': False,
                    'show_both_feet': self.cfg.viz.figures.contact.show_both_feet,
                    'show_colorbar': self.cfg.viz.figures.contact.show_colorbar
                }
            )
            
            if gt_viz is not None and pred_viz is not None:
                gt_images.append(gt_viz)
                pred_images.append(pred_viz)
        
        if gt_images and pred_images:
            self.manager.create_collage(gt_images, pred_images, save_path)

class CoMVisualizer(BaseVisualizer):
    def __init__(self, save_dir, cfg, manager=None):
        super().__init__(save_dir, cfg, manager)
        self.body_parts = BODY_PARTS
        self.data_type = cfg.data.data_type
        self.rgb_colors = rgb_colors
        self.figsize = (8, 4)
        if cfg.data.gt_com and cfg.data.data_type == 'BODY25':
            self.com_only = True
        else:
            self.com_only = False

    def visualize(self, data, auxiliary_data=None, return_fig=False, dpi=None):
        if auxiliary_data is None:
            raise ValueError("CoM visualization requires auxiliary_data")
           
        if dpi is None:
            dpi = self.dpi
             
        # Create figure
        fig = plt.figure(figsize=self.figsize)
        
        # Determine if 3D based on joint dimensions
        is_3d = auxiliary_data['joints'].shape[-1] == 3
        ax = fig.add_subplot(111, projection='3d' if is_3d else None)
      
        if self.com_only:
            fig = self.plot_com_only(
                com_gt=auxiliary_data['com_gt'],
                com_pred=auxiliary_data['com_pred'],
                com_gt_alpha=auxiliary_data['com_gt_conf'],
                com_pred_alpha=auxiliary_data['com_pred_conf'],
                ax=ax,
                figsize=self.figsize
            )
            return fig if return_fig else fig_to_img(fig, dpi=dpi)
         
        else: 
            # Convert confidence values to float for string formatting
            auxiliary_data['com_gt_conf'] = float(auxiliary_data['com_gt_conf'])
            auxiliary_data['com_pred_conf'] = float(auxiliary_data['com_pred_conf'])
            axis_limits = auxiliary_data['axis_limits']
        
            # Plot skeleton and both CoM points
            self.plot_skeleton(
                pose=auxiliary_data['joints'],
                joint_conf=auxiliary_data['joint_conf'],
                com_gt=auxiliary_data['com_gt'],
                com_pred=auxiliary_data['com_pred'],
                ax=ax,
                show_legend=True,
                show_axis_labels=False,
                show_ticks=False,
                com_gt_alpha=auxiliary_data['com_gt_conf'],
                com_pred_alpha=auxiliary_data['com_pred_conf'],
                com_gt_label=f'GT ({auxiliary_data["com_gt_conf"]:.2f})',
                com_pred_label=f'Pred ({auxiliary_data["com_pred_conf"]:.2f})',
                #flip_y=True,
                axis_limits=axis_limits
            )
            
            # Add error as title
            l2_error = np.sqrt(np.sum((auxiliary_data['com_gt'] - auxiliary_data['com_pred'])**2))
            ax.set_title(f'Error: {l2_error:.2f}', fontsize=8)
            
            return fig if return_fig else fig_to_img(fig, dpi=dpi)

    def plot_com_only(self, com_gt, com_pred, com_gt_alpha=1.0, com_pred_alpha=1.0, ax=None, figsize=(6,6)):
        """Plot CoM points only with optional confidence values."""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot CoM points with specified alpha
        ax.scatter(com_gt[0], com_gt[1], com_gt[2], 
                c='blue', s=100, marker='*', label='GT CoM', alpha=com_gt_alpha)
        ax.scatter(com_pred[0], com_pred[1], com_pred[2], 
                c='orange', s=100, marker='^', label='Pred CoM', alpha=com_pred_alpha)
        ax.plot([com_gt[0], com_pred[0]], 
                [com_gt[1], com_pred[1]], 
                [com_gt[2], com_pred[2]], 
                'k--', alpha=min(com_gt_alpha, com_pred_alpha) * 0.5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.grid(True)
        ax.legend(fontsize=8)
        ax.set_box_aspect([1,1,1])
        ax.view_init(elev=30, azim=45)
        
        return fig
        
    def calc_axis_limits(self, poses, com_pred=None, com_gt=None):
            """
            Calculate axis limits for CoM visualizations with capped values.
            Limits are capped at ±200 to prevent extreme zoom levels.
            """
            if poses.ndim == 2:
                poses = poses[np.newaxis, ...]
            joint_dim = poses.shape[-1]
            if joint_dim > 3:  # Has confidence values
                poses = poses[..., :-1]
            
            points = [poses.reshape(-1, poses.shape[-1])]
            if com_pred is not None:
                com_pred = np.atleast_2d(com_pred)
                if com_pred.shape[-1] > poses.shape[-1]:  # Has confidence values
                    com_pred = com_pred[..., :-1]
                points.append(com_pred)
            if com_gt is not None:
                com_gt = np.atleast_2d(com_gt)
                if com_gt.shape[-1] > poses.shape[-1]:  # Has confidence values
                    com_gt = com_gt[..., :-1]
                points.append(com_gt)
            
            points = np.vstack(points)
            center = np.mean(points, axis=0)
            distances = np.linalg.norm(points - center, axis=1)
            max_dist = np.percentile(distances, 95)
            max_dist = max_dist * 1.2  # Add 20% padding
            
            # Calculate limits for each dimension with capping
            limits = []
            for dim in range(points.shape[1]):
                dim_center = center[dim]
                lower = dim_center - max_dist
                upper = dim_center + max_dist
               
                if self.data_type == 'BODY25':
                    cap = 200
                else:
                    cap = 600
                     
                # Cap limits at ±200
                if abs(lower) > cap:
                    lower = -cap
                if abs(upper) > cap:
                    upper = cap
                
                limits.append((lower, upper))
            
            return limits

    def plot_skeleton(self, pose, joint_conf, flip_x=False, com_gt=None, 
                        com_pred=None, ax=None, figsize=(6, 6), show_legend=True, 
                        show_axis_labels=True, show_ticks=True, com_gt_alpha=1.0, 
                        com_pred_alpha=1.0, com_gt_label='GT CoM', com_pred_label='Pred CoM', axis_limits=None):
        """
        Plot skeleton with automatic y-axis orientation based on anatomical joint positions.
        Head/neck joints should be above feet joints in the final visualization.
        """
        # Create copies to avoid modifying original arrays
        pose = pose.copy()
        if com_gt is not None:
            com_gt = com_gt.copy()
        if com_pred is not None:
            com_pred = com_pred.copy()
        
        # Cap alpha values [0,1]
        com_gt_alpha = max(0, min(1, com_gt_alpha))
        com_pred_alpha = max(0, min(1, com_pred_alpha))
            
        # Determine if data is 3D or 2D
        joint_dim = pose.shape[-1]
        is_3d = joint_dim == 3
        
        # Determine if y-flip is needed based on anatomical position
        # For BODY25 model: nose is joint 0, left ankle is 14, right ankle is 11
        head_y = pose[0, 1]  # y-coordinate of nose
        feet_y = max(pose[14, 1], pose[11, 1])  # highest y-coordinate of ankles
        needs_y_flip = head_y < feet_y  # flip if head is below feet
        
        # Apply flips if needed
        if needs_y_flip:
            pose[:, 1] = -pose[:, 1]
            if com_gt is not None:
                com_gt[1] = -com_gt[1]
            if com_pred is not None:
                com_pred[1] = -com_pred[1]
        if flip_x:
            pose[:, 0] = -pose[:, 0]
            if com_gt is not None:
                com_gt[0] = -com_gt[0]
            if com_pred is not None:
                com_pred[0] = -com_pred[0]
        
        # Create figure and axis if not provided
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d' if is_3d else None)
        
        if is_3d:
            # Plot 3D joints with confidence-based alpha
            for i, conf in enumerate(joint_conf):
                if conf > 0.1:  # Only plot joints with reasonable confidence
                    ax.scatter(pose[i, 0], pose[i, 1], pose[i, 2], 
                            c='red', s=20, alpha=conf)
            
            # Plot 3D connections with joint confidence-based alpha
            for i, (joint1, joint2) in enumerate(self.body_parts):
                alpha = min(joint_conf[joint1], joint_conf[joint2])
                if alpha > 0.1:  # Only plot connections with reasonable confidence
                    ax.plot([pose[joint1, 0], pose[joint2, 0]],
                        [pose[joint1, 1], pose[joint2, 1]],
                        [pose[joint1, 2], pose[joint2, 2]],
                        color=self.rgb_colors[i], linewidth=2, alpha=alpha)
            
            # Plot 3D CoM points with specified alpha
            if com_gt is not None:
                ax.scatter(com_gt[0], com_gt[1], com_gt[2], 
                        c='blue', s=100, marker='*', label=com_gt_label, alpha=com_gt_alpha)
            if com_pred is not None:
                ax.scatter(com_pred[0], com_pred[1], com_pred[2], 
                        c='orange', s=100, marker='^', label=com_pred_label, alpha=com_pred_alpha)
                if com_gt is not None:
                    ax.plot([com_gt[0], com_pred[0]], 
                        [com_gt[1], com_pred[1]], 
                        [com_gt[2], com_pred[2]], 
                        'k--', alpha=min(com_gt_alpha, com_pred_alpha) * 0.5)
            
            if show_axis_labels:
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
        else:
            # Plot 2D joints with confidence-based alpha
            for i, conf in enumerate(joint_conf):
                if conf > 0.1:  # Only plot joints with reasonable confidence
                    ax.scatter(pose[i, 0], pose[i, 1], c='red', s=20, alpha=conf)
            
            # Plot 2D connections with joint confidence-based alpha
            for i, (joint1, joint2) in enumerate(self.body_parts):
                alpha = min(joint_conf[joint1], joint_conf[joint2])
                if alpha > 0.1:  # Only plot connections with reasonable confidence
                    ax.plot([pose[joint1, 0], pose[joint2, 0]],
                        [pose[joint1, 1], pose[joint2, 1]],
                        color=self.rgb_colors[i], linewidth=2, alpha=alpha)
            
            # Plot 2D CoM points with specified alpha
            if com_gt is not None:
                ax.scatter(com_gt[0], com_gt[1], c='blue', s=100, marker='*', 
                        label=com_gt_label, alpha=com_gt_alpha)
            if com_pred is not None:
                ax.scatter(com_pred[0], com_pred[1], c='orange', s=100, marker='^', 
                        label=com_pred_label, alpha=com_pred_alpha)
                if com_gt is not None:
                    ax.plot([com_gt[0], com_pred[0]], [com_gt[1], com_pred[1]], 
                        'k--', alpha=min(com_gt_alpha, com_pred_alpha) * 0.5)
            
            if show_axis_labels:
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
        
        # Common settings for both 2D and 3D
        ax.grid(True)
        if show_legend:
            ax.legend(fontsize=8)
        if not show_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
            if is_3d:
                ax.set_zticks([])
        
        # Calculate axis limits using only high-confidence points if limits are not provided
        if axis_limits is None:
            axis_limits = self.calc_axis_limits(
                poses=pose,
                com_pred=com_pred,
                com_gt=com_gt
            )
            
        # Apply limits
        ax.set_xlim(axis_limits[0])
        ax.set_ylim(axis_limits[1])
        
        if is_3d:
            ax.set_zlim(axis_limits[2])
            ax.set_box_aspect([1,1,1])
            ax.view_init(elev=30, azim=45)
        else:
            ax.set_aspect('equal')

    def create_grid(self, predictions, targets, save_path, **kwargs):
        """Create CoM grid visualization."""
        indices = super().create_grid(predictions, targets, save_path, **kwargs)
        poses = kwargs.get('poses')
        axis_limits = kwargs.get('axis_limits')
        
        if poses is None:
            raise ValueError("Poses are required for CoM visualization")
            
        # Calculate grid dimensions
        n_cols = min(4, len(indices))
        n_rows = (len(indices) + n_cols - 1) // n_cols
        
        # Create figure for grid
        fig = plt.figure(figsize=(3 * n_cols, 3 * n_rows))
      
        # Create each subplot
        for i, _ in enumerate(indices):
            ax = fig.add_subplot(n_rows, n_cols, i + 1, 
                               projection='3d' if poses[i][..., :-1].shape[-1] == 3 else None)
            
            if self.com_only:
                self.plot_com_only(
                    com_gt=targets[i][..., :-1],
                    com_pred=predictions[i][..., :-1],
                    com_gt_alpha=float(targets[i][..., -1]),
                    com_pred_alpha=float(predictions[i][..., -1]),
                    ax=ax,
                    figsize=(6, 6)
                )
                
                l2_error = np.sqrt(np.sum((targets[i][..., :-1] - predictions[i][..., :-1])**2))
                ax.set_title(f'Error: {l2_error:.2f}', fontsize=8)
            else:
                auxiliary_data = {
                    'joints': poses[i][..., :-1],
                    'joint_conf': poses[i][..., -1],
                    'com_gt': targets[i][..., :-1],
                    'com_gt_conf': targets[i][..., -1],
                    'com_pred': predictions[i][..., :-1],
                    'com_pred_conf': predictions[i][..., -1],
                    'axis_limits': axis_limits
                }
                
                self.plot_skeleton(
                    pose=auxiliary_data['joints'],
                    joint_conf=auxiliary_data['joint_conf'],
                    com_gt=auxiliary_data['com_gt'],
                    com_pred=auxiliary_data['com_pred'],
                    ax=ax,
                    show_legend=True,
                    show_axis_labels=False,
                    show_ticks=True,
                    #flip_y=False,
                    com_gt_alpha=float(auxiliary_data['com_gt_conf']),
                    com_pred_alpha=float(auxiliary_data['com_pred_conf']),
                    com_gt_label=f'GT ({float(auxiliary_data["com_gt_conf"]):.2f})',
                    com_pred_label=f'Pred ({float(auxiliary_data["com_pred_conf"]):.2f})',
                    axis_limits=axis_limits
                )
                
                # Add error as title
                l2_error = np.sqrt(np.sum((auxiliary_data['com_gt'] - auxiliary_data['com_pred'])**2))
                ax.set_title(f'Error: {l2_error:.2f}', fontsize=8)
        
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        plt.close()
            
class VisualizationManager:
    def __init__(self, save_dir, cfg):
        self.save_dir = Path(save_dir)
        self.cfg = cfg 
        pressure_config = ContactMapConfig(
            use_regions=cfg.data.use_regions,
            contact_threshold=cfg.data.contact_threshold,
            num_regions=cfg.data.num_regions if cfg.data.use_regions else None,
            active_only=cfg.data.active_only,
            binary_contact=cfg.data.binary_contact
        )
        self.processor = PressureMapProcessor(pressure_config)
        self.video_creator = VideoCreator(self.cfg.viz.fps)
        self.visualizers = {
            'pressure': PressureVisualizer(save_dir, cfg, self, processor=self.processor),
            'contact': ContactVisualizer(save_dir, cfg, self, processor=self.processor),
            'com': CoMVisualizer(save_dir, cfg, self)
        }
        
        # Load common assets
        self.foot_mask = np.load('assets/foot_mask_nans.npy')
        self.foot_mask = np.where(np.isnan(self.foot_mask), 0, 1)
        self.plt_style = cfg.viz.plt_style

    def create_learning_curves(self, losses, save_path, title=None, grad_norms=None):
        """
        Create learning curve plots for each loss type and gradient norms with mean and std shading.
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)

        num_losses = len(losses)

        if grad_norms:
            grad_norms_to_plot = {k: v for k, v in grad_norms.items() if k not in ['clip_freq', 'steps']}
            num_grad_plots = len(grad_norms_to_plot)
        else:
            grad_norms_to_plot = {}
            num_grad_plots = 0

        total_plots = num_losses + num_grad_plots

        # Dynamic layout based on number of total plots
        cols = min(3, math.ceil(math.sqrt(total_plots)))
        rows = math.ceil(total_plots / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), squeeze=False)
        axes = axes.flatten()

        def plot_curve(ax, values, steps, title, ylabel, color='blue'):
            window_size = min(50, len(values) // 10)
            window_size = max(2, window_size)

            means, stds, steps_avg = [], [], []
            for i in range(0, len(values), window_size):
                batch = values[i:i+window_size]
                if len(batch) >= 2:
                    means.append(np.mean(batch))
                    stds.append(np.std(batch))
                    steps_avg.append(np.mean(steps[i:i+window_size]))

            means = np.array(means)
            stds = np.array(stds)
            steps_avg = np.array(steps_avg)

            ax.plot(steps_avg, means, label='Mean', color=color, linewidth=2)
            ax.fill_between(steps_avg, means - stds, means + stds, alpha=0.3, color=color, label='±1 std')

            ax.set_xlabel('Steps')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend()

            if max(means) / (min(means) + 1e-10) > 100:
                ax.set_yscale('log')

        # Plot losses (generalized)
        for idx, (loss_name, loss_data) in enumerate(losses.items()):
            values = np.array(loss_data['values'])
            steps = np.array(loss_data['steps'])
            plot_curve(axes[idx], values, steps, 
                    f'{loss_name} Learning Curve', f'{loss_name} Loss')

        # Plot gradient norms if present (structure-specific)
        for idx, (norm_type, norm_data) in enumerate(grad_norms_to_plot.items()):
            plot_idx = num_losses + idx
            values = np.array(norm_data)
            steps = np.array(grad_norms['steps'])
            plot_curve(axes[plot_idx], values, steps,
                    f'{norm_type.replace("_", " ").capitalize()} Over Time',
                    norm_type.replace("_", " ").capitalize())

        for i in range(total_plots, len(axes)):
            axes[i].axis('off')

        if title:
            fig.suptitle(title, fontsize=16, y=1.02)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close() 
    
    def _stack_images(self, images, stack_vertical):
        """Stack images while maintaining aspect ratio"""
        if stack_vertical:
            # Target width is the max width among images
            target_width = max(img.shape[1] for img in images)
            
            # Resize images to the target width while maintaining aspect ratio
            images_resized = []
            for img in images:
                aspect_ratio = img.shape[0] / img.shape[1]
                target_height = int(target_width * aspect_ratio)
                images_resized.append(cv2.resize(img, (target_width, target_height)))
            
            return np.vstack(images_resized)
        else:
            # Target height is the max height among images
            target_height = max(img.shape[0] for img in images)
            
            # Resize images to the target height maintaining aspect ratio
            images_resized = []
            for img in images:
                aspect_ratio = img.shape[1] / img.shape[0]
                target_width = int(target_height * aspect_ratio)
                images_resized.append(cv2.resize(img, (target_width, target_height)))
            
            return np.hstack(images_resized)
            
    def stack_image_files(self, image_paths, save_path, stack_vertical=True):
        """Stack images from files"""
        images = [cv2.imread(str(path)) for path in image_paths]
        if any(img is None for img in images):
            raise ValueError("Failed to load one or more images")
        images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
        
        # Stack images
        stacked = self._stack_images(images, stack_vertical)
        
        # Save stacked image
        cv2.imwrite(str(save_path), cv2.cvtColor(stacked, cv2.COLOR_RGB2BGR))

    @with_style()
    def create_validation_grid(self, predictions, targets, save_path, start_idx=None, num_samples=8):
        """Create validation grids for all modalities with consistent indices."""
        # First, determine common indices for all modalities
        base_modality = next(iter(predictions.keys()))
        step_size = 5
        indices = np.arange( 
            start_idx if start_idx is not None else np.random.randint(0, len(predictions[base_modality]) - (num_samples*step_size)),
            start_idx + (num_samples*step_size) if start_idx is not None else start_idx + (num_samples*step_size),
            step_size 
        )

        img_paths = []
        # Build kwargs based on modality
        kwargs = {
            'indices': indices,  # Pass the same indices to all visualizers
            'num_samples': num_samples
        } 
        
        # If CoM in preds, calc axis limits
        if 'com' in predictions:
            axis_limits = self.visualizers['com'].calc_axis_limits(
                poses=targets['middle_frame_joints'][indices, ...],
                com_pred=predictions['com'][indices, ...],
                com_gt=targets['com'][indices, ...]
            )
            kwargs['axis_limits'] = axis_limits 
            
        for modality, visualizer in self.visualizers.items():
            if modality in predictions and modality in targets:
                modality_save_path = save_path.parent / f'{modality}_grid.png'
                img_paths.append(modality_save_path)
                
                # Add modality-specific data
                if modality == 'com':
                    kwargs['poses'] = targets['middle_frame_joints'][indices]
                    
                elif modality == 'contact':
                    kwargs['foot_mask'] = self.foot_mask
                    kwargs['num_regions'] = self.cfg.data.num_regions if hasattr(self.cfg.data, 'use_regions') else None

                    if self.visualizers['contact'].mirror:
                        num_regions = self.cfg.data.num_regions
                        targets['contact'] = mirror_contact_regions(targets['contact'], num_regions)
                        predictions['contact'] = mirror_contact_regions(predictions['contact'], num_regions)
                
                visualizer.create_grid(
                    predictions=predictions[modality],
                    targets=targets[modality],
                    save_path=modality_save_path,
                    **kwargs
                )
        
        # Stack images
        self.stack_image_files(img_paths, save_path, stack_vertical=True)
                   
    def create_collage(self, gt_images, pred_images, output_path): 
        """Create a collage of ground truth and predicted images."""
        n_images = len(gt_images)
        img_height, img_width = gt_images[0].shape[:2]
        
        # Create canvas with space for labels
        label_width = 30
        collage_width = img_width * n_images + label_width
        collage_height = img_height * 2
        collage = Image.new('RGB', (collage_width, collage_height), color='white')
        
        # Paste images
        for i, (gt_img, pred_img) in enumerate(zip(gt_images, pred_images)):
            gt_img_pil = Image.fromarray(gt_img)
            pred_img_pil = Image.fromarray(pred_img)
            
            collage.paste(gt_img_pil, (label_width + i * img_width, 0))
            collage.paste(pred_img_pil, (label_width + i * img_width, img_height))
        
        # Add labels
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except OSError:
            # Fallback to default font if DejaVu Sans is not available
            font = ImageFont.load_default()
        
        # Ground Truth label
        gt_label = Image.new('RGB', (img_height, label_width), color='white')
        gt_draw = ImageDraw.Draw(gt_label)
        gt_draw.text((5, 5), "GT", fill='black', font=font, anchor='lt')
        gt_label = gt_label.rotate(90, expand=True)
        collage.paste(gt_label, (0, 0))
        
        # Predicted label
        pred_label = Image.new('RGB', (img_height, label_width), color='white')
        pred_draw = ImageDraw.Draw(pred_label)
        pred_draw.text((5, 5), "Predicted", fill='black', font=font, anchor='lt')
        pred_label = pred_label.rotate(90, expand=True)
        collage.paste(pred_label, (0, img_height))
        
        # Save collage
        collage.save(output_path)

    def create_modality_videos(self, predictions, targets, subject, frame_mask=None, stack_vertical=False):
        """Create visualization video with flexible stacking orientation using parallelized frame generation."""
        if not self.cfg.viz.enabled or not self.cfg.viz.video.enabled:
            return

        modalities = self.cfg.viz.video.modalities
        downsample = self.cfg.viz.video.downsample
        video_dpi = self.cfg.viz.video.dpi

        # Create single output directory for frames
        viz_dir = self.save_dir / f'Subject{subject}' / 'visualization_frames'
        frames_dir = viz_dir / 'combined'
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        num_frames = len(next(iter(predictions.values())))
        frame_indices = range(0, num_frames, downsample)
       
        global_limits = None 
        if 'com' in modalities:
            axis_limits = self.visualizers['com'].calc_axis_limits(
            poses=predictions['middle_frame_joints'],
            com_pred=predictions['com'],
            com_gt=targets['com']
        )
        else:
            axis_limits = None
            
        try:
            # Parallelize frame generation
            print("Generating frames...")
            if self.cfg.viz.video.video_processors == 1:
                for frame_idx in tqdm(frame_indices):
                    self.process_frame(
                        frame_idx, modalities, predictions, targets, frames_dir, video_dpi, frame_mask, stack_vertical,axis_limits 
                    )
            else:
                process_map(
                    self.process_frame_for_video,  # Use a standalone method
                    [(frame_idx, modalities, predictions, targets, frames_dir, video_dpi, frame_mask, stack_vertical, axis_limits) for frame_idx in frame_indices],
                    max_workers=self.cfg.viz.video.video_processors
                )
            # Create single video from combined frames
            print("\nCreating video from frames...")
            output_path = viz_dir.parent / 'visualization.mp4'
            try:
                self.video_creator.create_video_from_frames(
                    frames_dir,
                    output_path,
                )
                print(f"Created video: {output_path}")
            except Exception as e:
                print(f"Error creating video: {e}")

            print("\nVideo creation complete!")
            
        except Exception as e:
            print(f"Error generating frames: {e}")
            raise
        
        # Clean up frames directory
        shutil.rmtree(frames_dir)

    def process_frame_for_video(self, args):
        """Standalone method for processing a single frame, compatible with multiprocessing."""
        frame_idx, modalities, predictions, targets, frames_dir, video_dpi, frame_mask, stack_vertical, axis_limits = args

        self.process_frame(
            frame_idx, modalities, predictions, targets, frames_dir, video_dpi, frame_mask, stack_vertical, axis_limits=axis_limits
        )
    
    def process_frame(self, frame_idx, modalities, predictions, targets, frames_dir, video_dpi, frame_mask, stack_vertical, axis_limits=None):
        """Process and save a single frame for the video."""
        if frame_mask is not None and not frame_mask[frame_idx]:
            return

        # Generate visualizations for each modality
        modality_images = []
        for modality in modalities:
            if modality == 'com':
                viz = self._handle_com_frame(frame_idx, predictions, targets, frames_dir, video_dpi, axis_limits)
                modality_images.append(viz)
            else:
                gt_viz, pred_viz = self._handle_modality_frame(
                    modality=modality,
                    frame_idx=frame_idx,
                    predictions=predictions,
                    targets=targets,
                    output_dir=frames_dir,
                    dpi=video_dpi
                )
                viz = np.hstack([gt_viz, pred_viz])
                modality_images.append(viz)

        # Resize images for stacking
        resized_images = prepare_images_for_stacking(modality_images, stack_vertical)

        # Stack images in specified direction
        if stack_vertical:
            full_frame = np.vstack(resized_images)
        else:
            full_frame = np.hstack(resized_images)

        # Save frame
        frame_path = frames_dir / f'frame_{frame_idx:06d}.png'
        Image.fromarray(full_frame).save(frame_path)

    def _handle_com_frame(self, frame_idx, predictions, targets, output_dir, dpi, axis_limits=None):
        """Handle frame generation for CoM visualization"""
        pose = predictions['middle_frame_joints'][frame_idx]
        auxiliary_data = {
            'joints': pose[..., :-1],
            'joint_conf': pose[..., -1],
            'com_gt': targets['com'][frame_idx][..., :-1],
            'com_gt_conf': targets['com'][frame_idx][..., -1],
            'com_pred': predictions['com'][frame_idx][..., :-1],
            'com_pred_conf': predictions['com'][frame_idx][..., -1],
            'axis_limits': axis_limits 
        }
        
        combined_viz = self.visualizers['com'].visualize(
            data=None,
            auxiliary_data=auxiliary_data,
            dpi=dpi
        )
        return combined_viz

    def _handle_modality_frame(self, modality, frame_idx, predictions, targets, 
                             output_dir, dpi):
        """Handle frame generation for pressure and contact visualizations"""
        auxiliary_data = {}
        if modality == 'contact':
            auxiliary_data = {
                'num_regions': self.cfg.data.num_regions if hasattr(self.cfg.data, 'num_regions') else None,
                'foot_mask': self.foot_mask
            }
       
        # Get the visualizations as arrays but don't save them separately
        gt_viz = self.visualizers[modality].visualize(
            targets[modality][frame_idx],
            auxiliary_data={**auxiliary_data, 'is_ground_truth': False},
            dpi=dpi
        )
        pred_viz = self.visualizers[modality].visualize(
            predictions[modality][frame_idx],
            auxiliary_data={**auxiliary_data, 'is_ground_truth': False},
            dpi=dpi
        )
        
        return gt_viz, pred_viz