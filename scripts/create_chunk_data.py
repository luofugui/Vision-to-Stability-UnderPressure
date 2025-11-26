import argparse 
from pathlib import Path
import json
import numpy as np
import json
import os
from glob import glob
import h5py
from scipy.io import loadmat
import re
import pickle
import sys
from pressure.data.com_support import CenterOfMass
from pressure.util.visualizer import BODY_PARTS

def parse_args():
    parser = argparse.ArgumentParser()
    # Required 
    parser.add_argument('--root_dir', type=str, required=True, help='Path to the PSU-TMM root')
    parser.add_argument('--save_path', type=str, default='Chunked_PSU', help='Path to save the chunked data')
    parser.add_argument('--data_type', type=str, default='BODY25', choices=['BODY25', 'BODY25_3D', 'MOCAP', 'MOCAP_3D', 'MOCAP_MRK'], help='Type of data to use')
    parser.add_argument('--sample_rate', type=int, default=5, help='Sample rate of the data (Hz)')
    parser.add_argument('--OM', action='store_true', help='Create chunks for ordinary movement data')
    parser.add_argument('--gt_com', action='store_true', help='Use ground truth CoM (from mocap) instead of calculated CoM on openpose')
    # Optional
    parser.add_argument('--view', type=int, default=1, help='View camera view to use (1 or 2)')
    parser.add_argument('--chunk_size', type=int, default=5000, help='Number of samples in each chunk')
    parser.add_argument('--save_prefix', type=str, default=None, help='Prefix for saved chunk files')
    parser.add_argument('--max_norm', action='store_true', help='Normalize data by max value')
    parser.add_argument('--weight_norm', action='store_true', help='Apply weight normalization')
    parser.add_argument('--sigmoid_pressure', action='store_true', help='Apply sigmoid to pressure data')
    parser.add_argument('--rotation_inv', action='store_true', help='Apply rotation invariance')
    parser.add_argument('--remove_zero_pressure', action='store_true', help='Remove samples with zero pressure')
    parser.add_argument('--make_pressure_distribution', action='store_true', help='Create pressure distribution')
    parser.add_argument('--threshold', type=float, default=0, help='Threshold applied to the pressure data')
    parser.add_argument('--bone_normalize', action='store_true', 
                       help='Normalize by average bone length instead of z-score')
    parser.add_argument('--center_w_op', action='store_true', help='Center CoM with OpenPose origin, not pelvis')
    args = parser.parse_args()
    return args

def rotation_invariance(joints, left_hip_idx=12, right_hip_idx=9, eps=1e-8):
    """  
    Normalize pose data by aligning a reference vector to a target axis.
    Handles both 2D and 3D data.
    
    Args:
        joints (np.array): joint data of shape (N, num_joints, joint_dim)
        left_hip_idx (int): left hip index
        right_hip_idx (int): right hip index
    """
    joints += eps
    joint_dim = joints.shape[-1]
    # Get primary vector (e.g., hip line)
    primary_vector = joints[:, right_hip_idx] - joints[:, left_hip_idx]
    primary_vector = primary_vector / np.linalg.norm(primary_vector, axis=1, keepdims=True)
    
    if joint_dim == 3:
        # 3D case - create orthonormal basis
        z_vector = np.tile(np.array([0, 0, 1]), (primary_vector.shape[0], 1))
        y_vector = np.cross(z_vector, primary_vector)
        y_vector = y_vector / np.linalg.norm(y_vector, axis=1, keepdims=True)
        x_vector = np.cross(y_vector, z_vector)
        
        # Create rotation matrix
        rotation_matrix = np.stack([x_vector, y_vector, z_vector], axis=1)
    else:
        # 2D case - create 2D rotation matrix
        x_vector = primary_vector
        y_vector = np.stack([-x_vector[:, 1], x_vector[:, 0]], axis=1)
        rotation_matrix = np.stack([x_vector, y_vector], axis=1)
    
    # Apply rotation
    rotated_joints = np.einsum('bij,bkj->bki', rotation_matrix, joints)
    
    # Store rotation matrix for potential denormalization
    return rotated_joints, rotation_matrix 

def get_bone_length_scale(joints, body_parts=BODY_PARTS):
    """Calculate average bone length for scaling using existing BODY_PARTS."""
    bone_lengths = []
    for joint1, joint2 in body_parts:
        # Skip confidence values
        bone_vec = joints[:, joint1, :] - joints[:, joint2,]
        lengths = np.linalg.norm(bone_vec, axis=-1)
        bone_lengths.append(np.median(lengths))
    return np.mean(bone_lengths)

def load_gt_com_OM(cfg, om_data_path, pelvis_idx=12, center_w_op=False):
    """Load ground truth CoM from OM data and center about the pelvis."""
    com_path = f'{om_data_path}/CoM.npy'
    mocap_path = f'{om_data_path}/MOCAP_3D.npy'
    
    # Load all mocap files
    if not os.path.exists(mocap_path) or not os.path.exists(com_path):
        raise FileNotFoundError(f'File not found: {mocap_path} or {com_path}')
   
    com_data = np.load(com_path) 
    com_data = com_data[::cfg.sample_rate]  # Downsample

    mocap_data = np.load(mocap_path)
    mocap_data = mocap_data[::cfg.sample_rate]  # Downsample
    pelvis_origin = mocap_data[:, pelvis_idx, :]  # Extract pelvis position
    org_points = np.zeros((len(com_data), 3))  # Shape (N, 3)     Z
   
    if not center_w_op: 
        # Identify valid CoM frames (i.e., when last value == 1)
        valid_mask = com_data[:, -1] == 1
        # Only center CoM when it exists (i.e., valid frames)
        com_data[valid_mask, :3] -= pelvis_origin[valid_mask,:3]

        org_points = np.zeros((len(com_data), 3))  # Shape (N, 3)
        org_points[valid_mask] = pelvis_origin[valid_mask, :3]
        org_points = np.array(org_points)
  
    return com_data, org_points

def load_gt_com(cfg, subject, used_takes, pelvis_idx=12, center_w_op=False):
    """Load ground truth CoM from Vicon data and center about the pelvis."""
    com_path = f'{cfg.root_dir}/Subject_wise/Subject{subject}/CoM_'
    mocap_path = f'{cfg.root_dir}/Subject_wise/Subject{subject}/MOCAP_3D_' 
    
    # Load all mocap files
    data = []
    pelvis_origins = []
    for take in used_takes:
        com_file = f'{com_path}{take}.mat'
        mocap_file = f'{mocap_path}{take}.mat'
        if not os.path.exists(mocap_file) or not os.path.exists(com_file):
            raise FileNotFoundError(f'File not found: {mocap_file} or {com_file}')
        
        com_data = loadmat(com_file)['CoM']
        com_data = com_data[::cfg.sample_rate]  # Downsample
        
        mocap_data = h5py.File(mocap_file, 'r')['POSE']
        mocap_data = np.transpose(mocap_data, (2, 1, 0))
        mocap_data = mocap_data[::cfg.sample_rate]  # Downsample
        pelvis_origin = mocap_data[:, pelvis_idx, :]  # Extract pelvis position
       
        if not center_w_op: 
            # Identify valid CoM frames (i.e., when last value == 1)
            valid_mask = com_data[:, -1] == 1
            # Only center CoM when it exists (i.e., valid frames)
            com_data[valid_mask, :3] -= pelvis_origin[valid_mask,:3]

            org_points = np.zeros((len(com_data), 3))  # Shape (N, 3)
            org_points[valid_mask] = pelvis_origin[valid_mask, :3]
            
            pelvis_origins.extend(org_points) 
            
        data.extend(com_data)
  
    com = np.array(data) 
    pelvis_origins = np.array(pelvis_origins)

    return com, pelvis_origins

def data_normalization(joints, pressure, data_type, remove_zero_pressure=False, rotation_inv=False, threshold=0, bone_normalize=False, subject=None, cfg=None, used_takes=None, om_data_path=None):
    # Clip pressure values between 0 and 862 and normalize with max
    pressure = np.nan_to_num(pressure)
    pressure = np.clip(pressure, 0, 862)
    max_fp = np.max(pressure, axis=0)

    if remove_zero_pressure:
        # Mask of any pressure samples with a sum of 0
        mask = np.sum(pressure, axis=(1, 2, 3)) != 0
        pressure = pressure[mask]
        joints = joints[mask]

        # Get number of samples removed with zero pressure
        print(f'Number of samples with zero pressure discarded: {len(mask) - len(pressure)} ({round((len(mask) - len(pressure)) / len(mask) * 100, 2)}%)')

    pressure[pressure < threshold] = 0
    if 'BODY25' in data_type:
        joints = joints[:, :24] 
        

    joint_conf = joints[:,:,-1]
    joint_data = joints[:,:,:-1] # Shape (N, joint_dim, 2)
     
    # Define the origin index based on data_type
    origin_idx = None
    idx = {'MOCAP': 12, 'HRNET': 12, 'BODY25': 8}
    for key in idx:
        if key in data_type:
            origin_idx = idx[key]
            
    if 'MRK' in data_type:
        # We are going to use the markers around the hip.
        lasi,rasi,lpsi,rpsi = 23, 24, 25, 26 # L/R Anterior/Posterior Superior Iliac Spine
        pelvis_indices = [lasi, rasi, lpsi, rpsi]
        origin_points = np.mean(joints[:, pelvis_indices, :-1], axis=1, keepdims=True)

        norm_values = {
            'origins': {
                'pelvis_avg': {
                    'indices': pelvis_indices,
                    'points': origin_points
                }
            }
        }
    else:
        if origin_idx is None:
            raise NotImplementedError("Data type not implemented")
        origin_points = joint_data[:, origin_idx:origin_idx+1, :]

        norm_values = {
            'origins': {
                'joint': {
                    'idx': origin_idx,
                    'points': origin_points
                }
            }
        }

    # Center joints
    joints_centered = joint_data - origin_points
    
    # Apply rotation invariance before CoM calculation
    if rotation_inv and 'MRK' not in data_type:
        joints_centered, rotation_mat = rotation_invariance(joints_centered)
        rotation_mat = np.nan_to_num(rotation_mat)
        norm_values['rotation'] = {
            'matrix': rotation_mat,
            'applied': True
        }
        joints_centered = np.nan_to_num(joints_centered) 
        joints_centered = np.round(joints_centered, 8)
        rotation_mat = np.round(rotation_mat, 8)
        
    if bone_normalize and 'BODY25' in data_type:
        scale = get_bone_length_scale(joints_centered) + 1e-8
        skeleton_norm = joints_centered / scale

        norm_values['joint'] = {
            'scale': scale,
            'type': 'bone'
        }
    else:
        joint_mean = np.mean(joints_centered, axis=0)  # shape (#joints, dims)
        joint_std  = np.std(joints_centered, axis=0) + 1e-8
        skeleton_norm = (joints_centered - joint_mean) / joint_std

        norm_values['joint'] = {
            'mean': joint_mean,
            'std':  joint_std,
            'type': 'zscore'
        }

    if cfg is not None and cfg.gt_com:
        # --- GT CoM from Vicon ---
        if om_data_path is not None:
            vicon_com, pelvis_origins = load_gt_com_OM(cfg, om_data_path, center_w_op=cfg.center_w_op) 
        else:
            vicon_com, pelvis_origins = load_gt_com(cfg, subject, used_takes=used_takes, center_w_op=cfg.center_w_op)
        
        if cfg.center_w_op:
            com_conf = vicon_com[:, -1]  # Extract CoM confidence values
            centered_com = np.zeros_like(vicon_com[:, :3])  # Placeholder for normalized CoM
            # Ensure pelvis_origins is correctly shaped
            og_pelvis_origins = np.squeeze(origin_points, axis=1) 
            pelvis_locs = np.zeros_like(og_pelvis_origins)
            
            # Mask valid CoM values (confidence == 1)
            mask = com_conf == 1
            pelvis_locs[mask] = og_pelvis_origins[mask]
            
            # Only normalize where CoM exists (confidence = 1)
            centered_com[mask] = vicon_com[mask, :3] - og_pelvis_origins[mask]

            # Assign back the normalized values
            vicon_com[:, :3] = centered_com 

            # Add confidence back to CoM
            vicon_com = np.concatenate([vicon_com[:, :3], com_conf[:, None]], axis=-1)
            com_values = vicon_com[:, :3]  # Extract (x, y, z)
            com_confidence = vicon_com[:, 3]  # Extract confidence

            # Compute Z-score normalization parameters
            com_mean = np.mean(com_values, axis=0)
            com_std = np.std(com_values, axis=0) + 1e-8

            # Normalize CoM values
            com_norm = (com_values - com_mean) / com_std
        else:
            com_values = vicon_com[:, :3]  # Extract (x, y, z)
            com_confidence = vicon_com[:, 3]  # Extract confidence
            
            # Compute Z-score normalization parameters
            com_mean = np.mean(com_values, axis=0)
            com_std = np.std(com_values, axis=0) + 1e-8
            com_norm = (com_values - com_mean) / com_std
            


        # Store normalization parameters
        norm_values['com'] = {
            'mean': com_mean,
            'std': com_std,
            'type': 'zscore_gt'
        }

        # Store pelvis origin to unnormalize later
        norm_values['origins']['pelvis'] = {
            'points': pelvis_origins
        }
    else:
        # --- OpenPose-derived CoM ---
        com_calculator = CenterOfMass(dims=joint_data.shape[-1])
        com_values, com_confidence = com_calculator.calculate_total_com(
            np.concatenate([joints_centered, joint_conf[..., None]], axis=-1)
        )

        if bone_normalize:
            scale = norm_values['joint']['scale']
            com_norm = com_values / scale
            norm_values['com'] = {
                'scale': scale,
                'type': 'bone'
            }
        else:
            com_mean = np.mean(com_values, axis=0)
            com_std = np.std(com_values, axis=0) + 1e-8
            com_norm = (com_values - com_mean) / com_std

            norm_values['com'] = {
                'mean': com_mean,
                'std': com_std,
                'type': 'zscore'
            }
        
    skeleton_norm = np.nan_to_num(skeleton_norm)
    com_norm      = np.nan_to_num(com_norm)
    normalized_joint_data = np.concatenate([skeleton_norm, joint_conf[...,None]], axis=-1)
    com_confidence = com_confidence.reshape(-1,1)
    normalized_com_data = np.concatenate([com_norm, com_confidence], axis=-1)

    return normalized_joint_data, pressure, max_fp, norm_values, normalized_com_data

def create_OM_chunks(args, OM_dir, OM, chunk_size, sample_rate, data_type, view, weight, save_path, remove_zero_pressure=False):
    """
    Create and save chunked ordinary movement (OM) data with the same processing as normal data,
    using OpenPose to calculate CoM instead of relying on external CoM files.

    Args:
        OM_dir (str): Path to OM data directory.
        OM (str): Identifier for OM instance.
        chunk_size (int): Number of samples per chunk.
        sample_rate (int): Downsampling rate.
        data_type (str): Type of joint data (e.g., BODY25, MOCAP).
        view (int): Camera view.
        zero_conf_threshold (int): Max allowed zero-confidence joints.
        save_path (str): Path to save chunks.
        rotation_inv (bool): Apply rotation invariance.
        remove_zero_pressure (bool): Remove samples with zero pressure.
        make_pressure_distribution (bool): Normalize pressure to a probability distribution.
        max_norm (bool): Normalize pressure by max value.
        weight_norm (bool): Apply weight normalization.
    """

    # Load Pressure and Joint Data
    pressure_path = os.path.join(OM_dir, 'Pressure.npy')
    if not os.path.exists(pressure_path):
        pressure_path = os.path.join(OM_dir, 'pressure.npy')
    pressure_data = np.load(pressure_path)
    if '3D' in data_type or 'MOCAP_MRK' in data_type:
        joint_data = np.load(os.path.join(OM_dir, f'{data_type}.npy'))
    else:
        joint_data = np.load(os.path.join(OM_dir, f'{data_type}_V{view}.npy'))
    # Apply sampling rate
    pressure_data = pressure_data[::sample_rate]
    joint_data = joint_data[::sample_rate]

    # Normalize Joint and Pressure Data
    joint_dataset, pressure_dataset, foot_pressure_max, joint_norm_info, com_data = data_normalization(
        joint_data, pressure_data, data_type, remove_zero_pressure=remove_zero_pressure, 
        rotation_inv=args.rotation_inv, bone_normalize=True, om_data_path=OM_dir if args.gt_com else None, cfg=args
    )

    max_pressure = foot_pressure_max.max()
    # Save Normalization Info
    normalization_info = {
        'pressure': {
            'max_pressure': max_pressure,
            'weight': weight,  
            'original_sums': None,
            'weight_normalized': args.weight_norm,
            'max_normalized': args.max_norm 
        },
        'joint': joint_norm_info['joint'],
        'com': joint_norm_info['com'],
        'origins': joint_norm_info['origins']
    }

    # Normalize Pressure if needed
    if args.weight_norm:
        pressure_dataset = weight_normalization(pressure_dataset, weight=weight)  # Assume uniform weight if no subject data
        max_pressure = pressure_dataset.max()
        
    if args.make_pressure_distribution:
        original_sums = np.sum(pressure_dataset, axis=(1, 2, 3))
        nonzero_mask = original_sums > 0
        pressure_dataset[nonzero_mask] = pressure_dataset[nonzero_mask] / original_sums[nonzero_mask, None, None, None]
        normalization_info['pressure']['original_sums'] = original_sums
        
    elif args.max_norm:
        mask = pressure_dataset != 0
        pressure_dataset = np.where(mask, pressure_dataset / max_pressure, 0)

    # Create Chunked Data
    data = [
        {
            'joint': joint,
            'pressure': pressure,
            'com': com
        }
        for joint, pressure, com in zip(joint_dataset, pressure_dataset, com_data)
    ]

    num_chunks = (len(data) + chunk_size - 1) // chunk_size  # Compute number of chunks

    metadata = []
    for i in range(num_chunks):
        chunk = data[i * chunk_size: (i + 1) * chunk_size]
        file_path = os.path.join(save_path, f'OM_{OM}_chunk_{i}.pkl')
        if sys.platform.startswith('win'):
            file_path = file_path.replace("/", "\\")  # Windows path fix

        with open(file_path, 'wb') as f:
            pickle.dump(chunk, f)

        metadata.append({
            "file_path": file_path,
            "chunk_size": len(chunk),
        })

    # Save metadata
    with open(os.path.join(save_path, f"OM_{OM}_metadata.json"), 'w') as f:
        json.dump(metadata, f)
        
    return max_pressure, normalization_info

def weight_normalization(pressure_dataset, weight):
    # Ensure zeros stay zeros
    mask = pressure_dataset != 0
    
    # Factor out weight and normalize only non-zero values
    normalized_pressure_dataset = np.where(mask, pressure_dataset / weight, 0)
   
    return normalized_pressure_dataset 

def create_chunks(args, subject, weight):
    save_path = args.save_path
    all_pressure_data = []
    all_joint_data = []

    subject_dir = os.path.join(args.root_dir, f'Subject_wise/Subject{subject}')
    pressure_files = glob(os.path.join(subject_dir, 'Pressure_*.mat'))

    if '3D' in args.data_type or 'MRK' in args.data_type:
        joint_files = glob(os.path.join(subject_dir, f'{args.data_type}_*.mat'))
    else:
        joint_files = glob(os.path.join(subject_dir, f'{args.data_type}_V{args.view}_*.mat'))

    natural_numerical_sort = lambda x: [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', os.path.splitext(os.path.basename(x))[0])]
    pressure_files.sort(key=natural_numerical_sort)
    joint_files.sort(key=natural_numerical_sort)
    
    # Function to extract take numbers from filenames
    def extract_take_number(filename):
        return re.search(r'_(\d+)\.mat', filename).group(1)

    # Create dictionaries to map take numbers to filenames
    pressure_files_dict = {extract_take_number(file): file for file in pressure_files}
    joint_files_dict = {extract_take_number(file): file for file in joint_files}

    # Iterate over joint files and match them with pressure files
    matched_files = []
    used_take_numbers = []
    for take_number, joint_file in joint_files_dict.items():
        if take_number in pressure_files_dict:
            pressure_file = pressure_files_dict[take_number]
            matched_files.append((pressure_file, joint_file))
            used_take_numbers.append(take_number)
        else:
            print(f"No corresponding pressure file for joint file: {joint_file}")
    
    for pressure_file, joint_file in matched_files:
        press_sub = pressure_file.split('_')[-1].split('.')[0]
        joint_sub = joint_file.split('_')[-1].split('.')[0]
        
        if press_sub != joint_sub:
            print(f'Pressure and joint files do not match: {pressure_file}, {joint_file}')
            continue

        pressure_data = np.array(h5py.File(pressure_file, 'r')['PRESSURE'])
        if 'MRK' in args.data_type:
            joint_data = loadmat(joint_file)['mcMarkers']
        else:
            joint_data = np.array(h5py.File(joint_file, 'r')['POSE'])
       
        if 'MRK' not in args.data_type: 
            joint_data = np.transpose(joint_data, (2, 1, 0))
        pressure_data = np.transpose(pressure_data, (3, 2, 1, 0))
        
        pressure_data = pressure_data[::args.sample_rate]
        joint_data = joint_data[::args.sample_rate]

        all_pressure_data.append(pressure_data)
        all_joint_data.append(joint_data)


    pressure_dataset = np.concatenate(all_pressure_data, axis=0)
    joint_dataset = np.concatenate(all_joint_data, axis=0)
    joint_dataset, pressure_dataset, foot_pressure_max, joint_norm_info, com_data = data_normalization(joint_dataset, pressure_dataset, args.data_type, args.remove_zero_pressure, args.rotation_inv, args.threshold, args.bone_normalize, subject=subject, cfg=args, used_takes=used_take_numbers)
  
    max_pressure = foot_pressure_max.max()
    
    # Create comprehensive normalization info dictionary
    normalization_info = {
        'pressure': {
            'max_pressure': max_pressure,
            'weight': weight,
            'original_sums': None,
            'weight_normalized': args.weight_norm,
            'max_normalized': args.max_norm
        },
        'joint': joint_norm_info['joint'],
        'com': joint_norm_info['com'],
        'origins': joint_norm_info['origins']
    }
    
    # Handle pressure normalization
    if args.weight_norm:
        pressure_dataset = weight_normalization(pressure_dataset, weight)
        max_pressure = pressure_dataset.max()
        
    if args.make_pressure_distribution:
        original_sums = np.sum(pressure_dataset, axis=(1, 2, 3))
        nonzero_mask = original_sums > 0
        pressure_dataset[nonzero_mask] = pressure_dataset[nonzero_mask] / original_sums[nonzero_mask, None, None, None]
        normalization_info['pressure']['original_sums'] = original_sums
    else:
        if args.max_norm:
            mask = pressure_dataset != 0
            pressure_dataset = np.where(mask, pressure_dataset / max_pressure, 0)
    
    # Create data samples with joint, pressure, and com information
    chunk_size = args.chunk_size
    data = [
        {
            'joint': joint,
            'pressure': pressure,
            'com': com
        }
        for joint, pressure, com in zip(joint_dataset, pressure_dataset, com_data)
    ]
    
    # Calculate number of chunks
    num_chunks = (len(data) + chunk_size - 1) // chunk_size
    
    # Create metadata for chunks
    metadata = []
    for i in range(num_chunks):
        chunk = data[i*chunk_size:(i+1)*chunk_size]
        file_path = os.path.join(save_path, f'subject_{subject}_chunk_{i}.pkl')
        file_path = Path(file_path).as_posix() 
            
        # Save chunk
        with open(file_path, 'wb') as f:
            pickle.dump(chunk, f)
            
        metadata.append({
            "file_path": file_path,
            "chunk_size": len(chunk),
        })
    
    # Save metadata
    with open(os.path.join(save_path, f"subject_{subject}_metadata.json"), 'w') as f:
        json.dump(metadata, f)
        
    return max_pressure, normalization_info

def process_single_subject_OMs(args, subject_dir, subject_name, base_save_path):
    """Process all OM takes for a single subject"""
    OM_dirs = sorted(glob(os.path.join(subject_dir, 'OM*/')))
    
    with open('assets/OM_weights.json', 'r') as f:
        weights = json.load(f)
    weight = weights[subject_name] 
    
    subject_save_path = os.path.join(base_save_path, subject_name)
    os.makedirs(subject_save_path, exist_ok=True)
    
    OM_data = {
        'subject_name': subject_name,
        'OM_ids': [],
        'max_pressures': [],
        'weights': [],
        'normalization_info': []
    }
    
    for OM_dir in OM_dirs:
        om_id = os.path.basename(OM_dir.rstrip('/')).replace('OM', '')
        om_save_path = os.path.join(subject_save_path, f'OM{om_id}')
        os.makedirs(om_save_path, exist_ok=True)
        
        print(f'Processing {subject_name}/OM{om_id}...')
        max_fp, normalization_info = create_OM_chunks(
            args, OM_dir, om_id, args.chunk_size, args.sample_rate,
            args.data_type, args.view, weight=weight, save_path=om_save_path
        )
        
        OM_data['OM_ids'].append(om_id)
        OM_data['max_pressures'].append(max_fp)
        OM_data['weights'].append(weight)
        OM_data['normalization_info'].append(normalization_info)
    
    # Save subject-level normalization info
    with open(os.path.join(subject_save_path, 'normalization_info.pkl'), 'wb') as f:
        pickle.dump(OM_data, f)
   
    # Save processing arguments
    args_save_path = os.path.join(subject_save_path, 'args.json')
    with open(args_save_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
             
    return OM_data

def check_arg_consistency(args):
    if args.make_pressure_distribution and args.max_norm:
        raise ValueError("make_pressure_distribution and max_norm cannot both be True simultaneously.")

    if args.make_pressure_distribution and args.weight_norm:
        print("Warning: weight_norm is redundant when make_pressure_distribution is enabled. It will have no effect.")

    return args
 
def generate_save_prefix(args):
    # Format is {data_type}_{sample_rate}fps_{processing} where processing is distribution, maxnorm, weightnorm, none
    processing = []
    if args.make_pressure_distribution:
        processing.append('distribution')
    if args.max_norm:
        processing.append('maxnorm')
    if args.weight_norm:
        processing.append('weightnorm')
    if not processing:
        processing.append('none')
    processing_str = '_'.join(processing)
    save_prefix = f'{args.data_type}_{args.sample_rate}fps_{processing_str}' 
    return save_prefix
     
if __name__ == '__main__':
    print('Initializing...')
    args = parse_args()
    args = check_arg_consistency(args)
    
    # Auto-generate prefix if not provided
    if args.save_prefix is None:
        args.save_prefix = generate_save_prefix(args)
    
    # Print configuration
    print(f'Creating chunks of size {args.chunk_size} for {args.data_type} data')
    print(f'Save prefix: {args.save_prefix}\n')
    print('Processing settings:')
    for key, value in vars(args).items():
        print(f'  {key}: {value}')
    print()
    
    if args.OM:
        # Determine directory structure
        folder_names = [f.name for f in os.scandir(args.root_dir) if f.is_dir()]
        has_OM_dirs = any("OM" in name for name in folder_names)
        has_subject_dirs = not has_OM_dirs
        base_save_path = os.path.join(args.save_path, args.save_prefix)
       
        if has_OM_dirs and not has_subject_dirs:
            # Case A: Direct OM directories (single subject)
            subject_name = os.path.basename(os.path.normpath(args.root_dir))
            print(f'Processing OM data for {subject_name}\n')
            process_single_subject_OMs(args, args.root_dir, subject_name, base_save_path)
            
        elif has_subject_dirs:
            # Case B: Multiple subjects with OM subdirectories
            print('Processing OM data for multiple subjects\n')
            subject_dirs = sorted([f.path for f in os.scandir(args.root_dir) 
                                  if f.is_dir()])
            
            for subject_dir in subject_dirs:
                subject_name = os.path.basename(subject_dir)
                process_single_subject_OMs(args, subject_dir, subject_name, base_save_path)
        else:
            raise ValueError(f"Could not determine directory structure in {args.root_dir}")
        
    else:
        # Regular subject processing (non-OM)
        with open('assets/subject_weights.json', 'r') as f:
            weights = json.load(f)
        
        save_path = os.path.join(args.save_path, args.save_prefix)
        args.save_path = save_path  # Update args with full save path
        os.makedirs(save_path, exist_ok=True)
        
        subjects_data = {
            'subject_ids': [],
            'max_pressures': [],
            'weights': [],
            'normalization_info': []
        }
        
        for subject in range(1, 11):
            weight = weights[f'Subject{subject}']
            print(f'Processing Subject {subject}...')
            max_pressure, normalization_info = create_chunks(args, subject, weight)
            subjects_data['subject_ids'].append(subject)
            subjects_data['max_pressures'].append(max_pressure)
            subjects_data['weights'].append(weight)
            subjects_data['normalization_info'].append(normalization_info)
        
        with open(os.path.join(save_path, 'normalization_info.pkl'), 'wb') as f:
            pickle.dump(subjects_data, f)
    
        # Save processing arguments
        args_save_path = os.path.join(args.save_path, 'args.json')
        with open(args_save_path, 'w') as f:
            json.dump(vars(args), f, indent=4)
    
    print(f'\nProcessing complete! Data saved to: {os.path.join(args.save_path, args.save_prefix)}')