import argparse
import pickle
import json
import os
import re
import numpy as np
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm

from pressure.util.util import *
from pressure.util.selection import *
from pressure.util.stats import * 
from pressure.util.manager import create_experiment
from pressure.util.pipeline import Pipeline
from pressure.data.data_support import DataNormalizer

def create_zero_pressure_mask(pressure_batch, active_only):
    """Create a mask for frames with zero pressure."""
    if active_only:
        return torch.sum(pressure_batch, dim=1) == 0
    else:
        return torch.sum(pressure_batch.reshape(pressure_batch.shape[0], -1), dim=1) == 0

def eval_model(model, test_loader, result_save_dir, data_id, experiment_manager,
               cfg, logger, writer=None, global_step=0):
    """
    Evaluate model performance on test data.
    """
    device = cfg.default.device
    model_dtype = next(model.parameters()).dtype
    model = model.to(device)
    model.eval()
 
    is_underpressure = getattr(cfg.data, 'dataset', 'psu').lower() == 'underpressure'
    normalizer = None if is_underpressure else DataNormalizer(cfg.default.data_path, cfg=cfg)
    if normalizer is not None:
        normalizer.verify_consistency(cfg)

    outputs = defaultdict(list)
    targets = defaultdict(list)
    frame_mask = []
    seq_len = cfg.data.sequence_length
    
    # Determine whether to reconstruct full maps for saving
    reconstruct_full = cfg.data.active_only
  
    included = {
        'invalid_com': [],
        'invalid_sequence': [],
        'invalid_middle_confs': [],
        'invalid_pressure': [],
        'total_valid': [],
        'total_frames': []
    }

    print(f"[INFO] Evaluating on device: {device}, data_id: {data_id}") 
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader)):
            # Process batch
            batch = {k: v.to(device).to(model_dtype) for k, v in batch.items()}
            batch['joint'] = batch['joint'].squeeze(2)
            if 'pressure' in batch:
                batch['pressure'] = batch['pressure'].view(-1, np.prod(batch['pressure'].size()[1:]))
            
            center_idx = seq_len // 2
            center_zero_count = torch.sum(batch['joint'][:, center_idx, :, -1] == 0, dim=-1)
            middle_valid = center_zero_count <= cfg.eval.zero_conf_thresh
            frame_valid = torch.all(batch['joint'][..., -1] != 0, dim=-1)
            non_center_frames = torch.cat([frame_valid[:, :center_idx], frame_valid[:, center_idx+1:]], dim=1)
            bad_frame_count = torch.sum(~non_center_frames, dim=1)
            other_frames_valid = bad_frame_count <= cfg.eval.bad_frames_in_seq_thresh
            if 'pressure' in batch and not is_underpressure:
                zero_pressure_mask = create_zero_pressure_mask(batch['pressure'], cfg.data.active_only)
                pressure_valid = ~zero_pressure_mask
            else:
                pressure_valid = torch.ones_like(middle_valid, dtype=torch.bool)
           
            # Check com if needed
            if cfg.data.gt_com and 'com' in batch:
                com_valid = batch['com'][:, -1] != 0
            else:
                com_valid = torch.ones_like(middle_valid, dtype=torch.bool)
           
            mask = middle_valid & other_frames_valid & pressure_valid & com_valid 
            # Track validation stats for this iteration
            batch_size = len(middle_valid)
            included['invalid_com'].append((i, torch.sum(~com_valid).item()))
            included['invalid_sequence'].append((i, torch.sum(~other_frames_valid).item()))
            included['invalid_middle_confs'].append((i, torch.sum(~middle_valid).item()))
            included['invalid_pressure'].append((i, torch.sum(~pressure_valid).item()))
            included['total_valid'].append((i, torch.sum(mask).item()))
            included['total_frames'].append((i, batch_size))
            frame_mask.extend(mask.cpu().numpy())
           
            # Get model predictions
            output = model(batch['joint'])
            
            # Collect outputs and targets
            for key in output:
                outputs[key].extend(output[key].cpu().numpy())
                targets[key].extend(batch[key].cpu().numpy())
            
            # Collect middle frame joints for CoM visualization
            if 'com' in cfg.default.mode:
                outputs['middle_frame_joints'].extend(batch['middle_frame_joints'].cpu().numpy())
              
    # Print detailed summary
    print("\n=== Frame Validation Summary ===")
    total_iterations = len(included['total_frames'])
    overall_stats = {}

    for key in ['invalid_com', 'invalid_sequence', 'invalid_middle_confs', 'invalid_pressure', 'total_valid', 'total_frames']:
        total = sum(count for _, count in included[key])
        overall_stats[key] = total
        print(f"{key}: {total} across {total_iterations} iterations")

    print(f"\nOverall validation rate: {overall_stats['total_valid']}/{overall_stats['total_frames']} = {overall_stats['total_valid']/overall_stats['total_frames']*100:.2f}%")
    
    # Convert lists to arrays
    for key in outputs:
        outputs[key] = np.array(outputs[key])
        targets[key] = np.array(targets[key])
    
    # Process frame mask
    frame_mask = np.array(frame_mask, dtype=np.float32)
    frame_mask[frame_mask == 0] = np.nan
    # Total number of non-masked frames
    total_valid_frames = np.sum(~np.isnan(frame_mask)).item()
    logger.info(f"\nTotal valid frames: {total_valid_frames}\n")
    
    # Denormalize data before saving or calculating metrics
    if 'com' in outputs and normalizer is not None:
        outputs['com'] = normalizer.denormalize_com(outputs['com'], data_id=data_id)
        targets['com'] = normalizer.denormalize_com(targets['com'], data_id=data_id)
        outputs['middle_frame_joints'] = normalizer.denormalize_joints(outputs['middle_frame_joints'], data_id=data_id) 

    # Denormalize pressure data if present
    if 'pressure' in outputs and normalizer is not None:
        outputs['pressure'] = normalizer.unnormalize_and_scale_to_kpa(
            outputs['pressure'], data_id, cfg, reconstruct_full=reconstruct_full
        )
        targets['pressure'] = normalizer.unnormalize_and_scale_to_kpa(
            targets['pressure'], data_id, cfg, reconstruct_full=reconstruct_full
        )
        
    # Denormalize contact data if present
    if 'contact' in outputs and normalizer is not None:
        outputs['contact'] = normalizer.unnormalize_contact(
            outputs['contact'], data_id, cfg, reconstruct_full=False, apply_sigmoid=~cfg.data.binary_contact
        )
        targets['contact'] = normalizer.unnormalize_contact(
            targets['contact'], data_id, cfg, reconstruct_full=False
        )

    # Save outputs if enabled
    if cfg.eval.save_output:
        os.listdir(result_save_dir)
        output_dir = os.path.join(result_save_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        output_dict = {
            'predictions': outputs,
            'targets': targets,
            'frame_mask': frame_mask,
        }
    
        if 'pressure' in outputs:
            output_dict['pressure_sums'] = np.nansum(targets['pressure'].reshape(len(targets['pressure']), -1), axis=1)

        if cfg.default.om:
            save_path = os.path.join(output_dir, f'OM{data_id}_output.pkl')
        else:
            save_path = os.path.join(output_dir, f'subject{data_id}_output.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(output_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Create videos if enabled
    if cfg.viz.video.enabled:
        start_time = time.time()
        logger.info("Creating visualization videos...")
        video_dir = Path(result_save_dir) / f'Subject{data_id}' / 'videos'
        video_dir.mkdir(parents=True, exist_ok=True)

        try:
            experiment_manager.visualizer.create_modality_videos(
                predictions=outputs,
                targets=targets,
                subject=data_id,
                frame_mask=None#frame_mask,
            )
            logger.info(f"Visualization videos created in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Failed to create visualization videos: {str(e)}")
            logger.exception(e)
            
    # Calculate metrics for pressure data
    metrics = {}
    metrics['total_valid_frames'] = total_valid_frames
    for key in outputs:
        if key == 'middle_frame_joints':
            continue
        metrics[key] = {}
        if key == 'pressure' and normalizer is not None:
            # Data is already in kPa
            pred_kpa = outputs['pressure']
            gt_kpa = targets['pressure']
            
            sub_weight = normalizer.get_subject_weight(data_id)
            foot_mask = np.load('assets/foot_mask_nans.npy')
            metrics['pressure'] = calc_stats(gt_kpa, pred_kpa, foot_mask, data_id, sub_weight, 
                                        frame_mask=frame_mask, writer=writer, 
                                        global_step=global_step, active_only=cfg.data.active_only)
        elif key == 'pressure':
            pred = outputs['pressure']
            gt = targets['pressure']
            valid = frame_mask.reshape(-1, *([1] * (pred.ndim - 1)))
            err = (pred - gt) * valid
            metrics['pressure']['mse'] = np.nanmean(err ** 2).item()
            metrics['pressure']['mae'] = np.nanmean(np.abs(err)).item()
            metrics['pressure']['rmse'] = np.sqrt(metrics['pressure']['mse']).item()
        elif key == 'com':
            metrics['com']['l2_error_mean'] = l2_error(outputs['com'], targets['com'], frame_mask, gt_com=cfg.data.gt_com)
            metrics['com']['l2_error_median'] = l2_error(outputs['com'], targets['com'], frame_mask, gt_com=cfg.data.gt_com, metric='median')
        elif key == 'contact':
            metrics['contact']['topK'] = top_k_accuracy(outputs['contact'], targets['contact'], frame_mask, binary_contact=cfg.data.binary_contact)
        else:
            continue
    
    # Save results
    if cfg.default.om:
        save_path = os.path.join(result_save_dir, 'eval', f'OM_{data_id}.json')
    else:
        save_path = os.path.join(result_save_dir, f'subject{data_id}.json')
        
    logger.info(f'Saving results to {save_path}')
    with open(save_path, 'w') as f:
        json.dump(metrics, f, cls=NumpyEncoder, indent=4)

    metrics['total_valid_frames'] = total_valid_frames
    print_evaluation_results(metrics, logger)
    return metrics

def main(cfg):
    """Main evaluation function."""
    # Create experiment manager
    manager = create_experiment(cfg)
    logger = manager.logger
    
    # Load foot mask
    foot_mask = np.load('assets/foot_mask_nans.npy')
    foot_mask[np.isnan(foot_mask)] = 0
    foot_mask = torch.tensor(foot_mask).to(torch.float32).to(cfg.default.device)
   
    # Setup pipeline
    pipeline = Pipeline(manager)
    if cfg.default.om:
        logger.info("Evaluating Ordinary Movement (OM) data")
        
        subject_name = cfg.default.data_path.split('/')[-1]
        results_dir = manager.experiment_path / subject_name 
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(results_dir / 'eval', exist_ok=True)
      
        # List all of the OMs folders in the data path
        om_folders = [f for f in os.listdir(cfg.default.data_path) if 'OM' in f]
        om_folders.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))  # Sort by OM index
        
        pipeline.setup_training('OM', skip_support=True)
      
        loaded=False 
        all_metrics = [] 
        for om_folder in om_folders:
            om_idx = extract_idx(om_folder)
            om_path = os.path.join(cfg.default.data_path, om_folder)
            
            print(f"Processing OM {om_idx} data ...")
             
            # Create dataset for OM
            _, _, test_dataset = create_dataset(cfg, subject=None,om_idx=om_idx)
            test_loader = DataLoader(
                test_dataset,
                batch_size=cfg.training.batch_size,
                shuffle=False,
                num_workers=cfg.training.dataloader_workers
            )

            # Load model checkpoint
            checkpoint_path = Path(f'{cfg.default.checkpoint_path}/model_{cfg.eval.checkpoint_flag}.pth')
            if not checkpoint_path.exists():
                logger.warning(f'Checkpoint {checkpoint_path} does not exist')
                return
            
            logger.info(f'Evaluating OM data using checkpoint {checkpoint_path}')
            if not loaded:
                pipeline.load_checkpoint(checkpoint_path, model_only=True)
                loaded=True
            
            # Evaluate
            metrics = eval_model(
                model=pipeline.model, 
                test_loader=test_loader, 
                result_save_dir=results_dir, 
                data_id=f'{om_idx}',  # Indicate OM evaluation
                experiment_manager=manager,
                cfg=cfg, 
                writer=manager.writer, 
                logger=logger
            )
            all_metrics.append(metrics)
    else: 
        eval_dir = manager.experiment_path / 'eval'
        eval_dir.mkdir(exist_ok=True)
        # Process each subject
        checkpoint_flag = cfg.eval.checkpoint_flag
        checkpoint_base = manager.experiment_path / 'checkpoints'
        checkpoint_paths = list(checkpoint_base.glob('Subject*'))
        # Sort them based on the last part of the folder (*/Subject{subject_id})
        checkpoint_paths.sort(key=lambda x: int(re.findall(r'\d+', x.name)[0])) 
        
        all_metrics = [] 
        for sub_path in checkpoint_paths:
            subject = int(re.findall(r'\d+', sub_path.name)[0])
            checkpoint = sub_path / f'model_{checkpoint_flag}.pth'
            
            if not checkpoint.exists():
                logger.warning(f'Checkpoint {checkpoint} does not exist')
                continue
                
            logger.info(f'Evaluating subject {subject} using checkpoint {checkpoint}')
          
            if cfg.data.chunk_data:
                # Create dataset and loader
                _, _, test_dataset = create_dataset(cfg, subject)
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=cfg.training.batch_size,
                    shuffle=False,
                    num_workers=cfg.training.dataloader_workers
                )
                
                # Setup pipeline and evaluate
                pipeline.setup_training(subject, skip_support=True)
                pipeline.load_checkpoint(checkpoint, model_only=True)
                
                metrics = eval_model(
                    model=pipeline.model, 
                    test_loader=test_loader, 
                    result_save_dir=eval_dir, 
                    experiment_manager=manager,  # Pass the experiment manager
                    data_id=subject,
                    cfg=cfg, 
                    writer=manager.writer, 
                    logger=logger
                )
                all_metrics.append(metrics)
            else:
                raise NotImplementedError("Non-chunked data evaluation not supported")
  
    # Save mean metrics
    mean_metrics = collect_metrics(all_metrics)
    if cfg.default.om:
        eval_dir = manager.experiment_path / subject_name / 'eval'
    else:
        eval_dir = manager.experiment_path / 'eval'
    metrics_path = eval_dir / 'mean_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(mean_metrics, f, cls=NumpyEncoder, indent=4)
                 
    manager.close()
    
    return all_metrics
    
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Evaluate pressure prediction model')
    argparser.add_argument('--config', type=str, default='configs/pns.yaml', 
                          help='Path to the config file')
    args = argparser.parse_args()
    
    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        print(f"Config file {args.config} not found.")
    
    main(cfg)
