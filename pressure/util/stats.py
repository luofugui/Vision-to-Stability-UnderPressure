import numpy as np
from collections import defaultdict
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from pressure.util.util import cast_array

def l2_error(predictions, targets, frame_mask, gt_com=False, metric='mean'):
    """
    Compute Euclidean error for CoM predictions.
    """
    joint_dim = predictions.shape[-1] - 1  # Exclude confidence value
    # Apply frame mask
    valid_preds = predictions[frame_mask == 1]
    valid_targets = targets[frame_mask == 1]
  
    if gt_com:
        # Extract confidence values and create additional mask
        confidence_mask = targets[:, joint_dim] > 0
        valid_preds = valid_preds[confidence_mask[frame_mask == 1]][:, :joint_dim]
        valid_targets = valid_targets[confidence_mask[frame_mask == 1]][:, :joint_dim]
    else:
        valid_preds = valid_preds[:, :joint_dim]
        valid_targets = valid_targets[:, :joint_dim]
    # Compute Euclidean distance
    errors = np.linalg.norm(valid_preds - valid_targets, axis=-1)
    
    if metric == 'mean':
        error = np.mean(errors) if len(errors) > 0 else 0.0
        std= np.std(errors) if len(errors) > 0 else 0.0
    elif metric == 'median':
        error = np.nanmedian(errors)
        # Compute the rSTD 
        std = 1.4826 * np.nanmedian(np.abs(errors - error))
        
    return error,std 

def precision_recall_f1(predictions, targets, frame_mask=None, binary_contact=False):
    """ Compute precision and recall for contact predictions. """
    # Apply frame mask if provided
    if frame_mask is not None:
        frame_mask = np.asarray(frame_mask, dtype=bool)
        predictions = predictions[frame_mask]
        targets = targets[frame_mask]

    # Convert to boolean if necessary
    predictions = np.asarray(predictions, dtype=bool) if binary_contact else np.asarray(predictions)
    # Convert targets to binary if they contain probabilities
    if not np.array_equal(targets, targets.astype(bool)):  
        print("Warning: Converting targets to binary (threshold = 0.5)")
        targets = (targets >= 0.5).astype(int)
    targets = np.asarray(targets, dtype=bool)
    
    # Convert predictions to binary if necessary
    if binary_contact or not np.array_equal(predictions, predictions.astype(bool)):
        predictions = (predictions >= 0.5).astype(int)

    # Compute precision and recall
    tp = np.sum(predictions & targets)
    fp = np.sum(predictions & ~targets)
    fn = np.sum(~predictions & targets)

    # Compute precision, recall, and F1-score with numerical stability
    precision = tp / np.clip(tp + fp, a_min=1, a_max=None)
    recall = tp / np.clip(tp + fn, a_min=1, a_max=None)
    f1_score = 2 * precision * recall / np.clip(precision + recall, a_min=1e-10, a_max=None)

    return precision, recall, f1_score
    
def top_k_accuracy(predictions, targets, frame_mask=None, k_values=[1, 2, 3], binary_contact=False):
    """
    Compute top-k accuracy for contact predictions.

    Args:
        predictions: Array of shape (N, C) containing either:
            - Probabilities between 0 and 1 if binary_contact=False
            - Binary values (0 or 1) if binary_contact=True
        targets: Array of shape (N, C) containing binary ground truth (0 or 1)
        frame_mask: Optional boolean array of shape (N,) indicating valid frames
        k_values: List of k values for top-k accuracy calculation
        binary_contact: Whether predictions are binary values (True) or probabilities (False)

    Returns:
        Dictionary mapping k values to accuracy scores.
    """
    # Apply frame mask if provided
    if frame_mask is not None:
        frame_mask = np.asarray(frame_mask, dtype=bool)
        predictions = predictions[frame_mask]
        targets = targets[frame_mask]

    # Convert to boolean if necessary
    predictions = np.asarray(predictions, dtype=bool) if binary_contact else np.asarray(predictions)
    
    # Convert targets to binary if they contain probabilities
    if not np.array_equal(targets, targets.astype(bool)):  
        print("Converting targets to binary (threshold = 0.5)")
        targets = (targets >= 0.5).astype(int)
   
    if not np.array_equal(predictions, predictions.astype(bool)):
        print("Converting predictions to binary (threshold = 0.5)")
        predictions = (predictions >= 0.5).astype(int)

    if binary_contact:
        assert np.array_equal(predictions, predictions.astype(bool)), \
            "When binary_contact=True, predictions must be binary (0 or 1)"
    else:
        assert np.all((predictions >= 0) & (predictions <= 1)), \
            "When binary_contact=False, predictions must be probabilities between 0 and 1"

    top_k_accuracies = {}

    for k in k_values:
        if k > predictions.shape[1]:
            print(f"Warning: k={k} is larger than number of classes {predictions.shape[1]}")
            continue

        if binary_contact:
            # For binary contact predictions, take indices where predictions are 1
            top_k_indices = [np.where(pred == 1)[0] for pred in predictions]
            top_k_indices = [indices[:k] if len(indices) >= k else np.pad(indices, (0, k - len(indices)), constant_values=-1) for indices in top_k_indices]
            top_k_indices = np.array(top_k_indices)
        else:
            # For probability-based predictions, sort and take top k indices
            top_k_indices = np.argsort(predictions, axis=1)[:, -k:]

        # Compute correctness
        correct = np.array([
            np.any(targets[i, top_k_indices[i][top_k_indices[i] >= 0]]) if len(top_k_indices[i]) > 0 else False
            for i in range(len(predictions))
        ])

        top_k_accuracies[k] = np.mean(correct)

    return top_k_accuracies

def collect_metrics(metrics_list):
    """
    Collects and averages [mean, std] metrics across all subjects.
    Returns dict of means and stds for each metric type.
    """
    all_metrics = defaultdict(list)

    # Collect per-subject values
    for metrics in metrics_list:
        # --- Pressure ---
        if 'pressure' in metrics:
            for metric_type, values in metrics['pressure'].items():
                all_metrics[f'pressure_{metric_type}'].append(values)  # [mean, std]

        # --- COM ---
        if 'com' in metrics:
            all_metrics['l2_error_mean'].append(metrics['com']['l2_error_mean'])     # [mean, std]
            all_metrics['l2_error_median'].append(metrics['com']['l2_error_median']) # [mean, std]

        # --- Contact ---
        if 'contact' in metrics:
            all_metrics['topK'].append(metrics['contact']['topK'])  # dict: {1: ..., 2: ..., 3: ...}

        # --- Valid Frame Count ---
        if 'total_valid_frames' in metrics:
            all_metrics['total_valid_frames'].append(metrics['total_valid_frames'])

    # Build final results
    mean_metrics = {}

    # --- Pressure aggregation ---
    pressure_metrics = {k: v for k, v in all_metrics.items() if k.startswith('pressure_')}
    if pressure_metrics:
        mean_metrics['pressure'] = {}
        for k, vals in pressure_metrics.items():
            means = [v[0] for v in vals]
            stds  = [v[1] for v in vals]
            mean_metrics['pressure'][k.replace('pressure_', '')] = [
                float(np.mean(means)), float(np.std(means))
            ]

    # --- COM aggregation ---
    if 'l2_error_mean' in all_metrics:
        mean_vals  = [v[0] for v in all_metrics['l2_error_mean']]
        std_vals   = [v[1] for v in all_metrics['l2_error_mean']]
        median_vals = [v[0] for v in all_metrics['l2_error_median']]
        median_std  = [v[1] for v in all_metrics['l2_error_median']]

        mean_metrics['com'] = {
            'l2_error_mean':   [float(np.mean(mean_vals)),   float(np.std(mean_vals))],
            'l2_error_median': [float(np.mean(median_vals)), float(np.std(median_vals))]
        }

    # --- Contact top-K aggregation ---
    if 'topK' in all_metrics:
        topk_dicts = all_metrics['topK']  # list of dicts
        keys = topk_dicts[0].keys()       # e.g., 1, 2, 3

        mean_metrics['contact'] = {'topK': {}}
        for k in keys:
            vals = [d[k] for d in topk_dicts]
            mean_metrics['contact']['topK'][str(k)] = [
                float(np.mean(vals)), float(np.std(vals))
            ]

    # --- Total valid frames ---
    if 'total_valid_frames' in all_metrics:
        valid_counts = all_metrics['total_valid_frames']
        mean_metrics['total_valid_frames'] = int(np.mean(valid_counts))

    return mean_metrics

def print_evaluation_results(metrics, logger):
    """Print evaluation results for all modalities in a clean format."""
    logger.info("\nEvaluation results:")
    
    # Handle pressure metrics
    if 'pressure' in metrics:
        logger.info("\n\nPressure Metrics:")
        pressure_metrics = metrics['pressure']
        for metric_name, values in pressure_metrics.items():
            logger.info(f"    {metric_name}: {round(values[0], 3)}")
    
    # Handle COM metrics
    if 'com' in metrics:
        logger.info("\n\nCenter of Mass Metrics:")
        logger.info(f"  CoM L2 Mean Error: {round(metrics['com']['l2_error_mean'][0], 3)}")
        logger.info(f"  CoM L2 Median Error: {round(metrics['com']['l2_error_median'][0], 3)}")
    
    # Handle contact metrics
    if 'contact' in metrics:
        logger.info("\n\nContact Metrics:")
        for k, value in metrics['contact']['topK'].items():
            logger.info(f"    Top-{k} Accuracy: {round(value, 3)}")
            
def get_metrics(calc):
    mean = np.nanmean(calc, axis=0)
    std = np.nanstd(calc, axis=0)
    median = np.nanmedian(calc, axis=0)
    max = np.nanmax(calc, axis=0)
    min = np.nanmin(calc, axis=0)
    # Count the non-NaN entries and divide by the length along the first dimension
    non_nan_count = np.sum(~np.isnan(calc), axis=0) / calc.shape[0]
    stats = [mean, std, median, max, min, non_nan_count]  
    return stats

def norm_vector(vector, keep_dims=True):
    # Assuming vector is now a 2D array: (batch_size, active_pixels)
    vec_sum = np.sum(vector, axis=1, keepdims=keep_dims)
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized_vec = vector / vec_sum 
    
    # If neg, set to 0
    normalized_vec[normalized_vec < 0] = 0
    return normalized_vec

def IG_stable(gt, pred, frame_mask=None):
    eps = np.finfo(float).eps
    neg_gt = (gt > 0).astype(int)
    
    first = np.log2(eps + pred * neg_gt)
    second = np.log2(eps + gt * neg_gt)
    diff = first - second
    ig = np.mean(diff, axis=1)  # Mean over active pixels for each sample
    if frame_mask is not None:
        ig = ig * frame_mask
    return ig

def kld_stable(gt_norm, pred_norm, frame_mask=None):
    eps = np.finfo(float).eps
    gt_norm_clipped = np.clip(gt_norm, a_min=eps, a_max=1-eps)
    pred_norm_clipped = np.clip(pred_norm, a_min=eps, a_max=1-eps)
    
    ratio = gt_norm_clipped / (pred_norm_clipped + eps)
    log_ratio = np.log(eps + ratio)
    
    KLD = np.sum(gt_norm_clipped * log_ratio, axis=1)  # Sum over active pixels for each sample
   
    if frame_mask is not None:
        KLD = KLD * frame_mask 
    return KLD

def safe_divide(a, b, fill_value=0.0):
    return np.divide(a, b, out=np.full_like(a, fill_value), where=b!=0)

def mape(gt, pred, frame_mask=None):
    ret = np.mean(np.abs(safe_divide(pred - gt, gt)), axis=1) * 100
    if frame_mask is not None:
        ret = ret * frame_mask
    return ret

def weight_normalized_mae(gt, pred, weight, frame_mask=None):
    gt_pressure = safe_divide(gt, weight)
    pred_pressure = safe_divide(pred, weight)
    ret = np.mean(np.abs(pred_pressure - gt_pressure), axis=1) 
    if frame_mask is not None:
        ret = ret * frame_mask
    return ret

def rmse(gt, pred, frame_mask=None):
    ret = np.sqrt(np.mean((gt - pred)**2, axis=1))
    if frame_mask is not None:
        ret = ret * frame_mask
    return ret

def nrmse(gt, pred, frame_mask=None):
    rmse_val = np.sqrt(np.mean((gt - pred)**2, axis=1))
    range_gt = np.max(gt, axis=1) - np.min(gt, axis=1)
    ret = safe_divide(rmse_val, range_gt)
    if frame_mask is not None:
        ret = ret * frame_mask
    return ret
    

def mae(gt, pred, frame_mask=None):
    mae = np.mean(np.abs(pred - gt), axis=1)  # Mean over active pixels for each sample
    if frame_mask is not None:
        mae = mae * frame_mask
    return mae

def sim(gt, pred, frame_mask=None):
    min_values = np.minimum(gt, pred)
    sim = np.sum(min_values, axis=1)  # Sum over active pixels for each sample
    if frame_mask is not None:
        sim = sim * frame_mask 
    return sim

def remove_inactive_pixels(x, foot_mask):
    if x[0].shape != foot_mask.shape:
        x = x.reshape(-1, *foot_mask.shape)
        
    expanded_mask = np.expand_dims(foot_mask, axis=0)
    expanded_mask = np.repeat(expanded_mask, x.shape[0], axis=0)
    valid_mask = ~np.isnan(expanded_mask)
    
    def remove_inactive(sample, mask):
        return sample[mask]
  
    # Apply the function to each sample in the batch
    x_valid = np.array([remove_inactive(sample, mask) for sample, mask in zip(x, valid_mask)])
    
    return x_valid

def psnr(pred, target, use_normalized=False, frame_mask=None):
    assert pred.shape == target.shape, "Shapes of pred and target must match"
    
    if use_normalized:
        pred = np.clip(pred, 0, 1)
        target = np.clip(target, 0, 1)
        data_range = 1
    else:
        # If using kPa, calculate the data range
        data_range = np.max(target) - np.min(target)
    
    # Calculate PSNR for each sample in the batch
    psnr_values = []
    for p, t in zip(pred, target):
        if len(p) > 0:  
            psnr = peak_signal_noise_ratio(t, p, data_range=data_range)
            psnr_values.append(psnr)
   
    psnr_values = np.array(psnr_values) 
    # Apply frame mask if provided
    if frame_mask is not None:
        psnr_values *= frame_mask
    
    return psnr_values

def ssim(pred, target, use_normalized=False, frame_mask=None):
    assert pred.shape == target.shape, "Shapes of pred and target must match"
    
    if use_normalized:
        pred = np.clip(pred, 0, 1)
        target = np.clip(target, 0, 1)
        data_range = 1
    else:
        data_range = np.max(target) - np.min(target)
    
    ssim_values = []
    for p, t in zip(pred, target):
        if len(p) > 0:
            # Reshape to 2D for SSIM calculation
            side_length = int(np.sqrt(len(p)))
            p_2d = p[:side_length**2].reshape(side_length, side_length)
            t_2d = t[:side_length**2].reshape(side_length, side_length)
            
            ssim = structural_similarity(t_2d, p_2d, data_range=data_range)
            ssim_values.append(ssim)
   
    ssim_values = np.array(ssim_values)
    if frame_mask is not None:
        ssim_values *= frame_mask
         
    return ssim_values

def calc_stats(gt, pred, foot_mask, subject, sub_weight, frame_mask=None, writer=None, global_step=0, active_only=True):
    gt_kpa = cast_array(gt, np.float32)
    pred_kpa = cast_array(pred, np.float32)
   
    if active_only:
        pred_kpa, gt_kpa = remove_inactive_pixels(pred_kpa, foot_mask), remove_inactive_pixels(gt_kpa, foot_mask)
        
    psnr_ = psnr(pred_kpa, gt_kpa, use_normalized=False, frame_mask=frame_mask)
    psnr_stats = get_metrics(psnr_)
   
    ssim_ = ssim(pred_kpa, gt_kpa, use_normalized=False, frame_mask=frame_mask)
    ssim_stats = get_metrics(ssim_)
    
    mae_ = mae(gt_kpa, pred_kpa, frame_mask)
    mae_stats = get_metrics(mae_)
    
    rmse_ = rmse(gt_kpa, pred_kpa, frame_mask=frame_mask)
    rmse_stats = get_metrics(rmse_)
    
    weight_norm_mae = weight_normalized_mae(gt_kpa, pred_kpa, sub_weight, frame_mask=frame_mask)
    weight_norm_mae_stats = get_metrics(weight_norm_mae) 
    
    normalized_rmse = nrmse(gt_kpa, pred_kpa, frame_mask=frame_mask)
    normalized_rmse_stats = get_metrics(normalized_rmse)
    
    mape_ = mape(gt_kpa, pred_kpa, frame_mask=frame_mask)
    mape_stats = get_metrics(mape_)
    
    gt_norm = norm_vector(gt_kpa)
    pred_norm = norm_vector(pred_kpa)
    
    sim_ = sim(gt_norm, pred_norm, frame_mask)
    sim_stats = get_metrics(sim_)
        
    kld = kld_stable(gt_norm, pred_norm, frame_mask)
    kld_stats = get_metrics(kld)
     
    ig = IG_stable(gt_norm, pred_norm, frame_mask)
    ig_stats = get_metrics(ig)
    
    metrics = {
        'mae': mae_stats,
        'sim': sim_stats,
        'kld': kld_stats,
        'ig': ig_stats,
        'psnr': psnr_stats,
        'ssim': ssim_stats,
        'rmse': rmse_stats, 
        'Mean Absolute Percentage Error': mape_stats,
        'Normalized RMSE': normalized_rmse_stats,
        'Weight Normalized MAE': weight_norm_mae_stats,
    }
   
    if writer is not None: 
        writer.add_scalar(f'Eval/Subject_{subject}/MAE', mae_stats[0], global_step)
        writer.add_scalar(f'Eval/Subject_{subject}/SIM', sim_stats[0], global_step)
        writer.add_scalar(f'Eval/Subject_{subject}/KLD', kld_stats[0], global_step)
        writer.add_scalar(f'Eval/Subject_{subject}/IG', ig_stats[0], global_step)
        writer.add_scalar(f'Eval/Subject_{subject}/PSNR', psnr_stats[0], global_step)
        writer.add_scalar(f'Eval/Subject_{subject}/SSIM', ssim_stats[0], global_step)
        writer.add_scalar(f'Eval/Subject_{subject}/rMSE', rmse_stats[0], global_step)
        writer.add_scalar(f'Eval/Subject_{subject}/Mean_Absolute_Percentage_Error', mape_stats[0], global_step)
        writer.add_scalar(f'Eval/Subject_{subject}/Normalized_RMSE', normalized_rmse_stats[0], global_step)
        writer.add_scalar(f'Eval/Subject_{subject}/Weight_Normalized_MAE', weight_norm_mae_stats[0], global_step)
   
    return metrics