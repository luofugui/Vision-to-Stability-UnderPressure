from typing import Dict, Tuple, Any
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm

from pressure.util.util import *
from pressure.util.manager import *
from pressure.util.selection import *
from pressure.data.data_support import *

class Pipeline:
    def __init__(self, manager):
        """
        Args:
            manager: ExperimentManager instance
            foot_mask: Preprocessed foot mask tensor
        """
        self.manager = manager
        self.cfg = manager.config
        self.logger = manager.logger
        self.writer = manager.writer
        self.visualizer = manager.visualizer
        self.device = self.cfg.default.device
        self.nan_foot_mask = np.load('assets/foot_mask_nans.npy')
        self.current_epoch = 0
        
        # Training attributes set in setup_training
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.current_subject = None
        
        if 'contact' in self.cfg.default.mode:
            self.contact_config = ContactMapConfig(
                use_regions=True,
                contact_threshold=self.cfg.data.contact_threshold,
                num_regions=self.cfg.data.num_regions,
                active_only=self.cfg.data.active_only
            )
            self.contact_processor = PressureMapProcessor(self.contact_config)
        else:
            self.contact_processor = None
                
    def setup_training(self, subject, dataset_size=None, skip_support=False):
        """Setup model, optimizer, scheduler, and loss function for a subject."""
        from pressure.util.selection import select_model, select_train_support
       
        self.current_subject = subject 
        self.model = select_model(self.cfg)
        self.model.to(self.device)
        self.pixel2region = None
        if not skip_support: 
            if 'contact' in self.cfg.default.mode:
                self.pixel2region = self.contact_config.pixel_regions
            self.optimizer, self.scheduler, self.loss_fn = select_train_support(self.cfg, self.model, dataset_size)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, subject, epoch_completed=0) -> Tuple[Any, int]:
        """Run complete training loop for a subject."""
        if self.current_subject != subject:
            self.current_subject = subject
            
        if self.cfg.training.per_sub_epochs is not None and subject in self.cfg.training.per_sub_epochs:
            epochs = self.cfg.training.per_sub_epochs[subject]
        else:
            epochs = self.cfg.training.epochs
            
        best_smoothed_loss = float('inf')
        val_loss_history = []
        no_improvement_counter = 0
        val_every = self.cfg.training.val_every
        global_step = 0
        smoothing_window = self.cfg.training.smoothing_window
        mem_usage = []
        
        # Initialize loss tracking
        train_losses = defaultdict(lambda: {'values': [], 'steps': []})
        val_losses = defaultdict(lambda: {'values': [], 'steps': []})
        grad_norms = {'pre_clip': [], 'post_clip': [], 'clip_freq': [], 'steps': []}
       
        for epoch in tqdm(range(epoch_completed, epochs)):
            # Train one epoch
            self.current_epoch = epoch
            batch_losses, epoch_losses, global_step, grad_norms = self.train_epoch(train_loader, global_step, subject, grad_norms)
            
            # Record training losses
            for loss_name, loss_data in batch_losses.items():
                train_losses[loss_name]['values'].extend(loss_data['values'])
                train_losses[loss_name]['steps'].extend(loss_data['steps'])
            
            # Log learning rate
            self.writer.add_scalar(f'Hparams/Subject_{subject}/lr', 
                                self.scheduler.get_last_lr()[0], 
                                global_step=global_step)
            self.scheduler.step()
           
            # Validation
            if epoch % val_every == 0:
                val_batch_losses = self.validate(val_loader, epoch, subject, global_step)
                val_loss_history.append(np.mean(val_batch_losses['total']['values']))
                
                # Record validation losses
                for loss_name, loss_data in val_batch_losses.items():
                    val_losses[loss_name]['values'].extend(loss_data['values'])
                    val_losses[loss_name]['steps'].extend(loss_data['steps'])
                
                # Update moving average of validation loss
                smoothed_loss = sum(val_loss_history[-smoothing_window:]) / min(len(val_loss_history), smoothing_window)
                
                # Initialize best loss if first validation
                if best_smoothed_loss == float('inf') or smoothed_loss < best_smoothed_loss - (best_smoothed_loss * 0.002):
                    best_smoothed_loss = smoothed_loss
                    no_improvement_counter = 0
                    self.manager.save_checkpoint(epoch+1, self.model, self.optimizer, 
                                                self.scheduler, flag='best', subject=subject)
                    self.manager.save_checkpoint(epoch+1, self.model, self.optimizer,
                                                self.scheduler, flag='latest', subject=subject)
                else:
                    no_improvement_counter += val_every
                    # save latest model
                    self.manager.save_checkpoint(epoch+1, self.model, self.optimizer,
                                                self.scheduler, flag='latest', subject=subject)
                
                # Create and save learning curves
                curves_path = self.manager.experiment_path / 'visualizations' / f'Subject{subject}' / 'learning_curves'
                curves_path.mkdir(parents=True, exist_ok=True)
                
                self.visualizer.create_learning_curves(
                    train_losses,
                    curves_path / 'train_losses.png',
                    title=f'Training Losses - Subject {subject}',
                    grad_norms=grad_norms
                )
                self.visualizer.create_learning_curves(
                    val_losses,
                    curves_path / 'val_losses.png',
                    title=f'Validation Losses - Subject {subject}'
                )
            else:
                self.manager.save_checkpoint(epoch+1, self.model, self.optimizer, self.scheduler, flag='latest', subject=subject)
            
            if no_improvement_counter >= self.cfg.training.early_stop_patience:
                self.logger.info(f'Early stopping triggered after {epoch+1} epochs.')
                break
            
        return self.model, global_step
   
    def track_norms(self, global_step, grad_norms):
        # Track pre-clip gradient norm
        pre_clip_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) 
                                            for p in self.model.parameters() if p.grad is not None]))
        grad_norms['pre_clip'].append(pre_clip_norm.item())
        grad_norms['steps'].append(global_step)
        
        # Clip gradient
        _ = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.cfg.training.max_grad_norm
        )
        
        # Calculate clip frequency based on whether clipping actually occurred
        was_clipped = pre_clip_norm > self.cfg.training.max_grad_norm
        grad_norms['clip_freq'].append(1.0 if was_clipped else 0.0)
        
        # Track post-clip gradient norm
        post_clip_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) 
                                        for p in self.model.parameters() if p.grad is not None]))
        grad_norms['post_clip'].append(post_clip_norm.item())
        
        return grad_norms
        
    def train_epoch(self, train_loader: DataLoader, global_step, subject, grad_norms):
        """Run one epoch of training."""
        self.model.train()
        batch_losses = defaultdict(lambda: {'values': [], 'steps': []})  # Store individual values and steps
        network_dtype = next(self.model.parameters()).dtype
        mem_usages, gpu_usages = [], []
         
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(self.device).to(network_dtype) for k, v in batch.items()}
            batch['joint'] = batch['joint'].squeeze(2)
            
            if 'pressure' in batch:
                batch['pressure'] = batch['pressure'].view(-1, np.prod(batch['pressure'].size()[1:]))
          
            self.optimizer.zero_grad()
            output = self.model(batch['joint'])
            loss = self.loss_fn(output, batch)
            loss['total'].backward()
            grad_norms = self.track_norms(global_step, grad_norms) 
            self.optimizer.step()
           
            # Store individual loss values with corresponding steps
            for loss_name, loss_value in loss.items():
                batch_losses[loss_name]['values'].append(loss_value.item())
                batch_losses[loss_name]['steps'].append(global_step)
       
            # Monitor memory usage
            mem_cpu = self.manager.memory_monitor.get_memory_mb()
            mem_gpu = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
            mem_usages.append(mem_cpu)
            gpu_usages.append(mem_gpu)

            
            if i % self.cfg.default.log_interval == 0:
                recent_losses = {
                    name: np.mean(loss_data['values'][-self.cfg.default.log_interval:])
                    for name, loss_data in batch_losses.items()
                }
                recent_pre_clip = np.mean(grad_norms['pre_clip'][-self.cfg.default.log_interval:])
                clip_freq = grad_norms['clip_freq'][-1]

                self.manager.logger.info(
                    f"[Epoch {self.current_epoch:02d} | Batch {i:04d}] "
                    f"Loss={recent_losses.get('total',0):.4f} "
                    f"(comp: {', '.join([f'{k}:{v:.4f}' for k,v in recent_losses.items() if k!='total'])}) | "
                    f"GradNorm pre={recent_pre_clip:.3f}, clip_freq={clip_freq:.2f} | "
                    f"CPU={mem_cpu:.1f} MB GPU={mem_gpu:.1f} MB"
                )
            global_step += 1
        
        # Calculate epoch averages for console output only
        epoch_losses = {k: np.mean(v['values']) for k, v in batch_losses.items()}
        avg_cpu = np.mean(mem_usages)
        avg_gpu = np.mean(gpu_usages)
        max_cpu = np.max(mem_usages)
        max_gpu = np.max(gpu_usages)
        peak_gpu = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

        self.manager.logger.info(
            f"[Epoch {self.current_epoch:02d}] Subject {subject}: "
            f"Loss summary: {', '.join([f'{k}:{v:.4f}' for k,v in epoch_losses.items()])} | "
            f"GradNorm mean={np.mean(grad_norms['pre_clip']):.3f} | "
            f"CPU avg/peak={avg_cpu:.1f}/{max_cpu:.1f} MB | "
            f"GPU avg/peak={avg_gpu:.1f}/{max_gpu:.1f} MB | "
            f"GPU peak alloc={peak_gpu:.1f} MB"
        )

        return batch_losses, epoch_losses, global_step, grad_norms

    @torch.no_grad()
    def validate(self, val_loader: DataLoader, epoch, subject, global_step) -> Dict[str, Dict[str, list]]:
        """Run validation."""
        self.model.eval()
        batch_losses = defaultdict(lambda: {'values': [], 'steps': []})
        network_dtype = next(self.model.parameters()).dtype
        is_underpressure = getattr(self.cfg.data, 'dataset', 'psu').lower() == 'underpressure'
        normalizer = None if is_underpressure else DataNormalizer(self.cfg.default.data_path, cfg=self.cfg)
        
        outputs = defaultdict(list)
        targets = defaultdict(list)
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device).to(network_dtype) for k, v in batch.items()}
                batch['joint'] = batch['joint'].squeeze(2)
                if 'pressure' in batch:
                    batch['pressure'] = batch['pressure'].view(-1, np.prod(batch['pressure'].size()[1:]))
                
                output = self.model(batch['joint'])
                loss = self.loss_fn(output, batch)
                
                # Store individual loss values
                for loss_name, loss_value in loss.items():
                    batch_losses[loss_name]['values'].append(loss_value.item())
                    batch_losses[loss_name]['steps'].append(global_step)
                
                # Collect outputs and targets for each modality
                for key in output:
                    if key == 'com' and normalizer is not None:
                        pred_com = output[key].cpu().numpy()
                        target_com = batch[key].cpu().numpy()
                        pred_com_denorm = normalizer.denormalize_com(pred_com, subject)
                        target_com_denorm = normalizer.denormalize_com(target_com, subject)
                        joints = batch['middle_frame_joints'].cpu().numpy()
                        denorm_joints = normalizer.denormalize_joints(joints, subject)
                        outputs[key].extend(pred_com_denorm)
                        targets[key].extend(target_com_denorm)
                        targets['middle_frame_joints'].extend(denorm_joints)
                    else:
                        outputs[key].extend(output[key].cpu().numpy())
                        targets[key].extend(batch[key].cpu().numpy())
                
        # For console logging and early stopping, calculate mean losses
        mean_losses = {
            loss_name: np.mean(loss_data['values'])
            for loss_name, loss_data in batch_losses.items()
        }
       
        for key in targets.keys():
            outputs[key] = np.array(outputs[key])
            targets[key] = np.array(targets[key])
            
        # Log metrics with means
        self.manager.log_metrics({
            **{f'Loss/Subject_{subject}/val_{name}': value 
            for name, value in mean_losses.items()}
        }, step=global_step)

        if self.cfg.viz.validation.enabled:
            viz_path = self.manager.experiment_path / 'visualizations' / f'Subject{subject}' / 'validation'
            # Convert each list of outputs and targets to numpy arrays
            outputs = {key: np.array(value) for key, value in outputs.items()}
            self.create_visualizations(outputs, targets, viz_path, epoch, is_test=False)
        
        return batch_losses 

    @torch.no_grad()
    def test(self, test_loader: DataLoader, subject, global_step= 0):
        """Run testing."""
        if self.current_subject != subject:
            self.current_subject = subject
            
        self.model.eval()
        test_loss = 0
        network_dtype = next(self.model.parameters()).dtype
        
        outputs = defaultdict(list)
        targets = defaultdict(list)
        normalizer = DataNormalizer(self.cfg.default.data_path, cfg=self.cfg)
        
        for batch in test_loader:
            batch = {k: v.to(self.device).to(network_dtype) for k, v in batch.items()}
            batch['joint'] = batch['joint'].squeeze(2)
            if 'pressure' in batch:
                batch['pressure'] = batch['pressure'].view(-1, np.prod(batch['pressure'].size()[1:]))
            
            output = self.model(batch['joint'])
            loss = self.loss_fn(output, batch)
            test_loss += loss['total'].item()
            
            # Collect outputs and targets for each modality in the model's output
            for key in output:
                if key == 'com':
                    # Denormalize CoM predictions and targets
                    pred_com = output[key].cpu().numpy()
                    target_com = batch[key].cpu().numpy()
                    pred_com_denorm = normalizer.denormalize_com(pred_com, subject)
                    target_com_denorm = normalizer.denormalize_com(target_com, subject)
                    joints = batch['middle_frame_joints'].cpu().numpy()
                    denorm_joints = normalizer.denormalize_joints(joints, subject)
                    targets['middle_frame_joints'].extend(denorm_joints)
                    outputs[key].extend(pred_com_denorm)
                    targets[key].extend(target_com_denorm)
                else:
                    outputs[key].extend(output[key].cpu().numpy())
                    targets[key].extend(batch[key].cpu().numpy())

        avg_test_loss = test_loss / len(test_loader)
        # Log metrics
        self.manager.log_metrics({
            f'Loss/Subject_{subject}/test': avg_test_loss,
            **{f'Loss/Subject_{subject}/test_{name}': value.item() 
            for name, value in loss.items()}
        }, step=global_step)
       
        for key in targets.keys():
            outputs[key] = np.array(outputs[key])
            targets[key] = np.array(targets[key]) 
            
        # Create test visualizations if enabled
        save_dir = self.manager.experiment_path / 'visualizations' / f'Subject{subject}' / 'test'
        self.create_visualizations(outputs, targets, save_dir, epoch=None, is_test=True)
        
        return avg_test_loss

    def create_visualizations(self, outputs, targets, viz_path, epoch, is_test=False):
            """Create visualizations for all enabled modalities."""
            viz_path.mkdir(parents=True, exist_ok=True)
            num_samples = self.cfg.viz.collage.samples_per_viz
            keys = list(outputs.keys())
            step_size=5
            start_idx = np.random.randint(0, len(outputs[keys[0]]) - (num_samples*step_size)) 
            epoch_str = 'test' if is_test else f'epoch_{epoch}'
            
            # Clean up outputs and targets to ensure proper data structure
            predictions = {}
            ground_truth = {}
            
            # Handle CoM data if present
            if 'com' in outputs:
                predictions['com'] = outputs['com']
                ground_truth['com'] = targets['com']
                ground_truth['middle_frame_joints'] = targets['middle_frame_joints']
            
            # Handle pressure data if present
            if 'pressure' in outputs:
                predictions['pressure'] = outputs['pressure']
                ground_truth['pressure'] = targets['pressure']
                
            # Handle contact data if present
            if 'contact' in outputs:
                predictions['contact'] = outputs['contact']
                ground_truth['contact'] = targets['contact']
            
            # Create visualization grids
            self.manager.visualizer.create_validation_grid(
                predictions=predictions,
                targets=ground_truth,
                save_path=viz_path / f'visualization_{epoch_str}.png',
                start_idx=start_idx,
                num_samples=num_samples
            )
            
    def load_checkpoint(self, checkpoint_path: Path, model_only=False) -> int:
        """Loads a checkpoint and returns the epoch number."""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        epoch = checkpoint.get('epoch', 0)
        self.logger.info(f"Loaded checkpoint '{checkpoint_path}' (epoch {epoch})")
        if model_only:
            return epoch
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['lr_scheduler'])
        return epoch

    def load_best_model(self, flag: str = "best") -> int:
        """Loads the best model checkpoint for the current subject."""
        checkpoint_path = self.manager.get_checkpoint_dir(self.current_subject) / f'model_{flag}.pth'
        if not checkpoint_path.exists():
            self.logger.warning(f"No {flag} checkpoint found for subject {self.current_subject}")
            return 0
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        return checkpoint.get('epoch', 0)
