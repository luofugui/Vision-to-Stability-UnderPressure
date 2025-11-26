from dataclasses import dataclass
from typing import Dict, Any
from pathlib import Path
import yaml
import json
import logging
import torch
import random
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import re

from pressure.util.util import dict_to_simplenamespace, simplenamespace_to_dict, SimpleNamespace, MemoryMonitor
from pressure.util.visualizer import VisualizationManager


def seed_everything(seed=42, cuda_deterministic=True, use_cuda=True):
    """
    Seeds random number generators for reproducible results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
        if cuda_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
class ExperimentLogger:
    """Handles logging to both file and console with consistent formatting."""
    def __init__(self, log_path, level=logging.INFO):
        self.logger = logging.getLogger()
        self.logger.setLevel(level)
        
        # Create log directory if it doesn't exist
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler(str(log_path))
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        
        # Remove any existing handlers
        self.logger.handlers = []
        
        # Add both handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, msg):
        self.logger.info(msg)
    
    def warning(self, msg):
        self.logger.warning(msg)
    
    def error(self, msg):
        self.logger.error(msg)

    def exception(self, msg):
        self.logger.exception(msg)

@dataclass
class ExperimentConfig:
    """Handles experiment configuration and naming conventions."""
    base_path: Path
    config: Dict[str, Any]
    
    def __post_init__(self):
        self.base_path = Path(self.base_path)
        self.model_type = self.config['network']['model']

    def _format_reduction_params(self):
        """Creates a compact string of reduction parameters based on mode and specific loss settings."""
        mode = self.config['default']['mode']
        reductions = []
        
        if 'pressure' in mode:
            press_loss = self.config['loss']['pressure_loss']
            reductions.append(f"{press_loss[:3]}")  # e.g., "mse" for mean_squared_error

        # Add loss type (e.g., weighted or contact_conditioned)
        if 'contact' in mode:
            loss_type = self.config['loss'].get('loss_type', 'default')
            reductions.append(f"{loss_type[:3]}")  # e.g., "lt_con" for contact_conditioned
        
        # If in contact mode and loss_type is contact_conditioned, add penalty type
        if 'contact' in mode and loss_type == 'contact_conditioned':
            penalty_type = self.config['loss'].get('penalty_type', 'none')
            reductions.append(f"{penalty_type}")  # e.g., "pt_sca" for scaled
            
        return '_'.join(reductions) if reductions else ''
     
    @property
    def common_params(self) -> Dict[str, str]:
        """Returns common parameters across all model types."""
        return {
            'data_type': self.config['data']['data_type'],
            'om': 'OM' if self.config['default']['om'] else '',
            'model': self.config['network']['model'],
            'mode': self.config['default']['mode'],
            'reduction': self._format_reduction_params(),
            'sequence': f"seq_{self.config['data']['sequence_length']}",
            'learning_rate': f"lr_{self.config['training']['lr']}",
            'batch_size': f"bs_{self.config['training']['batch_size']}",
            'optimizer': f"_{self.config['training']['optimizer']}",
            'scheduler': f"_{self.config['training']['scheduler']}"
        }
    
    @property
    def model_specific_params(self) -> Dict[str, str]:
        """Returns model-specific parameters based on model type."""
        if self.model_type == 'pns':
            return {
                'hidden': f"hidden_{self.config['network']['hidden_count']}",
                'fc': f"fc_{self.config['network']['FC_size']}",
                'dropout': f"dropout_{self.config['network']['dropout']}",
                'mask_mult': f"maskmult_{self.config['network']['mask_mult']}"
            }
        elif self.model_type == 'footformer':
            return {
                'pose_dim': f"pose_dim{self.config['network']['pose_embed_dim']}",
                'heads': f"heads_{self.config['network']['num_heads']}",
                'layers': f"layers_{self.config['network']['num_layers']}",
                'dropout': f"dropout_{self.config['network']['dropout']}",
                'pos': f"pos_{self.config['network']['pos']}",
                'decoder_dim': f"decoder_dim_{self.config['network']['decoder_dim']}",
                'mlp_dim': f"mlpd_{self.config['network']['mlp_dim']}",
                'embedder': f"embedder_{self.config['network']['pose_embedder']}",
                'transformer': f"{self.config['network']['transformer']}",
            }
        else:
            raise NotImplementedError(f"Model type '{self.model_type}' not supported.")

    def get_experiment_id(self, include_timestamp=False):
        """Generates a unique experiment identifier."""
        all_params = {**self.common_params, **self.model_specific_params}
        
        path_components = [
            all_params['data_type'],
            all_params['om'],
            all_params['model'],
            '_'.join(all_params['mode']),
            all_params['reduction']
        ]
        
        param_components = [
            all_params['sequence'],
            all_params['learning_rate'],
            all_params['batch_size'],
            all_params['optimizer'],
            all_params['scheduler']
        ]
        
        param_components.extend(self.model_specific_params.values())
        path = '/'.join(filter(None, path_components))
        params = '_'.join(filter(None, param_components))
        
        if include_timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            params = f"{params}_{timestamp}"
            
        return str(Path(path) / params)

class ExperimentManager:
    """Manages experiment directories, configurations, logging, and TensorBoard."""
    def __init__(self, config, base_path=None, include_timestamp=False):
        # Convert config dict to SimpleNamespace if it isn't already
        self.config = config if isinstance(config, SimpleNamespace) else dict_to_simplenamespace(config)
        self.base_path = Path(base_path or self.config.default.results_path)
        self.exp_config = ExperimentConfig(self.base_path, simplenamespace_to_dict(self.config))
        self.experiment_path = None
        self.logger = None
        self.writer = None
        self.visualizer = None
        self.memory_monitor = MemoryMonitor()
        self.setup_experiment(include_timestamp=include_timestamp)
        
    def setup_experiment(self, include_timestamp=False):
        """Sets up experiment directory, logging, and TensorBoard."""
        # Generate experiment path
        experiment_id = self.exp_config.get_experiment_id(include_timestamp)
        self.experiment_path = self.base_path / experiment_id
        self.experiment_path.mkdir(parents=True, exist_ok=True)
       
        # Set up visualization
        viz_path = self.experiment_path / 'visualizations'
        self.visualizer = VisualizationManager(save_dir=viz_path, cfg=self.config)
        
        # Set up TensorBoard
        tensorboard_path = self.experiment_path / 'Tensorboard'
        self.writer = SummaryWriter(log_dir=str(tensorboard_path))
        
        # Save initial config to TensorBoard
        self.writer.add_text('Experiment/Configuration', 
                           json.dumps(simplenamespace_to_dict(self.config), indent=4), 
                           global_step=0)
        
        log_path = self.experiment_path / 'log' / 'training.log'
        self.logger = ExperimentLogger(log_path)
        seed_everything(self.config.default.seed)
        self.save_config()
        
        return self.experiment_path
   
    def save_config(self) -> None:
        """Saves experiment configuration to yaml file."""
        config_path = self.experiment_path / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(simplenamespace_to_dict(self.config), f, default_flow_style=False)
    
    def get_checkpoint_dir(self, subject=None) -> Path:
        """Gets checkpoint directory for a subject."""
        checkpoint_dir = self.experiment_path / 'checkpoints'
        if subject is not None:
            checkpoint_dir = checkpoint_dir / f'Subject{subject}'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir
    
    def log_metrics(self, metrics, step, subject=None):
        """Logs metrics to both TensorBoard and log file."""
        prefix = f"Subject_{subject}/" if subject else ""
        for name, value in metrics.items():
            self.writer.add_scalar(f'{prefix}{name}', value, global_step=step)
            self.logger.info(f"{prefix}{name}: {value}")
            
        mem_cpu = self.memory_monitor.get_memory_mb()
        mem_gpu = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        self.writer.add_scalar(f'{prefix}Memory/CPU_MB', mem_cpu, global_step=step)
        self.writer.add_scalar(f'{prefix}Memory/GPU_MB', mem_gpu, global_step=step)
        self.gpu_peak = max(self.memory_monitor.peak_gpu_memory, mem_gpu)
    
    def close(self) -> None:
        """Closes TensorBoard writer."""
        if self.writer:
            self.writer.close()
            
    def find_most_recent_checkpoint(self):
        """Finds the most recent checkpoint across all subjects."""
        checkpoint_base = self.experiment_path / 'checkpoints'
        if not checkpoint_base.exists():
            return None, None
           
        subject_dirs = [d for d in checkpoint_base.iterdir() if d.is_dir()]
        latest_time = None
        latest_checkpoint = None
        latest_subject = None
        
        for subject_dir in subject_dirs:
            checkpoint_path = subject_dir / 'model_latest.pth'
            if checkpoint_path.is_file():
                modification_time = checkpoint_path.stat().st_mtime
                if latest_time is None or modification_time > latest_time:
                    latest_time = modification_time
                    latest_checkpoint = checkpoint_path
                    # Extract subject number from directory name (e.g., "Subject1" -> 1)
                    latest_subject = int(re.match(r'Subject(\d+)', subject_dir.name).group(1))
                    
        if latest_checkpoint is not None:
            self.logger.info(f"Found latest checkpoint '{latest_checkpoint}' for subject {latest_subject}")
            
        return latest_subject, latest_checkpoint

    def save_checkpoint(self, epoch, model, optimizer, scheduler, flag="lastest", subject=None):
        """Saves a checkpoint with specified flag."""
        checkpoint_dir = self.get_checkpoint_dir(subject)
        checkpoint_path = checkpoint_dir / f'model_{flag}.pth'
        
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': scheduler.state_dict()
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved {flag} checkpoint for subject {subject} at epoch {epoch}")
     
    def create_validation_visualization(self, outputs,
                                     targets,
                                     subject,
                                     epoch):
        """Create validation visualizations using appropriate visualizer."""
        if self.viz_manager and self.config.viz.validation.enabled:
            viz_path = self.experiment_path / 'visualizations' / f'Subject{subject}' / 'validation'
            self.viz_manager.create_collage(outputs=outputs, targets=targets, save_dir=viz_path, epoch=epoch)
            
def create_experiment(cfg, include_timestamp = False):
    """Helper function to create a new experiment."""
    manager = ExperimentManager(cfg, include_timestamp=include_timestamp)
    return manager