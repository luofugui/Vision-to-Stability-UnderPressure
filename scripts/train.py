import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from pressure.util.util import *
from pressure.util.stats import collect_metrics
from pressure.util.manager import *
from pressure.util.pipeline import Pipeline
from pressure.util.selection import *
from scripts.eval import eval_model

def run_training_epoch(pipeline, manager, cfg, writer, logger,
                       data_id, train_dataset, test_dataset,
                       resume_checkpoint=None, epoch_start=0):
    """
    Runs training + testing + evaluation for a single subject or full dataset.
    """
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.training.batch_size,
                              shuffle=(not cfg.default.loso),  # shuffle if not LOSO
                              num_workers=cfg.training.dataloader_workers)

    test_loader = None
    if test_dataset is not None and len(test_dataset) > 0:
        test_loader = DataLoader(test_dataset,
                                 batch_size=cfg.training.batch_size,
                                 shuffle=False,
                                 num_workers=cfg.training.dataloader_workers)

    dataset_size = len(train_dataset)
    pipeline.setup_training(data_id, dataset_size=dataset_size)

    # Resume or start fresh
    if resume_checkpoint is not None:
        epoch = pipeline.load_checkpoint(resume_checkpoint)
        logger.info(f"Resuming {data_id} from epoch {epoch}")
    else:
        epoch = epoch_start
        logger.info(f"Starting training from scratch for {data_id}")

    # --- Training ---
    if cfg.default.train:
        _, global_step = pipeline.train(train_loader, test_loader, data_id, epoch_completed=epoch)
    else:
        global_step = 0

    # --- Evaluation ---
    num_epochs = pipeline.load_best_model(cfg.eval.checkpoint_flag)
    logger.info(f'Loaded best checkpoint from epoch {num_epochs} for {data_id}')

    metrics = {}
    if cfg.default.eval and test_loader is not None:
        eval_dir = manager.experiment_path / 'eval'
        eval_dir.mkdir(exist_ok=True)
        metrics = eval_model(
            pipeline.model, test_loader, eval_dir,
            data_id=data_id,
            experiment_manager=manager,
            cfg=cfg, writer=writer, logger=logger,
            global_step=global_step
        )
        for metric_type, values in metrics.get('pressure', {}).items():
            writer.add_scalar(f'Metrics/{data_id}/{metric_type}', values[0], global_step=global_step)

    return num_epochs, metrics

def main(cfg, args=None):
    manager = create_experiment(cfg)
    cfg, logger, writer = manager.config, manager.logger, manager.writer

    logger.info(f"Model: {cfg.network}")
    logger.info(f"Data: {cfg.data}")
    logger.info(f"Training: {cfg.training}")

    # Subject setup
    if cfg.default.subjects == 'all':
        subjects = np.arange(1, 11)
    else:
        subjects = np.array(cfg.default.subjects)
    subjects.sort()

    foot_mask = np.load('assets/foot_mask_nans.npy')
    foot_mask[np.isnan(foot_mask)] = 0
    foot_mask = torch.tensor(foot_mask).float().to(cfg.default.device)

    pipeline = Pipeline(manager)

    # --- Resume checkpoint handling ---
    latest_checkpoint = None
    latest_subject = None
    if args is not None and args.resume:
        latest_subject, latest_checkpoint = manager.find_most_recent_checkpoint()
        if latest_subject:
            logger.info(f"Resuming training from subject {latest_subject}")
        else:
            logger.info("No checkpoint found, starting fresh.")

    epochs, all_metrics = [], []

    if cfg.default.loso:
        for subject in subjects:
            if args is not None and args.resume and latest_subject is not None:
                if subject < latest_subject:
                    continue  # skip completed
            logger.info(f"=== LOSO Training for Subject {subject} ===")

            train_dataset, _, test_dataset = create_dataset(cfg, subject)
            resume_ckpt = latest_checkpoint if subject == latest_subject else None
            epoch, metrics = run_training_epoch(
                pipeline, manager, cfg, writer, logger,
                data_id=subject,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                resume_checkpoint=resume_ckpt
            )
            epochs.append(epoch)
            if metrics:
                all_metrics.append(metrics)

    # Just use the entire dataset for training
    else:
        logger.info("=== Full-dataset training (no LOSO) ===")
        train_dataset, _, test_dataset = create_dataset(cfg, subject=None)
        epoch, metrics = run_training_epoch(
            pipeline, manager, cfg, writer, logger,
            data_id='all_subjects',
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            resume_checkpoint=latest_checkpoint
        )
        epochs.append(epoch)
        if metrics:
            all_metrics.append(metrics)

    manager.close()
    logger.info("Training completed.")
    logger.info(f"Epoch summary: {epochs}")

    if all_metrics:
        mean_metrics = collect_metrics(all_metrics)
        eval_dir = manager.experiment_path / 'eval'
        metrics_path = eval_dir / 'mean_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(mean_metrics, f, cls=NumpyEncoder, indent=4)
        logger.info(f"Saved mean metrics to {metrics_path}")
        return mean_metrics

    return {}      

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Train a model for pressure prediction')
    argparser.add_argument('--config', type=str, default='configs/pns.yaml', help='Path to the config file')
    argparser.add_argument('--resume', action='store_true', help='Resume training from the latest checkpoint') 
    args = argparser.parse_args() 
   
    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file '{args.config}' not found.")
    
    main(cfg, args) 