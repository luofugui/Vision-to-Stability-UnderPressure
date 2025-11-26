import torch
import torch.nn as nn
import torch.nn.functional as F

class PressureLoss(nn.Module):
    def __init__(self, cfg):
        super(PressureLoss, self).__init__()
        self.cfg = cfg
        self.mode = cfg.default.mode
        self.loss_fns = {}
        self.lambdas = {}
        self.gt_com = cfg.data.gt_com
        
        # Setup pressure loss if needed
        if 'pressure' in self.mode:
            if cfg.loss.pressure_loss == 'kld':
                if cfg.loss.pressure_reduction == 'mean':
                    cfg.loss.pressure_reduction = 'batchmean'
                    self.cfg.loss.pressure_reduction = 'batchmean' 
                self.loss_fns['pressure'] = nn.KLDivLoss(reduction=cfg.loss.pressure_reduction, log_target=False)
                self.eps = 1e-8
            elif cfg.loss.pressure_loss == 'mse':
                self.loss_fns['pressure'] = nn.MSELoss(reduction=cfg.loss.pressure_reduction)
            self.lambdas['pressure'] = cfg.loss.lambda_pressure
        
        if 'contact' in self.mode:
            self.loss_fns['contact'] = nn.BCEWithLogitsLoss(reduction=cfg.loss.contact_reduction)
            self.lambdas['contact'] = cfg.loss.lambda_contact
           
        if 'com' in self.mode:
            self.loss_fns['com'] = nn.MSELoss(reduction=cfg.loss.com_reduction)
            self.lambdas['com'] = cfg.loss.lambda_com
           
        self.cross_task = None 
            
    def compute_loss(self, modality, pred, target):
        """Compute loss for a specific modality"""
        if modality not in self.loss_fns:
            raise ValueError(f"No loss function configured for modality: {modality}")
        
        if modality == 'pressure':
            if self.cfg.loss.pressure_loss == 'kld':
                pred = torch.clamp(pred, min=self.eps, max=1.0)
                target = torch.clamp(target, min=self.eps, max=1.0)
                #KLD with batchmean already handles batch normalization
                if self.cfg.loss.pressure_reduction == 'sum':
                    num_elements = pred.size(1)  # Number of pressure cells
                    raw_loss = self.loss_fns[modality](torch.log(pred), target)
                    return raw_loss / num_elements
                #For batchmean, return as is
                return self.loss_fns[modality](torch.log(pred), target)
            return self.loss_fns[modality](pred, target)
            
        elif modality == 'com' and 'com' in self.mode:
            if target.shape != pred.shape:
                target = target.squeeze(1)

            if self.gt_com:  # Apply masking when using Vicon-derived ground truth
                mask = target[:, 3] > 0  # Confidence is in the 4th column
                pred = pred[mask, :3]  # Select only x, y, z
                target = target[mask, :3]  # Select only x, y, z
                
                if mask.sum() == 0:  # Avoid NaN loss if all values are masked out
                    return torch.tensor(0.0, device=pred.device, requires_grad=True)
                
            return self.loss_fns[modality](pred, target)
            
        return self.loss_fns[modality](pred, target)

    def forward(self, output, target):
        losses = {}
        total_loss = 0

        for modality in self.mode:
            if modality in output and modality in target:
                loss = self.compute_loss(modality, output[modality], target[modality])
                losses[modality] = loss
                total_loss += self.lambdas[modality] * loss
                losses[modality] *= self.lambdas[modality]

        losses['total'] = total_loss
        return losses
