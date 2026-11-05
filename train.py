import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from mmcv.parallel import MMDistributedDataParallel as MMDDP
from timm.data import (create_dataset, create_loader,
                       resolve_data_config, Mixup,
                       FastCollateMixup, AugMixDataset)
from torchvision import datasets
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from timm.models import (create_model, safe_model_name, resume_checkpoint,
                         load_checkpoint, convert_splitbn_model, model_parameters)
from timm.utils import *
from timm.loss import *
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

import models



class PSALoss(nn.Module):
    """
    Progressive Soft Asymmetric Loss (PSALoss)
    """
    def __init__(self, 
                 gamma_neg=2.0, 
                 gamma_pos=0.5, 
                 clip=0.03,
                 eps=1e-8,
                 warmup_epochs=10,
                 total_epochs=150,
                 smoothing=0.1, 
                 reduction='mean',
                 adaptive_weight=True,
                 progressive_enabled=True,
                 use_psa_loss=True):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.smoothing = smoothing
        self.reduction = reduction
        self.adaptive_weight = adaptive_weight
        self.progressive_enabled = progressive_enabled
        self.use_psa_loss = use_psa_loss
        
        self.current_epoch = 0
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch
    
    def progressive_adjust(self, epoch=None, total_epochs=None):
        if epoch is not None:
            self.current_epoch = epoch
        if total_epochs is not None:
            self.total_epochs = total_epochs
        return self.get_progressive_weight()
    
    def get_progressive_weight(self):
        if self.current_epoch < self.warmup_epochs:
            return 0.0
        elif self.current_epoch >= self.total_epochs * 0.8:
            return 1.0
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs * 0.8 - self.warmup_epochs)
            sigmoid_input = progress * 5 - 2.5
            return torch.sigmoid(torch.tensor(sigmoid_input, dtype=torch.float32, device=next(self.parameters()).device if list(self.parameters()) else torch.device('cpu'))).item()
    
    def forward(self, pred, target, lam_a=None, lam_b=None, epoch=None, total_epochs=None):
        if epoch is not None:
            self.current_epoch = epoch
        if total_epochs is not None:
            self.total_epochs = total_epochs
            
        num_classes = pred.size(-1)
        batch_size = pred.size(0)
        
        if self.progressive_enabled and self.use_psa_loss:
            w_as = self.get_progressive_weight()
        else:
            w_as = 0.0
        
        log_probs = F.log_softmax(pred, dim=-1)
        
        if self.smoothing > 0:
            target_smooth = target * (1 - self.smoothing) + self.smoothing / num_classes
        else:
            target_smooth = target
        
        loss_ce = -(target_smooth * log_probs).sum(dim=-1)  # [B]
        
        # Asymmetric Loss 
        if w_as > 0 and self.use_psa_loss:
            probs = pred.softmax(dim=-1)
            if self.clip is not None and self.clip > 0:
                probs_clipped = probs.clamp(min=self.clip, max=1 - self.clip)
            else:
                probs_clipped = probs
            
            pos_term = -((1 - probs_clipped) ** self.gamma_pos) * torch.log(probs_clipped + self.eps)
            neg_term = -(probs_clipped ** self.gamma_neg) * torch.log(1 - probs_clipped + self.eps)
            loss_as = torch.where(target_smooth > 0.5, pos_term, neg_term).sum(dim=-1)  # [B]
        else:
            loss_as = torch.zeros_like(loss_ce)
        
        loss_original = loss_ce * (1 - w_as) + loss_as * w_as

        if lam_b is not None and lam_a is not None:
            lam_a_v = lam_a.view(-1).float()
            lam_b_v = lam_b.view(-1).float()

            mixed = (lam_b_v > 0).to(loss_ce.dtype)  # [B]
            
            if self.adaptive_weight:
                mix_uncertainty = torch.sin(lam_b_v * np.pi)
                boundary_factor = (1 - torch.abs(lam_a_v - 0.5) * 2).clamp(min=0, max=1)
                adaptive_weight = 1.0 - 0.15 * mix_uncertainty * boundary_factor
            else:
                adaptive_weight = torch.ones_like(lam_b_v)
            
            loss_mixed = loss_ce * adaptive_weight
            
            loss = mixed * loss_mixed + (1 - mixed) * loss_original
        else:
            loss = loss_original
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class ActivationMapExtractor:
    """
    Activation-Guided CutMix、class-agnostic activation map
    """
    def __init__(self, model, debug=False):
        self.model = model
        self.cached_activation_map = None
        self.debug = debug
        self.call_count = 0
        self.failed_count = 0
        
        self.has_forward_embeddings = hasattr(model, 'forward_embeddings')
        self.has_forward_tokens = hasattr(model, 'forward_tokens')
        self.has_fork_feat = hasattr(model, 'fork_feat')
        
        if debug:
            print(f"[ActivationMapExtractor] Model structure check:")
            print(f"  - has forward_embeddings: {self.has_forward_embeddings}")
            print(f"  - has forward_tokens: {self.has_forward_tokens}")
            print(f"  - has fork_feat: {self.has_fork_feat}")
        
    def compute_activation_map(self, input_tensor, target_class):

        self.call_count += 1
        

        was_training = self.model.training
        
        try:
            self.model.eval()
            
            with torch.no_grad():

                if self.has_forward_embeddings and self.has_forward_tokens:
                    x = self.model.forward_embeddings(input_tensor)
                    multi_scale_features = self.model.forward_tokens(x)
                else:
                    multi_scale_features = self.model.forward(input_tensor)
                
                if isinstance(multi_scale_features, list):
                    last_feature = multi_scale_features[-1]  # [B, C, H', W']
                else:
                    last_feature = multi_scale_features
                
                orig_h, orig_w = input_tensor.shape[2], input_tensor.shape[3]
                feat_h, feat_w = last_feature.shape[2], last_feature.shape[3]
                
                act_map = last_feature.mean(dim=1, keepdim=False)  # [B, H', W']
                
                if feat_h != orig_h or feat_w != orig_w:
                    act_map = torch.nn.functional.interpolate(
                        act_map.unsqueeze(1),  # [B, 1, H', W']
                        size=(orig_h, orig_w),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(1)  # [B, H, W]
                
                act_maps = []
                for i in range(act_map.size(0)):
                    act_map_i = act_map[i]
                    if act_map_i.max() > 0:
                        act_map_i = act_map_i / act_map_i.max()
                    act_maps.append(torch.relu(act_map_i))
                
                result = torch.stack(act_maps, dim=0)  # [B, H, W]
                
                if self.debug and self.call_count <= 3:
                    print(f"[ActivationMapExtractor Debug #{self.call_count}]")
                    print(f"  Input shape: {input_tensor.shape}")
                    print(f"  Feature shape: {last_feature.shape}")
                    print(f"  Activation map shape: {result.shape}")
                    print(f"  Activation map stats: min={result.min().item():.4f}, max={result.max().item():.4f}, mean={result.mean().item():.4f}")
                    print(f"  Activation map variance (spatial importance): {result.var().item():.6f}")
                
                return result
                
        except Exception as e:
            self.failed_count += 1
            if self.debug:
                import traceback
                print(f"[ActivationMapExtractor Error #{self.failed_count}] {e}")
                traceback.print_exc()
            return None
        finally:
            if was_training:
                self.model.train()
            else:
                self.model.eval()
    
    def get_stats(self):
        """获取激活图计算统计"""
        return {
            'total_calls': self.call_count,
            'failed_calls': self.failed_count,
            'success_rate': (self.call_count - self.failed_count) / max(self.call_count, 1)
        }

class ActivationGuidedCutMix:
    """Activation-Guided CutMix: activation-map-guided mixing for medical imaging"""

    def __init__(self, model, beta=1.0, prob=1.0, num_classes=4, use_cls_weights=False,
                 activation_enabled=True, activation_layer='last', debug=False, save_visualizations=None):
        self.model = model
        self.beta = beta
        self.prob = prob
        self.num_classes = num_classes
        self.use_cls_weights = use_cls_weights
        self.activation_enabled = activation_enabled
        self.activation_layer = activation_layer
        self.debug = debug
        self.save_visualizations = save_visualizations  # e.g., './act_cutmix_debug'
        self.model.eval()
        self.activation_map_extractor = None
        self.visual_count = 0
        

        if self.activation_enabled:
            try:
                self.activation_map_extractor = ActivationMapExtractor(model, debug=debug)
                _logger.info("Activation-Guided CutMix with activation-map guidance enabled")
            except Exception as e:
                _logger.warning(f"Failed to initialize activation map extractor, using fallback: {e}")
                self.activation_map_extractor = None

    def get_activation_map(self, input, target):
        """
        Generate the class-agnostic activation map for Activation-Guided CutMix
        Uses the channel-averaged last-stage feature activation to identify semantically important regions
        """
        if self.activation_map_extractor is None:
            return None, None
        
        act_map = self.activation_map_extractor.compute_activation_map(input, target)
        return act_map, None

    def _extract_features(self, x):
        """Extract intermediate features for activation map computation - deprecated, use ActivationMapExtractor"""
        return None

    def _compute_activation_map(self, features, target):
        """Compute activation map - deprecated, use ActivationMapExtractor"""
        return None

    def rand_bbox(self, size, lam):
        """
        Generate a random bounding box for mixing (per-sample).
        Args:
            size: [B, C, H, W]
            lam: scalar or array of per-sample keep-ratio (original area fraction)
        Returns:
            list of (x1, y1, x2, y2) per sample
        """
        W = size[-1]
        H = size[-2]
        bs = size[0]
        lam_arr = np.asarray(lam, dtype=np.float64).reshape(-1)
        if lam_arr.size == 1:
            lam_arr = np.repeat(lam_arr, bs)

        boxes = []
        for b in range(bs):
            cut_rat = np.sqrt(1. - float(lam_arr[b]))
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)

            # Random center point
            cx = np.random.randint(W)
            cy = np.random.randint(H)

            x1 = int(np.clip(cx - cut_w // 2, 0, W))
            y1 = int(np.clip(cy - cut_h // 2, 0, H))
            x2 = int(np.clip(cx + cut_w // 2, 0, W))
            y2 = int(np.clip(cy + cut_h // 2, 0, H))
            boxes.append((x1, y1, x2, y2))
        return boxes

    def __call__(self, input, target):
        """
        Apply Activation-Guided CutMix with activation-map-guided mixing

        Args:
            input: Input images [B, C, H, W]
            target: Target labels [B]

        Returns:
            mixed_input: Augmented images
            target_a: Original labels
            target_b: Mixed labels (from random batch)
            lam_a: Mixing coefficient for original (per-sample)
            lam_b: Mixing coefficient for mixed (per-sample)
        """
        bs = input.size(0)
        device = input.device

        # Random permutation for batch mixing
        rand_index = torch.randperm(bs).to(device)
        target_b = target[rand_index]

        apply_mask = torch.rand(bs, device=device) < self.prob

        lam_keep = np.random.beta(self.beta, self.beta, size=bs)

        act_map = None
        if self.activation_map_extractor is not None:
            try:
                act_map = self.activation_map_extractor.compute_activation_map(input, target)
            except Exception as e:
                print(f"Activation map computation failed: {e}")
                act_map = None


        if act_map is not None:
            boxes = self._activation_guided_bbox(input.size(), act_map, lam_keep)
        else:
            boxes = self.rand_bbox(input.size(), lam_keep)

        # Apply bounding box replacement per sample
        mixed_input = input.clone()
        H, W = input.size(-2), input.size(-1)
        total_area = H * W

        lam_a_list = []
        lam_b_list = []

        for b in range(bs):
            x1, y1, x2, y2 = boxes[b]
            if bool(apply_mask[b]) and x2 > x1 and y2 > y1:
                ncont = input[rand_index[b], :, y1:y2, x1:x2].clone()
                mixed_input[b, :, y1:y2, x1:x2] = ncont
                area = (y2 - y1) * (x2 - x1)
            else:
                area = 0

            lam_b_val = area / total_area
            lam_a_list.append(1.0 - lam_b_val)
            lam_b_list.append(lam_b_val)

        lam_a = torch.tensor(lam_a_list, dtype=torch.float32, device=device)
        lam_b = torch.tensor(lam_b_list, dtype=torch.float32, device=device)

        # Handle NaN values
        lam_a = torch.where(torch.isnan(lam_a), torch.ones_like(lam_a), lam_a)
        lam_b = torch.where(torch.isnan(lam_b), torch.zeros_like(lam_b), lam_b)

        lam_sum = lam_a + lam_b
        lam_a = lam_a / lam_sum
        lam_b = lam_b / lam_sum

        if self.save_visualizations and self.visual_count < 20:
            self._save_visualization(input, mixed_input, act_map, target, target_b,
                                    lam_a, lam_b, *boxes[0])
            self.visual_count += 1

        return mixed_input, target, target_b, lam_a, lam_b
    
    def _save_visualization(self, input, mixed_input, act_map, target, target_b, 
                          lam_a, lam_b, x1, y1, x2, y2):
        """Save the activation-guided CutMix visualization, which is used to verify whether the Activation map guidance works"""
        try:
            import os
            import matplotlib.pyplot as plt
            import numpy as np
            
            os.makedirs(self.save_visualizations, exist_ok=True)
            
            img_a = input[0].cpu()  # [C, H, W]
            img_b = input[torch.randperm(input.size(0))[0]].cpu()
            mixed = mixed_input[0].cpu()
            
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            
            def denorm(t):
                t = t * std + mean
                return torch.clamp(t, 0, 1)
            
            img_a_np = denorm(img_a).permute(1, 2, 0).numpy()
            mixed_np = denorm(mixed).permute(1, 2, 0).numpy()
            
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            axes[0].imshow(img_a_np)
            axes[0].set_title(f'Original (Class {target[0].item()})', fontsize=11)
            axes[0].axis('off')
            
            axes[1].imshow(mixed_np)
            axes[1].set_title(f'Mixed (λ_a={lam_a[0].item():.2f}, λ_b={lam_b[0].item():.2f})', fontsize=11)
            axes[1].axis('off')
            
            ax = axes[1]
            rect1 = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                   fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect1)
            
            if act_map is not None:
                act_map_np = act_map[0].cpu().numpy()
                im = axes[2].imshow(act_map_np, cmap='jet')
                axes[2].set_title('Activation Map (Feature Activation)', fontsize=11)
                axes[2].axis('off')
                plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
            else:
                axes[2].text(0.5, 0.5, 'Activation Map Failed\n(Fallback to Random)', 
                           ha='center', va='center', fontsize=11)
                axes[2].axis('off')
            
            plt.suptitle(f'Activation-Guided CutMix #{self.visual_count} - Activation-Guided Mixing', fontsize=12, fontweight='bold')
            plt.tight_layout()
            
            save_path = os.path.join(self.save_visualizations, f'act_cutmix_{self.visual_count:03d}.png')
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            if self.debug:
                print(f"[Activation-Guided CutMix Debug] Saved visualization to {save_path}")
                
        except Exception as e:
            if self.debug:
                print(f"[Activation-Guided CutMix Visualization Error] {e}")
    
    def _activation_guided_bbox(self, size, act_map, lam, offset=0):
         """
        Use activation map to guide bounding box generation - Select mixing regions near high activation regions (sample by sample)
        Args:
        size: Input size [B, C, H, W]
        act_map: Activation map [B, H, W]
        lam: beta distribution parameter (scalar or array per sample, keep-ratio)
        offset: A random offset
        Returns:
        list of (x1, y1, x2, y2), one box per sample
        """
        B, C, H, W = size
        bs = act_map.size(0)
        
        lam_arr = np.asarray(lam, dtype=np.float64).reshape(-1)
        if lam_arr.size == 1:
            lam_arr = np.repeat(lam_arr, bs)
        
        boxes = []
        
        for b in range(bs):
            cut_rat = np.sqrt(1. - float(lam_arr[b]))
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)
            
            act_map_b = act_map[b]  # [H, W]
            
            if act_map_b.max() > 0:
                threshold = act_map_b.max() * 0.5
                high_act_mask = act_map_b > threshold
                
                if high_act_mask.sum() > 0:
                    high_indices = torch.where(high_act_mask)
                    idx = np.random.randint(len(high_indices[0]))
                    cy_high = high_indices[0][idx].item()
                    cx_high = high_indices[1][idx].item()
                    
                    cx = int(np.clip(cx_high + np.random.randint(-offset, offset+1), 0, W))
                    cy = int(np.clip(cy_high + np.random.randint(-offset, offset+1), 0, H))
                else:
                    cx = np.random.randint(W)
                    cy = np.random.randint(H)
            else:
                cx = np.random.randint(W)
                cy = np.random.randint(H)
            
            x1 = int(np.clip(cx - cut_w // 2, 0, W))
            y1 = int(np.clip(cy - cut_h // 2, 0, H))
            x2 = int(np.clip(cx + cut_w // 2, 0, W))
            y2 = int(np.clip(cy + cut_h // 2, 0, H))
            
            boxes.append((x1, y1, x2, y2))
        
        return boxes


def act_guided_cutmix_data(input, target, model=None, beta=1.0, prob=1.0, num_classes=4):
    """
    Activation-Guided CutMix data augmentation function (standalone version)

    This is a simplified version that can work without model-specific activation map computation
    For best results, use ActivationGuidedCutMix class with proper activation map extraction
    """
    r = np.random.rand(1)
    if r > prob:
        # No Activation-Guided CutMix applied
        lam_a = torch.ones(input.size(0), device=input.device)
        lam_b = torch.zeros(input.size(0), device=input.device)
        return input, target, target.clone(), lam_a, lam_b

    bs = input.size(0)
    device = input.device

    # Beta distribution parameter
    lam = np.random.beta(beta, beta)

    # Random permutation for batch mixing
    rand_index = torch.randperm(bs).to(device)
    target_b = target[rand_index]

    # Generate random bounding box
    W = input.size(-1)
    H = input.size(-2)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # Apply mixing
    mixed_input = input.clone()
    if bbx2 > bbx1 and bby2 > bby1:
        ncont = input[rand_index, :, bbx1:bbx2, bby1:bby2].clone()
        # Resize to match target region
        if ncont.shape[2:] != (bbx2 - bbx1, bby2 - bby1):
            ncont = torch.nn.functional.interpolate(
                ncont, size=(bbx2 - bbx1, bby2 - bby1),
                mode='bilinear', align_corners=True
            )
        mixed_input[:, :, bbx1:bbx2, bby1:bby2] = ncont

    # Calculate mixing coefficients based on area
    area = (bby2 - bby1) * (bbx2 - bbx1)
    total_area = W * H
    lam_a = 1 - area / total_area
    lam_b = area / total_area

    # Handle same label case
    same_label = target == target_b
    if same_label.any():
        tmp = lam_a.clone()
        lam_a[same_label] += lam_b[same_label]
        lam_b[same_label] += tmp[same_label]

    return mixed_input, target, target_b, lam_a, lam_b


try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(
    description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset parameters 
parser.add_argument('--data-dir', default='/clas/data/class2', type=str, metavar='DIR',
                    help='path to dataset') 
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
parser.add_argument('--val-split', metavar='NAME', default='train',
                    help='dataset validation split (default: train, will auto-split from train)')
parser.add_argument('--val-split-ratio', type=float, default=0.2,
                    help='validation split ratio from train data (default: 0.2)')
parser.add_argument('--auto-val-split', action='store_true', default=True,
                    help='automatically split validation set from train data')
parser.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')

# Model parameters
parser.add_argument('--model', default='MedTransXNet_t', type=str, metavar='MODEL',
                    help='Name of model to train (default: "transxnet_progressive_asym_a4_t")') 
parser.add_argument('--pretrained', action='store_true', default=True,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=4, metavar='N', #类别
                    help='number of label classes (Model default if None)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                    help='validation batch size override (default: None)')
parser.add_argument('--grad-checkpoint', action='store_true', default=False,
                    help='Using gradient checkpointing for saving GPU memory')
parser.add_argument('--checkpoint-stage', default=[0, 0, 1, 1], type=int, nargs='+',
                    help='stage for using grad checkpoint')
parser.add_argument('--val-freq', default=1, type=int,
                    help='do evaluation per n epoch')
parser.add_argument('--val-start-epoch', default=0, type=int,
                    help='do evaluation per epoch after n-th epoch')
# Optimizer parameters
parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "adamw"')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.05,
                    help='weight decay (default: 0.05)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')


# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--auto-lr', action='store_true',
                    default=False, help='auto scaling learning rate')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                    help='amount to decay each learning rate cycle (default: 0.5)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit, cycles enabled if > 1')
parser.add_argument('--lr-k-decay', type=float, default=1.0,
                    help='learning rate k-decay for cosine/poly (default: 1.0)')
parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate (default: 1e-6)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=150, metavar='N',
                    help='number of epochs to train (default: 300)')
parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=100, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
parser.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default="rand-m9-mstd0.5-inc1", metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: rand-m9-mstd0.5-inc1)'),
parser.add_argument('--aug-repeats', type=int, default=0,
                    help='Number of augmentation repetitions (distributed training only) (default: 0)')
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--jsd-loss', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
parser.add_argument('--bce-loss', action='store_true', default=False,
                    help='Enable BCE loss w/ Mixup/CutMix use.')
parser.add_argument('--bce-target-thresh', type=float, default=None,
                    help='Threshold for binarizing softened BCE targets (default: None, disabled)')
parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                    help='Random erase prob (default: 0.25)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.8,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
parser.add_argument('--cutmix', type=float, default=1.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
# Activation-Guided CutMix parameters (activation-map-guided semantic mixing)
parser.add_argument('--act-cutmix', '--snapmix', dest='act_cutmix', action='store_true', default=True,
                    help='Enable Activation-Guided CutMix augmentation (activation-map guided mixing)')
parser.add_argument('--act-cutmix-beta', '--snapmix-beta', dest='act_cutmix_beta', type=float, default=1.0,
                    help='Beta parameter of the mixing-ratio distribution (default: 1.0)')
parser.add_argument('--act-cutmix-prob', '--snapmix-prob', dest='act_cutmix_prob', type=float, default=0.5,
                    help='Per-sample probability of applying Activation-Guided CutMix (default: 0.5)')
parser.add_argument('--act-cutmix-cls-weights', '--snapmix-cls-weights', dest='act_cutmix_cls_weights', action='store_true', default=False,
                    help='Use class weights for activation map computation')
parser.add_argument('--act-cutmix-debug', '--snapmix-debug', dest='act_cutmix_debug', action='store_true', default=False,
                    help='Enable Activation-Guided CutMix debug mode (print activation map stats)')
parser.add_argument('--act-cutmix-visualize', '--snapmix-visualize', dest='act_cutmix_visualize', type=str, default=None,
                    help='Save Activation-Guided CutMix visualization to this directory (e.g., ./act_cutmix_debug)')
# PSALoss parameters (Progressive Soft Asymmetric Loss)
parser.add_argument('--psa-loss', '--asym-loss', dest='psa_loss', action='store_true', default=True,
                    help='Enable PSALoss for Activation-Guided CutMix (default: True)')
parser.add_argument('--psa-gamma-neg', '--asym-gamma-neg', dest='psa_gamma_neg', type=float, default=2.0,
                    help='PSALoss negative-class focusing parameter gamma_neg (default: 2.0)')
parser.add_argument('--psa-gamma-pos', '--asym-gamma-pos', dest='psa_gamma_pos', type=float, default=0.5,
                    help='PSALoss positive-class focusing parameter gamma_pos (default: 0.5)')
parser.add_argument('--psa-clip', '--asym-clip', dest='psa_clip', type=float, default=0.03,
                    help='PSALoss probability clipping alpha (default: 0.03)')
parser.add_argument('--psa-adaptive-weight', '--asym-mixup-weight', dest='psa_adaptive_weight', action='store_true', default=True,
                    help='Enable the adaptive weighting w_adaptive for mixed samples (default: True)')
parser.add_argument('--psa-progressive', '--asym-progressive', dest='psa_progressive', action='store_true', default=True,
                    help='Enable the progressive schedule of w_as (default: True)')
parser.add_argument('--psa-warmup-epochs', '--asym-warmup-epochs', dest='psa_warmup_epochs', type=int, default=10,
                    help='Warm-up epochs of the progressive schedule w_as (default: 10)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn', action='store_true', default=True,
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='reduce',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--worker-seeding', type=str, default='all',
                    help='worker seed mode (default: all)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
# parser.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
#                     help='number of checkpoints to keep')
parser.add_argument('--checkpoint-hist', type=int, default=1, metavar='N',
                    help='number of checkpoints to keep (default: 1, only save best)')
parser.add_argument('-j', '--workers', type=int, default=8, metavar='N',
                    help='how many training processes to use (default: 8)')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--no-ddp-bb', action='store_true', default=False,
                    help='Force broadcast buffers for native DDP to off.')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                    help='name of train experiment, name of sub-folder for output')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--log-wandb', action='store_true', default=False,
                    help='log training and validation metrics to wandb')



def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)
    # Cache the args as a text string to save them in the output dir later
    args.device_name = torch.cuda.get_device_name()
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    setup_default_logging()
    args, args_text = _parse_args()
    if args.local_rank == 0:
        _logger.info(args_text)
    if args.log_wandb:
        if has_wandb:
            wandb.init(project=args.experiment, config=args)
        else:
            _logger.warning("You've requested to log metrics to wandb but package not found. "
                            "Metrics not being logged to wandb, try `pip install wandb`")

    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        _logger.info('Training with a single process on 1 GPUs.')
    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")

    random_seed(args.seed, args.rank)

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        grad_checkpoint=args.grad_checkpoint,
        checkpoint_stage=args.checkpoint_stage,)
        # checkpoint_path=args.initial_checkpoint,)
    if args.num_classes is None:
        assert hasattr(
            model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        # FIXME handle model default vs config num_classes more elegantly
        args.num_classes = model.num_classes

    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    if args.local_rank == 0:
        _logger.info(model)
        _logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    # move model to GPU, enable channels last layout if set
    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        assert not args.split_bn
        if has_apex and use_amp == 'apex':
            # Apex SyncBN preferred unless native amp is activated
            model = convert_syncbn_model(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.local_rank == 0:
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)

    if args.auto_lr:
        args.lr = (args.batch_size * args.world_size / 1024) * 1e-3
        args_text = yaml.load(args_text, Loader=yaml.FullLoader)
        args_text['lr'] = args.lr
        args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    if args.local_rank == 0:
        _logger.info(f'Initail learning rate: {args.lr}.')
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info(
                'Using native Torch AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            _logger.info('AMP not enabled. Training in float32.')

    if args.grad_checkpoint and args.local_rank == 0:
        _logger.info(
            f'Using gradient checkpointing, checkpoint stage: {args.checkpoint_stage}.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0)

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp == 'apex':
            # Apex DDP preferred unless native amp is activated
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[
                              args.local_rank], broadcast_buffers=not args.no_ddp_bb)
        # NOTE: EMA model does not need to be wrapped by DDP

    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        _logger.info('Scheduled epochs: {}'.format(num_epochs))

    if args.auto_val_split and args.val_split == args.train_split:
        # Auto split validation set from train data
        _logger.info(f'Auto splitting validation set from train data with ratio {args.val_split_ratio}')

        # Load the full train dataset to get indices
        full_train_dataset = datasets.ImageFolder(
            os.path.join(args.data_dir, args.train_split)
        )

        # Get indices for train/val split
        train_indices, val_indices = train_test_split(
            range(len(full_train_dataset)),
            test_size=args.val_split_ratio,
            random_state=42,
            stratify=full_train_dataset.targets
        )

        # Create custom dataset classes that mimic timm behavior
        from timm.data import create_transform
        data_config = resolve_data_config(vars(args), model=None, verbose=False)

        class TimmCompatibleSubset:
            def __init__(self, dataset, indices, transform):
                self.dataset = dataset
                self.indices = indices
                self.transform = transform
                # Copy important attributes from original dataset
                self.classes = dataset.classes
                self.class_to_idx = dataset.class_to_idx
                self.targets = [dataset.targets[i] for i in indices]

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, idx):
                img, target = self.dataset[self.indices[idx]]
                if self.transform:
                    img = self.transform(img)
                return img, target

        # Create transforms
        train_transform = create_transform(**data_config, is_training=True)
        val_transform = create_transform(**data_config, is_training=False)

        # Create compatible subsets
        dataset_train = TimmCompatibleSubset(full_train_dataset, train_indices, train_transform)
        dataset_eval = TimmCompatibleSubset(full_train_dataset, val_indices, val_transform)

        _logger.info(f'Auto-split: Train size: {len(train_indices)}, Val size: {len(val_indices)}')
    else:
        # Use separate train and val splits
        dataset_train = create_dataset(
            args.dataset, root=args.data_dir, split=args.train_split, is_training=True,
            class_map=args.class_map,
            download=args.dataset_download,
            batch_size=args.batch_size,
            repeats=args.epoch_repeats)
        dataset_eval = create_dataset(
            args.dataset, root=args.data_dir, split=args.val_split, is_training=False,
            class_map=args.class_map,
            download=args.dataset_download,
            batch_size=args.batch_size)

    # setup mixup / cutmix / Activation-Guided CutMix
    collate_fn = None
    mixup_fn = None
    act_cutmix_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None or args.act_cutmix
    if mixup_active:
        # Check if Activation-Guided CutMix is enabled
        if args.act_cutmix:
            # Use Activation-Guided CutMix instead of Mixup/CutMix
            _logger.info(f"Enabling Activation-Guided CutMix with beta={args.act_cutmix_beta}, prob={args.act_cutmix_prob}")
            # Note: Activation-Guided CutMix requires model for activation map computation
            # We'll create the act_cutmix_fn after model is created
            act_cutmix_active = True
        else:
            # Original Mixup/CutMix
            mixup_args = dict(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing, num_classes=args.num_classes)
            if args.prefetcher:
                # collate conflict (need to support deinterleaving in collate mixup)
                assert not num_aug_splits
                collate_fn = FastCollateMixup(**mixup_args)
            else:
                mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        worker_seeding=args.worker_seeding,
    )

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )

    # Initialize Activation-Guided CutMix function if enabled (after model is created)
    act_cutmix_fn = None
    if args.act_cutmix:
        _logger.info("Initializing Activation-Guided CutMix augmentation...")
        act_cutmix_fn = ActivationGuidedCutMix(
            model=model,
            beta=args.act_cutmix_beta,
            prob=args.act_cutmix_prob,
            num_classes=args.num_classes,
            use_cls_weights=args.act_cutmix_cls_weights,
            debug=args.act_cutmix_debug,
            save_visualizations=args.act_cutmix_visualize
        )
        _logger.info(f"Activation-Guided CutMix initialized with beta={args.act_cutmix_beta}, prob={args.act_cutmix_prob}")
        if args.act_cutmix_debug:
            _logger.info("Activation-Guided CutMix debug mode enabled")
        if args.act_cutmix_visualize:
            _logger.info(f"Activation-Guided CutMix visualizations will be saved to: {args.act_cutmix_visualize}")

    # setup loss function
    # Note: Activation-Guided CutMix uses soft labels with PSALoss
    if args.jsd_loss:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(
            num_splits=num_aug_splits, smoothing=args.smoothing)
    elif args.act_cutmix:
        # Activation-Guided CutMix with Progressive Soft Asymmetric Loss (PSALoss)
        if args.psa_loss:
            _logger.info(f"Using PSALoss: "
                        f"gamma_neg={args.psa_gamma_neg}, gamma_pos={args.psa_gamma_pos}, "
                        f"clip={args.psa_clip}, warmup={args.psa_warmup_epochs}")
            train_loss_fn = PSALoss(
                gamma_neg=args.psa_gamma_neg,
                gamma_pos=args.psa_gamma_pos,
                clip=args.psa_clip,
                warmup_epochs=args.psa_warmup_epochs,
                total_epochs=args.epochs,
                smoothing=args.smoothing,
                adaptive_weight=args.psa_adaptive_weight,
                progressive_enabled=args.psa_progressive
            )
        else:
            _logger.info("Using SoftTargetCrossEntropy for Activation-Guided CutMix")
            train_loss_fn = SoftTargetCrossEntropy()
    elif mixup_active:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(
                target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(
                smoothing=args.smoothing, target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(
                smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    if args.model_ema:
        ema_save_tag = 'ema'
        eval_metric = f'{args.eval_metric}_{ema_save_tag}'
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    resume_mode = args.resume
    if args.rank == 0:
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        output_dir = get_outdir(
            args.output if args.output else './output7/train', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(model=model, optimizer=optimizer, 
                                args=args, model_ema=model_ema, 
                                amp_scaler=loss_scaler, checkpoint_dir=output_dir, 
                                recovery_dir=output_dir, decreasing=decreasing, 
                                max_history=args.checkpoint_hist)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)
    try:
        if args.rank == 0:
            from datetime import timedelta
            start_time = time.time()
            overall_pbar = tqdm(total=num_epochs - start_epoch,
                               desc='Overall Training',
                               unit='epoch',
                               ncols=120,
                               bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        for epoch in range(start_epoch, num_epochs):
            if hasattr(train_loss_fn, 'set_epoch'):
                train_loss_fn.set_epoch(epoch)
            
            if args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)

            train_metrics = train_one_epoch(
                epoch, model, loader_train, 
                optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, 
                saver=saver, output_dir=output_dir,
                amp_autocast=amp_autocast, 
                loss_scaler=loss_scaler, 
                model_ema=model_ema, mixup_fn=mixup_fn,
                act_cutmix_fn=act_cutmix_fn,
                num_epochs=num_epochs)

            if (epoch % args.val_freq == 0) or epoch > args.val_start_epoch or resume_mode:
                resume_mode = False
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    if args.local_rank == 0:
                        _logger.info("Distributing BatchNorm running means and vars")
                    distribute_bn(model, args.world_size,
                                  args.dist_bn == 'reduce')

                eval_metrics = validate(model, loader_eval, validate_loss_fn, 
                                        args, amp_autocast=amp_autocast)

                if model_ema is not None and not args.model_ema_force_cpu:
                    if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                        distribute_bn(model_ema, args.world_size,
                                      args.dist_bn == 'reduce')
                    ema_eval_metrics = validate(model_ema.module, loader_eval,
                                                validate_loss_fn, args,
                                                amp_autocast=amp_autocast,
                                                log_suffix=f'_{ema_save_tag}')
                    eval_metrics.update(ema_eval_metrics)
            else:
                eval_metrics = {key: -100 for key in eval_metrics}

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if output_dir is not None:
                update_summary(epoch,
                               train_metrics,
                               eval_metrics,
                               os.path.join(output_dir, 'summary.csv'),
                               write_header=best_metric is None,
                               log_wandb=args.log_wandb and has_wandb)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(
                    epoch, metric=save_metric)

            if args.rank == 0:
                overall_pbar.update(1)

    except KeyboardInterrupt:
        pass

    if args.rank == 0:
        overall_pbar.close()
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        _logger.info(f'*** Training completed in {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f} ***')

    if best_metric is not None:
        _logger.info(
            f'*** Best metric ({eval_metric}): {best_metric} (epoch {best_epoch}) ***')


def train_one_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, output_dir=None, amp_autocast=suppress,
        loss_scaler=None, model_ema=None, mixup_fn=None, act_cutmix_fn=None, num_epochs=None):

    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(
        optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()
    # torch.cuda.empty_cache()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    
    epoch_str = f'{epoch}/{num_epochs-1}' if num_epochs else str(epoch)
    pbar = tqdm(enumerate(loader), total=len(loader), 
                desc=f'Epoch {epoch_str} [Train]',
                disable=args.local_rank != 0,
                ncols=120)
    
    for batch_idx, (input, target) in pbar:
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        # Flags for augmentation
        use_act_cutmix = act_cutmix_fn is not None
        target_a, target_b, lam_a, lam_b = None, None, None, None

        if not args.prefetcher:
            input = input.cuda()
            target = target.cuda()

        # Apply Activation-Guided CutMix (activation-map guided) — must run in the training loop because it
        # needs the model on GPU for activation-map computation, so it cannot be
        # delegated to the prefetcher/collate pipeline.
        if use_act_cutmix:
            if not input.is_cuda:
                input = input.cuda()
                target = target.cuda()
            input, target_a, target_b, lam_a, lam_b = act_cutmix_fn(input, target)
        elif not args.prefetcher and mixup_fn is not None:
            input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        with amp_autocast():
            output = model(input)
            # Compute loss for Activation-Guided CutMix with PSALoss
            if use_act_cutmix:
                if target_a is not None and lam_b is not None and lam_b.sum().item() > 0:
                    # Activation-Guided CutMix需要将类别索引转换为one-hot软标签
                    num_classes = output.size(-1)
                    batch_size = target_a.size(0)

                    target_a_onehot = torch.zeros(batch_size, num_classes, device=output.device)
                    target_b_onehot = torch.zeros(batch_size, num_classes, device=output.device)

                    target_a_onehot.scatter_(1, target_a.view(-1, 1), 1)
                    target_b_onehot.scatter_(1, target_b.view(-1, 1), 1)

                    target_mixed = lam_a.view(-1, 1) * target_a_onehot + lam_b.view(-1, 1) * target_b_onehot

                    if hasattr(loss_fn, 'forward') and hasattr(loss_fn, 'progressive_adjust'):
                        loss = loss_fn(output, target_mixed, lam_a, lam_b, epoch, num_epochs)
                    else:
                        loss = loss_fn(output, target_mixed)
                else:
                    num_classes = output.size(-1)
                    batch_size = target.size(0)
                    target_soft = torch.zeros(batch_size, num_classes, device=output.device)
                    target_soft.scatter_(1, target.view(-1, 1), 1)
                    
                    # use PSALoss
                    if hasattr(loss_fn, 'forward') and hasattr(loss_fn, 'progressive_adjust'):
                        lam_a_dummy = torch.ones(batch_size, device=output.device)
                        lam_b_dummy = torch.zeros(batch_size, device=output.device)
                        loss = loss_fn(output, target_soft, lam_a_dummy, lam_b_dummy, epoch, num_epochs)
                    else:
                        loss = loss_fn(output, target_soft)
            else:
                loss = loss_fn(output, target)

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(
                    model, exclude_head='agc' in args.clip_mode),
                create_graph=second_order)
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(
                        model, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)
            optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        if args.distributed:
            reduced_loss = reduce_tensor(loss.data, args.world_size)
            losses_m.update(reduced_loss.item(), input.size(0))
        
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.local_rank == 0:
                pbar.set_postfix({
                    'Loss': f'{losses_m.avg:.4f}',
                    'LR': f'{lr:.2e}',
                    'Speed': f'{input.size(0) * args.world_size / batch_time_m.avg:.1f} img/s'
                })
                
                if batch_idx % (args.log_interval * 5) == 0 or last_batch:
                    current_time = time.strftime(
                        '%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                    _logger.info(
                        '[{}]  Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                        'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                        '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                        'LR: {lr:.3e}  '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                            current_time,
                            epoch,
                            batch_idx, len(loader),
                            100. * batch_idx / last_idx,
                            loss=losses_m,
                            batch_time=batch_time_m,
                            rate=input.size(0) * args.world_size /
                            batch_time_m.val,
                            rate_avg=input.size(
                                0) * args.world_size / batch_time_m.avg,
                            lr=lr,
                            data_time=data_time_m))

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(
                            output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(
                num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for
    
    pbar.close()

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])


def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    
    all_preds = []
    all_targets = []
    all_probs = []

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    
    pbar = tqdm(enumerate(loader), total=len(loader),
                desc='[Validate]',
                disable=args.local_rank != 0,
                ncols=100)
    
    with torch.no_grad():
        for batch_idx, (input, target) in pbar:
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
                if isinstance(output, (tuple, list)):
                    output = output[0]
                loss = loss_fn(output, target)

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(
                    0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            
            # 收集预测结果
            probs = torch.softmax(output, dim=1)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            
            if args.local_rank == 0:
                pbar.set_postfix({
                    'Loss': f'{losses_m.avg:.4f}',
                    'Acc@1': f'{top1_m.avg:.2f}%',
                    'Acc@5': f'{top5_m.avg:.2f}%'
                })
                
                if last_batch or batch_idx % max(1, args.log_interval // 2) == 0:
                    log_name = 'Test' + log_suffix
                    _logger.info(
                        '{0}: [{1:>4d}/{2}]  '
                        'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                        'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                        'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                        'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                            log_name, batch_idx, last_idx, batch_time=batch_time_m,
                            loss=losses_m, top1=top1_m, top5=top5_m))
    
    pbar.close()
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Precision, Recall, F1 (macro)
    precision_macro = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    try:
        auc_macro = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='macro')
    except:
        auc_macro = 0.0

    metrics = OrderedDict([
        (f'loss{log_suffix}', losses_m.avg),
        (f'top1{log_suffix}', top1_m.avg), 
        (f'top5{log_suffix}', top5_m.avg),
        (f'precision{log_suffix}', precision_macro * 100),
        (f'recall{log_suffix}', recall_macro * 100),
        (f'f1{log_suffix}', f1_macro * 100),
        (f'auc{log_suffix}', auc_macro)
    ])

    return metrics


if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()
