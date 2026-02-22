"""
Sanity Check Script for DualStream Attention Model
Adapted to work with the dualstream attention architecture and saved weights.
"""

import os
import random
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from skimage.feature import hog
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from tqdm import tqdm
import argparse

# Import your model and dataset loaders
from models.dualstream_attention import DualStream_Attention
from models.BUI_loader import BUI_Dataset
from models.distal_myopathy_loader import Distal_Dataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set seeds for reproducibility
torch.manual_seed(12)
torch.cuda.manual_seed(12)
np.random.seed(12)
random.seed(12)
torch.backends.cudnn.deterministic = True


class DualStreamSanityCheck:
    """Sanity check for dualstream attention model."""
    
    def __init__(self, model, number_of_samples, dataloader, dataset_name="Distal", make_single_channel=True):
        self.dataset_name = dataset_name
        self.make_single_channel = make_single_channel
        self.model = model
        self.number_of_samples = number_of_samples
        self.dataloader = dataloader
        self.dataiter = iter(dataloader)
        self.image = None
        self.label = None
        self.results = {}
        self.avg_spearman = {}
        self.avg_spearman_abs = {}
        self.spearman_rank_correlations = {}
        self.spearman_rank_correlations_abs = {}
        self.HOGs = {}
        self.pearson_HOGs = {}
        self.avg_pearson_HOGs = {}
        self.SSIMs = {}
        self.avg_SSIMs = {}
        self.spearman_path = None
        self.current_sample_id = 0
        self.saliency_methods = ['Global', 'Local', 'Fusion_Gate', 'Fusion_Concat', 'Fusion_Product']
    
    def weight_randomization(self):
        """Progressively randomize model weights and compute saliency maps."""
        all_mods = []
        for mod in self.model.modules():
            all_mods.append(mod)
        
        # Get initial saliency maps
        self.get_all_saliency_maps()

        # NOTE:
        # The sequence is: original → progressively more randomized.

        # Progressively randomize from top to bottom
        for i in range(len(all_mods)-1, -1, -1):
            module = all_mods[i]
            change = False
            
            if hasattr(module, 'weight'):
                change = True
                # Randomize weights (Gaussian noise)
                torch.nn.init.normal_(module.weight, 0., 0.1)
            
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.normal_(module.bias, 0., 0.1)

            # At each "key" layer, record saliency maps after randomization
            if change and (
                isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d))
                or "Attention" in module.__class__.__name__
                or "Encoder" in module.__class__.__name__
            ):
                self.get_all_saliency_maps()
        
        # Calculate metrics
        self.calc_spearman()
        self.calc_HOG()
        self.calc_SSIM()
        self.archive()
    
    def get_all_saliency_maps(self):
        """Generate saliency maps using all methods."""
        # Initialize results dict if empty
        if len(self.results) == 0:
            for method in self.saliency_methods:
                self.results[method] = []
        
        self.model.eval()
        
        # Generate saliency maps using inherent interpretability (attention maps)
        with torch.no_grad():
            # Get all attention maps from the model
            g_out, l_out, f_out, attn_data = self.model(self.image)
            
            # Extract attention maps for each branch (using same naming as incremental_deletion.py)
            attention_maps = {
                'Global': attn_data.get('g_attn', None),
                'Local': attn_data.get('l_attn', None),
                'Fusion_Gate': None,  # Will get from eval_fusion_branches
                'Fusion_Concat': None,
                'Fusion_Product': None
            }
            
            # Get fusion attention maps from eval branches
            try:
                eval_results = self.model.evaluate_all_fusion_types(self.image)
                attention_maps['Fusion_Gate'] = eval_results['gate']['attention']
                attention_maps['Fusion_Concat'] = eval_results['concat']['attention']
                attention_maps['Fusion_Product'] = eval_results['product']['attention']
            except:
                # Fallback: use main fusion branch attention
                attention_maps['Fusion_Gate'] = attn_data.get('attns', None)
        
        # Process each attention map
        for method in self.saliency_methods:
            attn_map = attention_maps.get(method, None)
            
            if attn_map is None:
                # Skip if attention map not available
                continue
            
            # Convert to numpy and ensure proper shape
            if isinstance(attn_map, torch.Tensor):
                saliency_np = attn_map.detach().cpu().numpy()
            else:
                saliency_np = np.array(attn_map)
            
            # Handle batch dimension if present
            if saliency_np.ndim == 4:  # (B, C, H, W)
                saliency_np = saliency_np[0]  # Take first batch
            elif saliency_np.ndim == 3:  # (C, H, W) or (H, W, C)
                if saliency_np.shape[0] < saliency_np.shape[-1]:
                    # Likely (C, H, W)
                    if self.make_single_channel:
                        saliency_np = saliency_np.mean(axis=0, keepdims=True)  # (1, H, W)
                    else:
                        pass  # Keep as (C, H, W)
                else:
                    # Likely (H, W, C) -> convert to (C, H, W)
                    saliency_np = saliency_np.transpose(2, 0, 1)
                    if self.make_single_channel:
                        saliency_np = saliency_np.mean(axis=0, keepdims=True)
            elif saliency_np.ndim == 2:  # (H, W)
                saliency_np = saliency_np[np.newaxis, :, :]  # (1, H, W)
            
            # Resize to input image size if needed
            if saliency_np.shape[-2:] != self.image.shape[-2:]:
                attn_tensor = torch.from_numpy(saliency_np).unsqueeze(0).float()
                resized = F.interpolate(attn_tensor, size=self.image.shape[-2:], 
                                       mode='bilinear', align_corners=False)
                saliency_np = resized.squeeze(0).numpy()
            
            # Normalize saliency
            saliency_np = saliency_np / (np.max(np.abs(saliency_np).flatten()) + 1e-8)
            
            # Ensure final shape is (1, H, W) for consistency
            if saliency_np.ndim == 3 and saliency_np.shape[0] > 1:
                saliency_np = saliency_np.mean(axis=0, keepdims=True)
            
            self.results[method].append(torch.from_numpy(saliency_np))
    
    def calc_spearman(self):
        """Calculate Spearman rank correlations."""
        for k in self.results.keys():
            if k not in self.spearman_rank_correlations:
                self.spearman_rank_correlations[k] = []
                self.spearman_rank_correlations_abs[k] = []
            
            if len(self.results[k]) > 1:
                # Compare first (original) with all others
                original = self.results[k][0].view(-1).cpu().numpy()
                for j in range(1, len(self.results[k])):
                    randomized = self.results[k][j].view(-1).cpu().numpy()
                    res = stats.spearmanr(original, randomized)
                    res_abs = stats.spearmanr(np.abs(original), np.abs(randomized))
                    self.spearman_rank_correlations[k].append(res[0])
                    self.spearman_rank_correlations_abs[k].append(res_abs[0])
    
    def calc_HOG(self):
        """Calculate HOG (Histogram of Oriented Gradients) similarity."""
        for k in self.results.keys():
            if k not in self.HOGs:
                self.HOGs[k] = []
            
            for j in range(len(self.results[k])):
                saliency_np = self.results[k][j].cpu().numpy()

                # Robustly reduce to a 2D (H, W) image for HOG
                # Handle shapes like (1, H, W), (C, H, W), (1, 1, H, W), etc.
                if saliency_np.ndim == 4:
                    # Assume (N, C, H, W) -> take first sample
                    saliency_np = saliency_np[0]
                if saliency_np.ndim == 3:
                    # (C, H, W) -> average over channels -> (H, W)
                    saliency_np = saliency_np.mean(axis=0)
                if saliency_np.ndim != 2:
                    # As a final fallback, try to squeeze and check again
                    saliency_np = np.squeeze(saliency_np)
                    if saliency_np.ndim != 2:
                        # If still not 2D, skip this entry to avoid crashing
                        continue

                try:
                    hog_features = hog(saliency_np, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
                    self.HOGs[k].append(hog_features)
                except ValueError:
                    # If HOG still complains for any reason, skip this sample
                    continue
        
        # Calculate Pearson correlations
        for k in self.HOGs.keys():
            if k not in self.pearson_HOGs:
                self.pearson_HOGs[k] = []
            
            if len(self.HOGs[k]) > 1:
                original_hog = self.HOGs[k][0]
                for j in range(1, len(self.HOGs[k])):
                    res = stats.pearsonr(original_hog, self.HOGs[k][j])
                    self.pearson_HOGs[k].append(res[0])
    
    def calc_SSIM(self):
        """Calculate Structural Similarity Index."""
        for k in self.results.keys():
            if k not in self.SSIMs:
                self.SSIMs[k] = []
            
            if len(self.results[k]) > 1:
                original = self.results[k][0].cpu().numpy()
                # Convert to (H, W, C) format
                if len(original.shape) == 3:
                    original = original.transpose(1, 2, 0)
                elif len(original.shape) == 2:
                    original = original[:, :, np.newaxis]
                
                # Ensure 3 channels
                if original.shape[2] == 1:
                    original = np.repeat(original, 3, axis=2)

                # Skip if image is too small for a stable SSIM window
                if original.shape[0] * original.shape[1] < 9:
                    continue
                
                for j in range(1, len(self.results[k])):
                    randomized = self.results[k][j].cpu().numpy()
                    if len(randomized.shape) == 3:
                        randomized = randomized.transpose(1, 2, 0)
                    elif len(randomized.shape) == 2:
                        randomized = randomized[:, :, np.newaxis]
                    
                    if randomized.shape[2] == 1:
                        randomized = np.repeat(randomized, 3, axis=2)

                    # Normalize to [0, 1]
                    original_norm = (original - original.min()) / (original.max() - original.min() + 1e-8)
                    randomized_norm = (randomized - randomized.min()) / (randomized.max() - randomized.min() + 1e-8)

                    # Choose a valid odd win_size and guard against tiny images
                    max_win = min(original.shape[0], original.shape[1], randomized.shape[0], randomized.shape[1])
                    if max_win < 3:
                        continue
                    # Use smallest odd number up to 7 that fits
                    win_size = min(7, max_win)
                    if win_size % 2 == 0:
                        win_size -= 1
                    if win_size < 3:
                        continue

                    try:
                        ssim_val = ssim(original_norm, randomized_norm, data_range=1.0, channel_axis=2, win_size=win_size)
                        self.SSIMs[k].append(ssim_val)
                    except ZeroDivisionError:
                        # Extremely small or degenerate case; skip
                        continue
    
    def archive(self):
        """Archive results and reset for next sample."""
        spearman = self.spearman_rank_correlations
        spearman_abs = self.spearman_rank_correlations_abs
        pearson_HOG = self.pearson_HOGs
        ssim_vals = self.SSIMs
        
        for k in spearman.keys():
            # Convert current lists to numpy arrays (may differ in length across samples)
            cur_spr = np.asarray(spearman[k]) if len(spearman[k]) > 0 else np.array([])
            cur_spr_abs = np.asarray(spearman_abs[k]) if len(spearman_abs[k]) > 0 else np.array([])
            cur_hog = np.asarray(pearson_HOG[k]) if len(pearson_HOG[k]) > 0 else np.array([])
            cur_ssim = np.asarray(ssim_vals[k]) if len(ssim_vals[k]) > 0 else np.array([])

            if k not in self.avg_spearman:
                # First sample: just store
                self.avg_spearman[k] = cur_spr
                self.avg_spearman_abs[k] = cur_spr_abs
                self.avg_pearson_HOGs[k] = cur_hog
                self.avg_SSIMs[k] = cur_ssim
            else:
                # Later samples: align by truncating to the minimum common length
                if cur_spr.size > 0 and self.avg_spearman[k].size > 0:
                    L = min(len(self.avg_spearman[k]), len(cur_spr))
                    self.avg_spearman[k][:L] += cur_spr[:L]
                if cur_spr_abs.size > 0 and self.avg_spearman_abs[k].size > 0:
                    L = min(len(self.avg_spearman_abs[k]), len(cur_spr_abs))
                    self.avg_spearman_abs[k][:L] += cur_spr_abs[:L]
                if cur_hog.size > 0 and self.avg_pearson_HOGs[k].size > 0:
                    L = min(len(self.avg_pearson_HOGs[k]), len(cur_hog))
                    self.avg_pearson_HOGs[k][:L] += cur_hog[:L]
                if cur_ssim.size > 0 and self.avg_SSIMs[k].size > 0:
                    L = min(len(self.avg_SSIMs[k]), len(cur_ssim))
                    self.avg_SSIMs[k][:L] += cur_ssim[:L]
        
        # Reset for next sample
        self.spearman_rank_correlations = {}
        self.spearman_rank_correlations_abs = {}
        self.HOGs = {}
        self.pearson_HOGs = {}
        self.SSIMs = {}
        self.results = {}
        self.current_sample_id += 1
    
    def plot_results(self, num_plot):
        """Plot sanity check results."""
        to_be_plot = [self.avg_spearman, self.avg_spearman_abs, self.avg_SSIMs, self.avg_pearson_HOGs]
        name_of_plot = ["SPR", "ABS SPR", "SSIM", "HOG"]
        
        if self.spearman_path is None:
            now = str(datetime.now()).replace(':', '-')
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Save under dataset-named folder: sanity_checks/plots/{DatasetName}/{timestamp}
            plot_dir = os.path.join(script_dir, "sanity_checks", "plots", self.dataset_name, now)
            os.makedirs(plot_dir, exist_ok=True)
            self.spearman_path = plot_dir
        
        sns.set_style("darkgrid")
        for i in range(len(to_be_plot)):
            fig, ax = plt.subplots(figsize=(12, 8))
            for k in to_be_plot[i].keys():
                if len(to_be_plot[i][k]) > 0:
                    tmp = np.divide(to_be_plot[i][k], num_plot)
                    # Sanitize filename: remove invalid Windows characters (*, :, etc.)
                    safe_name = k.replace(' ', '_').replace('*', 'x').replace(':', '-').replace('/', '_')
                    np.save(os.path.join(self.spearman_path, 
                                       f"{safe_name}_{i}_numberOfSamples_{num_plot}.npy"), tmp)
                    plt.plot(tmp, label=str(k), linewidth=3)
            
            legend = plt.legend(loc='best', fontsize=18, title_fontsize=20, framealpha=0.9)
            for text in legend.get_texts():
                text.set_fontweight('bold')
            plt.ylim(-1, 1.1)
            plt.ylabel(name_of_plot[i], fontsize=20, fontweight='bold', labelpad=15)
            plt.xlabel("Layer Index (from top to bottom)", fontsize=20, fontweight='bold', labelpad=15)
            plt.title(f"{name_of_plot[i]} - {num_plot} samples", fontsize=24, fontweight='bold', pad=20)
            plt.xticks(fontsize=16, fontweight='bold')
            plt.yticks(fontsize=16, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.spearman_path, f"plot_{name_of_plot[i]}_numberOfSamples_{num_plot}.png"), dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close()
    
    def run(self):
        """Run sanity check."""
        plot_interval = 10
        for i in tqdm(range(self.number_of_samples), desc="Running sanity checks"):
            try:
                image, label = next(self.dataiter)
            except StopIteration:
                self.dataiter = iter(self.dataloader)
                image, label = next(self.dataiter)
            
            self.image = image.to(device)
            self.label = label.to(device)
            self.weight_randomization()
            
            if (i + 1) % plot_interval == 0:
                self.plot_results(i + 1)
        
        # Final plot
        self.plot_results(self.number_of_samples)
        print(f"\nSanity check completed! Dataset: {self.dataset_name}. Results saved in: {self.spearman_path}")


def load_model_and_data(dataset_name, weight_path, img_size=64, batch_size=1):
    """Load model and dataset for sanity checks."""
    
    # Dataset configuration
    if dataset_name == "BUI":
        dataset_dir = "D:/Medical Image Datasets/Breast Ultrasound Images dataset/Dataset_BUSI_with_GT"
        class_name = {0: 'Benign', 1: 'Malignant'}
        full_dataset_cv = True
    elif dataset_name == "Distal":
        dataset_dir = "D:/Medical Image Datasets/Distal Myopathies/Data"
        class_name = {0: 'Healthy', 1: 'Affected'}
        full_dataset_cv = True
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    num_cls = len(class_name)
    backbone_model = "resnet50"
    out_channels = 2048
    weights = ResNet50_Weights.IMAGENET1K_V2
    # IMPORTANT: use the same fusion_type as during training ("gate"), otherwise the saved weights (which expect AttentionGate) will not match.
    fusion_type = "gate"
    
    # Create transform (minimal, no augmentation for sanity checks)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    # Load dataset
    seed = 42
    if dataset_name == "BUI":
        data_loader = BUI_Dataset(dataset_dir, batch_size, img_size, list(class_name.values()), 
                                data_split=0.8, n_splits=5, data_transform=transform, 
                                use_cv='off', use_full_dataset_cv=full_dataset_cv, seed=seed)
    else:  # Distal
        data_loader = Distal_Dataset(dataset_dir, img_size, batch_size, test_size=0.2, 
                                    data_transform=transform, use_cv='off', n_splits=5, 
                                    use_full_dataset_cv=full_dataset_cv, seed=seed)
    
    loaders = data_loader.prepare_dataloaders()
    dataloader = loaders["train_loader"]  # Use train loader for sanity checks
    
    # Create model
    model = DualStream_Attention(
        global_net=backbone_model,
        local_net='bagnet33',
        num_cls=num_cls,
        in_channels=3,
        out_channels=out_channels,
        in_size=(img_size, img_size),
        global_weight=0.3,
        local_weight=0.3,
        fusion_weight=0.4,
        dropout=0.3,
        weights=weights,
        load_local=True,
        use_rgb=True,
        fusion_type=fusion_type).to(device)
    
    # Load weights
    print(f"Loading weights from: {weight_path}")
    state_dict = torch.load(weight_path, map_location=device)
    
    # Handle DataParallel wrapper if present
    if any(k.startswith('module.') for k in state_dict.keys()):
        model = nn.DataParallel(model)
        # Relax strictness to allow extra keys like eval_fusion_branches.*.fc.*
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            print(f"Warning: Ignoring unexpected keys in state_dict: {unexpected}")
        model = model.module  # Unwrap for sanity checks
    else:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            print(f"Warning: Ignoring unexpected keys in state_dict: {unexpected}")
    
    model.eval()
    print("Model loaded successfully!")
    
    return model, dataloader


def main():
    parser = argparse.ArgumentParser(
        description="Run sanity checks on dualstream attention model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # For Distal dataset (auto-detects weight path):
  python run_sanity_checks_dualstream.py --dataset Distal
  
  # For BUI dataset (auto-detects weight path):
  python run_sanity_checks_dualstream.py --dataset BUI
  
  # With custom weight path:
  python run_sanity_checks_dualstream.py --dataset Distal --weight_path "path/to/weights.pth"
        """
    )
    parser.add_argument('--dataset', type=str, default='Distal', choices=['BUI', 'Distal'],
                       help='Dataset to use (default: Distal)')
    parser.add_argument('--weight_path', type=str, default=None,
                       help='Path to model weights (.pth file). If not provided, will use default based on dataset.')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of samples for sanity check (default: 50)')
    parser.add_argument('--img_size', type=int, default=64,
                       help='Image size (default: 64)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size, should be 1 for sanity checks (default: 1)')
    
    args = parser.parse_args()
    
    # Auto-detect weight path if not provided
    if args.weight_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if args.dataset == "BUI":
            default_weight = os.path.join(script_dir, "BUI_weights", "best_model_fold_1_2025-07-18.pth")
        else:  # Distal
            default_weight = os.path.join(script_dir, "Distal_weights", "best_model_fold_1_2025-07-17.pth")
        
        if os.path.exists(default_weight):
            args.weight_path = default_weight
            print(f"Using default weight path: {args.weight_path}")
        else:
            print(f"ERROR: Weight file not found at default path: {default_weight}")
            print("Please provide --weight_path argument with the full path to your .pth file")
            print("\nExample:")
            print(f'  python run_sanity_checks_dualstream.py --dataset {args.dataset} --weight_path "path/to/your/weights.pth"')
            return
    
    # Verify weight file exists
    if not os.path.exists(args.weight_path):
        print(f"ERROR: Weight file not found: {args.weight_path}")
        return
    
    # Load model and data
    print(f"Dataset: {args.dataset}")
    print(f"Weights: {args.weight_path}")
    model, dataloader = load_model_and_data(args.dataset, args.weight_path, 
                                           args.img_size, args.batch_size)
    
    sanity_check = DualStreamSanityCheck(
        model=model,
        number_of_samples=args.num_samples,
        dataloader=dataloader,
        dataset_name=args.dataset,
        make_single_channel=True,
    )
    print(f"\nStarting sanity checks with {args.num_samples} samples (dataset: {args.dataset})...")
    sanity_check.run()
    
    print("\nSanity check completed successfully!")

if __name__ == "__main__":
    main()