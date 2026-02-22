import numpy as np
import torch
from skimage.transform import resize
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from scipy import stats

class XAIMetric:
    def __init__(self, heatmap, ground_truth_mask, threshold=0.3):
        """
        Initialize the XAIMetric class with a heatmap and ground truth mask.

        Args:
            heatmap (numpy.ndarray): The relevance heatmap (H, W, C) or (H, W).
            ground_truth_mask (numpy.ndarray or torch.Tensor): Binary ground truth mask (H, W).
            threshold (float): Threshold for relevance map binarization (default: 0.3)
        """
        self.heatmap = heatmap
        self.ground_truth_mask = ground_truth_mask
        self.threshold = threshold

        # Convert heatmap to numpy.ndarray if necessary
        if not isinstance(self.heatmap, np.ndarray):
            self.heatmap = self.heatmap.cpu().detach().numpy()

        # Reduce heatmap to a single channel if it's 3D
        if self.heatmap.ndim == 3:
            self.relevance_map = np.mean(self.heatmap, axis=-1)  # Use mean instead of sum for better scaling

        else:
            self.relevance_map = self.heatmap

        # Convert ground truth mask to numpy.ndarray if it's a torch.Tensor
        if isinstance(self.ground_truth_mask, torch.Tensor):
            self.ground_truth_mask = self.ground_truth_mask.cpu().detach().numpy()
            self.ground_truth_mask = np.squeeze(self.ground_truth_mask)

        # Ensure the mask is reduced to (H, W)
        if self.ground_truth_mask.ndim == 4:  # Case: (B, C, H, W)
            self.ground_truth_mask = self.ground_truth_mask[0, 0]
        elif self.ground_truth_mask.ndim == 3:  # Case: (C, H, W)
            self.ground_truth_mask = self.ground_truth_mask[0]

        # Ensure binary nature of ground truth mask
        self.ground_truth_mask = (self.ground_truth_mask > 0).astype(np.float32)

        # Ensure relevance map and ground truth mask have matching shapes
        if self.relevance_map.shape != self.ground_truth_mask.shape:
            self.relevance_map = resize(
                self.relevance_map, 
                self.ground_truth_mask.shape, 
                order=3, 
                mode='reflect', 
                anti_aliasing=True
            )
        
        if self.relevance_map.shape != self.ground_truth_mask.shape:
            raise ValueError(
                f"Shape mismatch after resizing: relevance_map={self.relevance_map.shape}, "
                f"ground_truth_mask={self.ground_truth_mask.shape}"
            )
        
        # Robust normalization using percentile-based scaling
        p_low, p_high = np.percentile(self.relevance_map, [1, 99])
        self.relevance_map = np.clip(self.relevance_map, p_low, p_high)
        denom = p_high - p_low
        if denom > 0:
            self.relevance_map = (self.relevance_map - p_low) / denom
        else:
            self.relevance_map = np.zeros_like(self.relevance_map)
        
        # Apply threshold
        self.relevance_map_binary = (self.relevance_map > self.threshold).astype(np.float32)

    def PointingGame(self):
        """
        Compute Pointing Game (Hit Rate) metric.
        Checks if the most salient point in the heatmap falls within the ground truth explanation region.

        Returns:
            float or str: Pointing Game score or "normal" if no ground truth mask
        """
        if np.sum(self.ground_truth_mask) == 0:
            return "normal"

        # Find the coordinates of the maximum value in the relevance map
        max_coords = np.unravel_index(np.argmax(self.relevance_map), self.relevance_map.shape)
        
        # Check if the maximum point falls within the ground truth mask
        hit = self.ground_truth_mask[max_coords] > 0
        
        return 1.0 if hit else 0.0

    def IoU(self):
        """
        Compute Intersection over Union (IoU) between relevance map and ground truth mask.
        
        Returns:
            float or str: IoU score or "normal" if no ground truth mask
        """
        if np.sum(self.ground_truth_mask) == 0:
            return "normal"

        intersection = np.sum(self.relevance_map_binary * self.ground_truth_mask)
        union = np.sum((self.relevance_map_binary + self.ground_truth_mask) > 0)
        
        if union == 0:
            return 0.0
            
        return round(intersection / union, 4)

    def RMA(self):
        """
        Compute Relevance Mass Accuracy (RMA).

        Returns:
            float or str: Relevance Mass Accuracy or "normal" if no ground truth mask.
        """
        if np.sum(self.ground_truth_mask) == 0:
            return "normal"

        R_within = np.sum(self.relevance_map * self.ground_truth_mask)
        R_total = np.sum(self.relevance_map)

        if R_total == 0:
            return 0.0

        return round(R_within / R_total, 4)

    def RRA(self):
        """
        Compute Relevance Rank Accuracy (RRA).

        Returns:
            float or str: Relevance Rank Accuracy or "normal" if no ground truth mask.
        """
        if np.sum(self.ground_truth_mask) == 0:
            return "normal"

        relevance_values = self.relevance_map.flatten()
        ground_truth_mask_flat = self.ground_truth_mask.flatten()
        ground_truth_indices = np.where(ground_truth_mask_flat == 1)[0]
        K = len(ground_truth_indices)
        
        if K == 0:
            return 0.0

        top_k_indices = np.argsort(relevance_values)[-K:][::-1]
        intersection = np.intersect1d(top_k_indices, ground_truth_indices)

        return round(len(intersection) / K, 4)

    @staticmethod
    def aggregate_metrics(metrics_dict):
        """
        Aggregate metrics across images with confidence intervals.
        
        Args:
            metrics_dict (dict): Dictionary of image names to (RMA, RRA, IoU, PointingGame) tuples
            
        Returns:
            dict: Dictionary containing mean, std, and confidence intervals for metrics
        """
        metrics = {
            'RMA': [m[0] for m in metrics_dict.values() if isinstance(m[0], (int, float))],
            'RRA': [m[1] for m in metrics_dict.values() if isinstance(m[1], (int, float))],
            'IoU': [m[2] for m in metrics_dict.values() if isinstance(m[2], (int, float))] if len(metrics_dict[next(iter(metrics_dict))]) > 2 else [],
            'PointingGame': [m[3] for m in metrics_dict.values() if isinstance(m[3], (int, float))] if len(metrics_dict[next(iter(metrics_dict))]) > 3 else []
        }
        
        normal_count = sum(1 for m in metrics_dict.values() if m[0] == "normal")
        
        results = {
            'total_cases': len(metrics_dict),
            'normal_cases': normal_count,
            'valid_cases': len(metrics_dict) - normal_count
        }
        
        for metric_name, values in metrics.items():
            if len(values) > 0:
                mean = np.mean(values)
                std = np.std(values)
                ci = stats.t.interval(0.95, len(values)-1, loc=mean, scale=std/np.sqrt(len(values)))
                
                results.update({
                    f'{metric_name}_mean': mean,
                    f'{metric_name}_std': std,
                    f'{metric_name}_ci_lower': ci[0],
                    f'{metric_name}_ci_upper': ci[1]
                })
            
        return results

    @staticmethod
    def plot_metrics_distribution(metrics_df, save_dir, mode, today):
        """Plot the distribution of metrics with enhanced visualization."""
        plt.figure(figsize=(15, 5))
        
        metrics = ['RMA', 'RRA', 'IoU', 'PointingGame'] if 'PointingGame' in metrics_df.columns else ['RMA', 'RRA', 'IoU'] if 'IoU' in metrics_df.columns else ['RMA', 'RRA']
        
        for i, metric in enumerate(metrics, 1):
            plt.subplot(1, len(metrics), i)
            data = metrics_df[metric][metrics_df[metric] != "normal"].astype(float)
            
            # Create violin plot with box plot inside
            sns.violinplot(data=data, inner='box')
            sns.stripplot(data=data, color='red', alpha=0.3, size=4)
            
            plt.title(f'{mode} {metric} Distribution')
            plt.ylabel('Score')
            
            # Add mean and CI
            mean = data.mean()
            ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=data.std()/np.sqrt(len(data)))
            plt.axhline(y=mean, color='g', linestyle='--', label=f'Mean: {mean:.3f}')
            plt.axhline(y=ci[0], color='r', linestyle=':', label=f'95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]')
            plt.axhline(y=ci[1], color='r', linestyle=':')
            plt.legend(fontsize='small')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{mode}_metrics_distribution_{today}.png"), dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def compare_attention_modes(metrics_dict_global, metrics_dict_local, metrics_dict_dualstream, save_dir, today):
        """Compare metrics across different attention modes with enhanced visualization."""
        # Aggregate metrics for each mode
        stats = {
            'Global': XAIMetric.aggregate_metrics(metrics_dict_global),
            'Local': XAIMetric.aggregate_metrics(metrics_dict_local),
            'Dual-stream': XAIMetric.aggregate_metrics(metrics_dict_dualstream)
        }
        
        # Create comparison plots
        plt.figure(figsize=(15, 10))
        
        metrics = ['RMA', 'RRA', 'IoU', 'PointingGame'] if 'PointingGame_mean' in stats['Global'] else ['RMA', 'RRA', 'IoU'] if 'IoU_mean' in stats['Global'] else ['RMA', 'RRA']
        
        for i, metric in enumerate(metrics):
            plt.subplot(2, len(metrics), i+1)
            
            # Prepare data for plotting
            means = [s[f'{metric}_mean'] for s in stats.values()]
            errors = [(s[f'{metric}_ci_upper'] - s[f'{metric}_ci_lower'])/2 for s in stats.values()]
            
            # Create bar plot with error bars
            bars = plt.bar(list(stats.keys()), means, yerr=errors, capsize=5)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom')
            
            plt.title(f'{metric} Comparison')
            plt.ylabel('Score')
            
            # Add individual points
            x_positions = range(len(stats))
            for idx, mode in enumerate(stats.keys()):
                data = [m[metrics.index(metric)] for m in globals()[f'metrics_dict_{mode.lower()}'.replace('-', '_')].values() 
                       if isinstance(m[metrics.index(metric)], (int, float))]
                plt.scatter([idx] * len(data), data, alpha=0.2, color='red')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"attention_modes_comparison_{today}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed statistics
        detailed_stats = []
        for mode, stat in stats.items():
            row = {'Mode': mode}
            row.update({k: v for k, v in stat.items() if not k.startswith('_ci_')})
            detailed_stats.append(row)
        
        stats_df = pd.DataFrame(detailed_stats)
        stats_df.to_csv(os.path.join(save_dir, f"attention_modes_statistics_{today}.csv"), index=False)