import os
import torch
import pandas as pd
import numpy as np
from models.xai_metric import XAIMetric
from tqdm import tqdm

def save_attention_maps(model, data_loader, image_names, mask_loader, save_dir, device, img_size, mode="Dual-stream", today=None):
    """
    Save attention maps and compute XAI metrics with enhanced error handling and progress tracking.
    
    Args:
        model: The model to generate attention maps
        data_loader: DataLoader containing images
        image_names: List of image names
        mask_loader: DataLoader containing masks (can be None)
        save_dir: Directory to save outputs
        device: Device to run the model on
        img_size: Size of the images
        mode: Attention mode ("Global", "Local", or "Dual-stream")
        today: Date string for file naming
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # Dictionary to store metrics for each image
    metrics_dict = {}
    
    # Store metrics for all modes to enable comparison
    if mode == "Dual-stream":
        global_metrics = {}
        local_metrics = {}
        dualstream_metrics = {}

    # Check if mask loader exists and prepare iterator
    has_masks = mask_loader is not None
    if has_masks:
        val_mask_iter = iter(mask_loader)
    else:
        print(f"Note: No masks provided for {mode} attention maps. XAI metrics will not be computed.")

    # Set up progress bar
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Processing {mode} attention maps")

    with torch.no_grad():
        for batch_idx, batch in pbar:
            try:
                # Process batch based on format
                if len(batch) == 4:  # ISIC dataset
                    images, _, labels, _ = batch
                elif len(batch) == 3:  # Other datasets with masks
                    images, _, labels = batch
                elif len(batch) == 2:  # Datasets without masks
                    images, labels = batch
                else:
                    raise ValueError(f"Unexpected batch format with {len(batch)} elements")

                images = images.to(device, memory_format=torch.channels_last)
                
                # Get masks if available
                if has_masks:
                    try:
                        masks = next(val_mask_iter)[0]
                        masks = masks.to(device)
                    except StopIteration:
                        print(f"Warning: Mask loader exhausted at batch {batch_idx + 1}")
                        has_masks = False
                        masks = None
                else:
                    masks = None

                # Process each image in the batch
                for j in range(images.size(0)):
                    idx = batch_idx * data_loader.batch_size + j
                    if idx >= len(image_names):
                        print(f"Warning: Index {idx} out of range for image_names. Skipping.")
                        continue

                    current_image = images[j].unsqueeze(0)
                    current_mask = masks[j].unsqueeze(0) if masks is not None else None
                    
                    try:
                        # Generate and save attention maps for all modes if this is Dual-stream
                        if mode == "Dual-stream":
                            # Get attention maps for all modes
                            global_map = model.get_global_attention_map(current_image)
                            local_map = model.get_local_attention_map(current_image)
                            dualstream_map = model.get_final_attention_map(current_image)
                            
                            # Save attention maps and compute metrics for each mode
                            for attn_mode, attn_map in [
                                ("Global", global_map),
                                ("Local", local_map),
                                ("Dual-stream", dualstream_map)
                            ]:
                                save_path = os.path.join(save_dir, f"{image_names[idx]}_{attn_mode}_attention_{today}.png")
                                attention_mask = model.plot_attention_on_image(
                                    current_image[0], attn_map, save_path=save_path)
                                
                                # Compute metrics if masks are available
                                if current_mask is not None:
                                    try:
                                        metrics = XAIMetric(attention_mask, current_mask)
                                        rma = metrics.RMA()
                                        rra = metrics.RRA()
                                        iou = metrics.IoU()
                                        
                                        # Store metrics in appropriate dictionary
                                        metric_tuple = (rma, rra, iou)
                                        if attn_mode == "Global":
                                            global_metrics[image_names[idx]] = metric_tuple
                                        elif attn_mode == "Local":
                                            local_metrics[image_names[idx]] = metric_tuple
                                        else:  # Dual-stream
                                            dualstream_metrics[image_names[idx]] = metric_tuple
                                            
                                        # Update progress bar description
                                        pbar.set_description(f"Processing {mode} attention maps - {image_names[idx]}")
                                    except Exception as e:
                                        print(f"Warning: Failed to compute metrics for {image_names[idx]} ({attn_mode}): {str(e)}")
                        
                        else:  # Single mode processing
                            # Get attention map for specified mode
                            if mode == "Global":
                                attention_map = model.get_global_attention_map(current_image)
                            elif mode == "Local":
                                attention_map = model.get_local_attention_map(current_image)
                            else:
                                attention_map = model.get_final_attention_map(current_image)
                            
                            # Save attention map
                            save_path = os.path.join(save_dir, f"{image_names[idx]}_attention_{mode}_{today}.png")
                            attention_mask = model.plot_attention_on_image(
                                current_image[0], attention_map, save_path=save_path)
                            
                            # Compute metrics if masks are available
                            if current_mask is not None:
                                try:
                                    metrics = XAIMetric(attention_mask, current_mask)
                                    rma = metrics.RMA()
                                    rra = metrics.RRA()
                                    iou = metrics.IoU()
                                    metrics_dict[image_names[idx]] = (rma, rra, iou)
                                    
                                    # Update progress bar description
                                    pbar.set_description(f"Processing {mode} attention maps - {image_names[idx]}")
                                except Exception as e:
                                    print(f"Warning: Failed to compute metrics for {image_names[idx]}: {str(e)}")

                    except Exception as e:
                        print(f"Error processing image {image_names[idx]}: {str(e)}")
                        continue

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {str(e)}")
                continue

    # Save and visualize metrics
    if mode == "Dual-stream" and has_masks:
        # Compare all attention modes
        XAIMetric.compare_attention_modes(
            global_metrics, local_metrics, dualstream_metrics,
            save_dir, today
        )
        
        # Save individual mode metrics and create distributions
        for mode_metrics, mode_name in [
            (global_metrics, "Global"),
            (local_metrics, "Local"),
            (dualstream_metrics, "Dual-stream")
        ]:
            metrics_df = pd.DataFrame.from_dict(mode_metrics, orient='index', columns=['RMA', 'RRA', 'IoU'])
            metrics_df.to_csv(os.path.join(save_dir, f"{mode_name}_metrics_{today}.csv"))
            XAIMetric.plot_metrics_distribution(metrics_df, save_dir, mode_name, today)
            
            # Print summary statistics
            stats = XAIMetric.aggregate_metrics(mode_metrics)
            print(f"\n{mode_name} Attention Statistics:")
            print(f"RMA: {stats['RMA_mean']:.3f} ± {stats['RMA_std']:.3f} (95% CI: [{stats['RMA_ci_lower']:.3f}, {stats['RMA_ci_upper']:.3f}])")
            print(f"RRA: {stats['RRA_mean']:.3f} ± {stats['RRA_std']:.3f} (95% CI: [{stats['RRA_ci_lower']:.3f}, {stats['RRA_ci_upper']:.3f}])")
            print(f"IoU: {stats['IoU_mean']:.3f} ± {stats['IoU_std']:.3f} (95% CI: [{stats['IoU_ci_lower']:.3f}, {stats['IoU_ci_upper']:.3f}])")
            print(f"Valid cases: {stats['valid_cases']}/{stats['total_cases']} ({stats['normal_cases']} normal cases)")
            
    elif has_masks:  # Single mode
        # Save metrics to CSV
        metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index', columns=['RMA', 'RRA', 'IoU'])
        metrics_df.to_csv(os.path.join(save_dir, f"{mode}_metrics_{today}.csv"))
        
        # Plot distribution
        XAIMetric.plot_metrics_distribution(metrics_df, save_dir, mode, today)
        
        # Calculate and save aggregate statistics
        stats = XAIMetric.aggregate_metrics(metrics_dict)
        stats_df = pd.DataFrame([stats])
        stats_df.to_csv(os.path.join(save_dir, f"{mode}_aggregate_metrics_{today}.csv"), index=False)
        
        # Print summary statistics
        print(f"\n{mode} Attention Statistics:")
        print(f"RMA: {stats['RMA_mean']:.3f} ± {stats['RMA_std']:.3f} (95% CI: [{stats['RMA_ci_lower']:.3f}, {stats['RMA_ci_upper']:.3f}])")
        print(f"RRA: {stats['RRA_mean']:.3f} ± {stats['RRA_std']:.3f} (95% CI: [{stats['RRA_ci_lower']:.3f}, {stats['RRA_ci_upper']:.3f}])")
        print(f"IoU: {stats['IoU_mean']:.3f} ± {stats['IoU_std']:.3f} (95% CI: [{stats['IoU_ci_lower']:.3f}, {stats['IoU_ci_upper']:.3f}])")
        print(f"Valid cases: {stats['valid_cases']}/{stats['total_cases']} ({stats['normal_cases']} normal cases)")

    print(f"\n{mode} attention maps and metrics saved successfully in {save_dir}")