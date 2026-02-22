import os
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torchvision.transforms import ToTensor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from models.xai_metric import XAIMetric
from shap_explanation import SHAPExplainer
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_degradation_score(morf_curve, lerf_curve):
    """Calculate the Degradation Score (DS) between MoRF and LeRF curves."""
    if len(morf_curve) == 0 or len(lerf_curve) == 0:
        return 0.0
    
    # Convert to numpy arrays if they're lists
    morf_array = np.array(morf_curve) if isinstance(morf_curve, list) else morf_curve
    lerf_array = np.array(lerf_curve) if isinstance(lerf_curve, list) else lerf_curve
    
    return np.mean(lerf_array - morf_array)

def calculate_class_penalty(class_ds_scores):
    """Calculate the penalty term for class-adjusted degradation score."""
    n = len(class_ds_scores)
    if n < 2:
        return 0.0
    total_diff = 0
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            total_diff += abs(class_ds_scores[i] - class_ds_scores[j])
            count += 1
    return (total_diff / (2 * count)) if count > 0 else 0.0

def calculate_class_adjusted_ds(ds, class_penalty, alpha=1.0):
    """Calculate the class-adjusted degradation score DSc."""
    return ds - alpha * class_penalty

def incremental_deletion_analysis(model, correctly_classified_images, device, fold_save_dir, class_names,
                                  methods=['global','local','gate','concat','product'],
                                  heatmap_types=['att', 'shap'],
                                  shap_explainer=None, num_steps=10, val_metrics_all_folds=None):
    
    # Method name mapping for plotting
    def get_display_name(method_key):
        method_mapping = {
            'global_att': 'Global',
            'global_shap': 'Global_SHAP',
            'local_att': 'Local', 
            'local_shap': 'Local_SHAP',
            'gate_att': 'Fusion_Gate',
            'gate_shap': 'Fusion_Gate_SHAP',
            'concat_att': 'Fusion_Concat',
            'concat_shap': 'Fusion_Concat_SHAP',
            'product_att': 'Fusion_Product',
            'product_shap': 'Fusion_Product_SHAP'
        }
        return method_mapping.get(method_key, method_key)

    model.eval()

    os.makedirs(fold_save_dir, exist_ok=True)

    images, labels, image_names, masks = [], [], [], []
    
    for img in correctly_classified_images:
        images.append(img['image'])
        labels.append(img['label'])
        image_names.append(img['name'])
        masks.append(img['mask'])

    summary = []
    # Add storage for MoRF/LeRF curves for each method/heatmap/class
    all_curves = {}
    # Add storage for per-image MoRF/LeRF curves
    per_image_curves = {}
    for method in methods:
        for hm_type in heatmap_types:
            method_key = f"{method}_{hm_type}"
            all_curves[method_key] = {c: {'morf': [], 'lerf': []} for c in range(len(class_names))}
            per_image_curves[method_key] = {img_name: {'morf': [], 'lerf': []} for img_name in image_names}

    for method in methods:
        for hm_type in heatmap_types:
            method_key = f"{method}_{hm_type}"
            base_dir = os.path.join(fold_save_dir, method_key)
            os.makedirs(base_dir, exist_ok=True)

            # Storage for each image's explanation
            expl_data = {}

            folder_step_0 = os.path.join(base_dir, "Step_0")
            os.makedirs(folder_step_0, exist_ok=True)

            # Initialize SHAP explainer if needed (will be used later in the loop)
            shap_explainer = None
            if hm_type == 'shap':
                # Process images one by one to avoid memory issues
                # Convert class_names list to dict for SHAPExplainer
                class_names_dict = {i: name for i, name in enumerate(class_names)}
                shap_explainer = SHAPExplainer(model, device, save_dir=folder_step_0, class_names=class_names_dict)
                
                # Process images individually instead of batching
                method_results = []
                for i, (img, img_name, mask) in enumerate(zip(images, image_names, masks)):
                    # Process single image
                    sv, shap_map, metrics = shap_explainer.explain_image(img, method, mask, img_name, save_dir=folder_step_0, max_evals=100, batch_size=50)                    
                    method_results.append((sv, shap_map, metrics))
                    
                    # Clear GPU memory after each image
                    torch.cuda.empty_cache()
                
                # Store results in expl_data
                for i, (sv, shap_map, _) in enumerate(method_results):
                    idxs = np.argsort(-shap_map.flatten())
                    expl_data[image_names[i]] = {
                        'heatmap':    shap_map,
                        'shap_values': sv,              # Store SHAP values for visualization
                        'sorted_idx': idxs,
                        'H':          shap_map.shape[0],
                        'W':          shap_map.shape[1],
                        'label':      labels[i],
                        'mask':       masks[i],
                        'orig_img':   images[i]
                    }

            else:
                core = model.module if isinstance(model, torch.nn.DataParallel) else model

                # Build expl_data for every image
                for i, (img, lbl, mask, name) in enumerate(zip(images, labels, masks, image_names)):
                    with torch.no_grad():
                        raw_att = (
                            core.get_global_attention_map(img.unsqueeze(0).to(device))
                            if method=='global' else
                            core.get_local_attention_map(img.unsqueeze(0).to(device))
                            if method=='local' else
                            core.evaluate_all_fusion_types(img.unsqueeze(0).to(device))[method]['attention']
                        )
                        att = raw_att.detach().cpu()
                        if att.dim()==4:    att = att[0]
                        if att.dim()==3:    att = att.mean(dim=0)
                        att_np = att.numpy()
                        H, W = img.shape[1], img.shape[2]
                        hm = cv2.resize(att_np, (W, H), interpolation=cv2.INTER_LINEAR)
                        hm = (hm - hm.min())/(hm.max()-hm.min()+1e-8)
                        idxs = np.argsort(-hm.flatten())
                        expl_data[name] = {
                            'heatmap':    hm,
                            'raw_att':    raw_att,
                            'sorted_idx': idxs,
                            'H':          H,
                            'W':          W,
                            'label':      lbl,
                            'mask':       mask,
                            'orig_img':   img
                        }
                    
                    # Clear GPU memory after each image
                    torch.cuda.empty_cache()

                metrics = []
                for img_name, d in expl_data.items():
                    hm   = d['heatmap']
                    gt_m = d['mask']
                    
                    # Handle both cases: when mask exists and when it doesn't
                    if gt_m is not None:
                        gt = gt_m.squeeze().cpu().numpy()
                        try:
                            m  = XAIMetric(heatmap=hm, ground_truth_mask=gt)
                            rma = m.RMA()
                            rra = m.RRA()
                            iou = m.IoU()
                            pg = m.PointingGame()
                        except Exception as e:
                            rma = rra = iou = pg = np.nan
                    else:
                        rma = rra = iou = pg = np.nan
                    
                    metrics.append({
                        "Image":       img_name,
                        "RMA":          rma,
                        "RRA":          rra,
                        "IoU":          iou,
                        "PointingGame": pg
                    })

                if metrics:
                    df = pd.DataFrame(metrics)
                    df.loc["Average"] = df.drop("Image", axis=1).mean(numeric_only=True).round(4)
                    df.at["Average", "Image"] = "Average"
                    out_csv = os.path.join(folder_step_0, f"ATT_XAI_Metrics [{method}].csv")
                    df.to_csv(out_csv, index=False)

            # Incremental deletion loop
            for step in range(0, num_steps + 1):
                frac = step / num_steps
                step_dir = os.path.join(base_dir, f"Step_{int(frac*100)}")
                os.makedirs(step_dir, exist_ok=True)
                
                y_true, y_pred = [], []
                xai_vals = {'RMA': [], 'RRA': [], 'IoU': [], 'PointingGame': []}
                records = []
                # For MoRF/LeRF curves  
                morf_curve = []
                lerf_curve = []
                class_curves = {c: {'morf': [], 'lerf': []} for c in range(len(class_names))}
                for i, (img, lbl, mask, img_name) in enumerate(zip(images, labels, masks, image_names)):
                    
                    img_tensor = img.unsqueeze(0).to(device)
                    # Robustly get true_label as class index
                    if hasattr(lbl, 'item'):
                        true_label = int(lbl.item())
                    elif isinstance(lbl, str):
                        true_label = class_names.index(lbl)
                    else:
                        true_label = int(lbl)
                    # Use pre-computed heatmap from expl_data for both attention and SHAP
                    att_resized = expl_data[img_name]['heatmap']

                    # MoRF
                    flat = att_resized.flatten()
                    idxs = np.argsort(-flat)                    # MoRF: most relevant first
                    idxs_lerf = np.argsort(flat)                # LeRF: least relevant first

                    # LeRF
                    mask_flat = np.zeros_like(flat)
                    mask_flat_lerf = np.zeros_like(flat)
                    mask_flat[idxs[:int(frac * len(flat))]] = 1
                    mask_flat_lerf[idxs_lerf[:int(frac * len(flat))]] = 1
                    mask2 = mask_flat.reshape(att_resized.shape)
                    mask2_lerf = mask_flat_lerf.reshape(att_resized.shape)

                    orig = img_tensor[0].permute(1, 2, 0).cpu().numpy()
                    # Gaussian-blur the original for perturbation
                    orig_uint8   = (orig * 255).astype(np.uint8)
                    blurred_uint8 = cv2.GaussianBlur(orig_uint8, (31, 31), sigmaX=0)
                    blurred = blurred_uint8.astype(np.float32) / 255.0

                    out_img_morf = np.where(mask2[..., None], blurred, orig)
                    out_img_morf = np.clip(out_img_morf, 0, 1)

                    out_img_lerf = np.where(mask2_lerf[..., None], blurred, orig)
                    out_img_lerf = np.clip(out_img_lerf, 0, 1)

                    # Generate overlay for all steps using the original heatmap from step 0
                    if step == 0 and hm_type == 'shap':
                        # For SHAP in step 0, use the original SHAP visualization
                        overlay_path = os.path.join(step_dir, f"{os.path.basename(img_name)}_SHAP.png")
                        # Use the original image and stored SHAP values for step 0
                        shap_explainer.visualize_shap(img, expl_data[img_name]['shap_values'], method, save_path=overlay_path)
                    else:
                        # Convert heatmap to tensor and normalize (ensure on CPU for memory efficiency)
                        heat_tensor = torch.from_numpy(att_resized).float().cpu()
                        # Enhance contrast for better visibility
                        heat_tensor = torch.pow(heat_tensor, 0.5)  # Gamma correction to brighten mid-tones
                        heat_tensor = (heat_tensor - heat_tensor.min()) / (heat_tensor.max() - heat_tensor.min() + 1e-8)
                        
                        # Use OpenCV for colormap but keep tensor operations for blending
                        heat_uint8 = (heat_tensor.numpy() * 255).astype(np.uint8)
                        heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
                        heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
                        heat_rgb = torch.from_numpy(heat_color).float() / 255.0  # Convert back to tensor
                        
                        # For step 0, use original image tensor; for step > 0, use perturbed image
                        if step == 0:
                            base_img_tensor = img_tensor[0].permute(1, 2, 0).cpu()  # (H, W, C) - move to CPU
                        else:
                            base_img_tensor = torch.from_numpy(out_img_morf).float().cpu()
                        
                        # Blend using tensor operations (all tensors now on CPU)
                        overlay_tensor = 0.3 * base_img_tensor + 0.7 * heat_rgb
                        overlay_tensor = torch.clamp(overlay_tensor, 0, 1)
                        
                        # Convert to PIL and save
                        overlay_np = (overlay_tensor.cpu().numpy() * 255).astype(np.uint8)
                        overlay_path = os.path.join(step_dir, f"{os.path.basename(img_name)}_overlay.png")
                        Image.fromarray(overlay_np).save(overlay_path)
                        
                        # Free memory
                        del overlay_tensor, overlay_np, heat_tensor, heat_rgb, base_img_tensor
                        torch.cuda.empty_cache()

                    inp = ToTensor()(out_img_morf).unsqueeze(0).to(device)
                    inp_lerf = ToTensor()(out_img_lerf).unsqueeze(0).to(device)

                    with torch.no_grad():
                        # Get the core model if it's wrapped in DataParallel
                        core_model = model.module if isinstance(model, torch.nn.DataParallel) else model
                        res = core_model.evaluate_all_fusion_types(inp)
                        res_lerf = core_model.evaluate_all_fusion_types(inp_lerf)
                        if method == 'global':
                            out = res['global']['output']
                            out_lerf = res_lerf['global']['output']
                        elif method == 'local':
                            out = res['local']['output']
                            out_lerf = res_lerf['local']['output']
                        else:
                            out = res[method]['output']
                            out_lerf = res_lerf[method]['output']
                    probs = torch.softmax(out, dim=1).squeeze().cpu().numpy()
                    probs_lerf = torch.softmax(out_lerf, dim=1).squeeze().cpu().numpy()
                    pred = int(np.argmax(probs))
                    y_true.append(true_label)
                    y_pred.append(pred)
                    # Store MoRF/LeRF curves for true class
                    morf_curve.append(probs[true_label])
                    lerf_curve.append(probs_lerf[true_label])
                    class_curves[true_label]['morf'].append(probs[true_label])
                    class_curves[true_label]['lerf'].append(probs_lerf[true_label])
                    
                    # Store per-image MoRF/LeRF curves for this step
                    per_image_curves[method_key][img_name]['morf'].append(probs[true_label])
                    per_image_curves[method_key][img_name]['lerf'].append(probs_lerf[true_label])

                    if mask is not None:
                        gt = mask.squeeze().cpu().numpy()
                        gt_res = np.array(Image.fromarray((gt * 255).astype(np.uint8)).resize(att_resized.shape[::-1], Image.NEAREST)) / 255.0
                        metric = XAIMetric(heatmap=att_resized, ground_truth_mask=gt_res)
                        mvals = [metric.RMA(), metric.RRA(), metric.IoU(), metric.PointingGame()]
                        for k, v in zip(xai_vals, mvals):
                            xai_vals[k].append(v)
                    else:
                        mvals = [np.nan] * 4
                        for k in xai_vals:
                            xai_vals[k].append(np.nan)

                    rec = {
                        "Image": img_name,
                        "Fusion": method,
                        "Step": int(frac * 100),
                        "GT": class_names[true_label],
                        "Pred": class_names[pred],
                    }
                    if mask is not None:
                        rec.update({
                            "RMA": xai_vals['RMA'][-1],
                            "RRA": xai_vals['RRA'][-1],
                            "IoU": xai_vals['IoU'][-1],
                            "PointingGame": xai_vals['PointingGame'][-1]
                        })
                    for ci, cname in enumerate(class_names):
                        rec[f"Prob_{cname}"] = float(probs[ci])
                    records.append(rec)
                    
                    # Clear GPU memory after processing each image
                    del inp, inp_lerf, out, out_lerf, probs, probs_lerf
                    torch.cuda.empty_cache()

                # Average MoRF/LeRF curves for this step (over all images)
                avg_morf_curve = np.mean(morf_curve) if morf_curve else 0.0
                avg_lerf_curve = np.mean(lerf_curve) if lerf_curve else 0.0
                # Store for later DS/DS_c calculation
                all_curves[method_key][0]['morf'].append(avg_morf_curve)  # 0: all classes
                all_curves[method_key][0]['lerf'].append(avg_lerf_curve)

                # For each class, store average for this step
                for c in range(len(class_names)):
                    if class_curves[c]['morf']:
                        all_curves[method_key][c]['morf'].append(np.mean(class_curves[c]['morf']))
                        all_curves[method_key][c]['lerf'].append(np.mean(class_curves[c]['lerf']))
                    else:
                        all_curves[method_key][c]['morf'].append(0.0)
                        all_curves[method_key][c]['lerf'].append(0.0)

                step_summary = {
                    'Method': method_key,
                    'Step': int(frac * 100),
                    'Accuracy': accuracy_score(y_true, y_pred),
                    'Precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
                    'Recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
                    'F1_Score': f1_score(y_true, y_pred, average='macro', zero_division=0),
                }
                if mask is not None:
                    step_summary.update({f"Mean_{k}": np.nanmean(v) for k, v in xai_vals.items()})
                summary.append(step_summary)
                
                # Clear memory after each step
                torch.cuda.empty_cache()

    # After all steps, compute DS, DS_c for each method/heatmap
    ds_summary = []
    for method in methods:
        for hm_type in heatmap_types:
            method_key = f"{method}_{hm_type}"
            # Get validation accuracy for this method if available
            val_acc = None
            if val_metrics_all_folds is not None:
                try:
                    if method in ['global', 'local']:
                        val_accs = [fold_metrics[method.capitalize()]['Accuracy'] for fold_metrics in val_metrics_all_folds]
                    else:
                        val_accs = [fold_metrics[f"Fusion_{method.capitalize()}"]['Accuracy'] for fold_metrics in val_metrics_all_folds]
                    val_acc = np.mean(val_accs)
                except Exception:
                    pass

            # Create list to store per-image statistics using stored curves
            per_image_stats = []
            
            # Process each image using stored per-image curves from the first loop
            for _, (img, lbl, mask, img_name) in enumerate(zip(images, labels, masks, image_names)):
                # Get true label
                if hasattr(lbl, 'item'):
                    true_label = int(lbl.item())
                elif isinstance(lbl, str):
                    true_label = class_names.index(lbl)
                else:
                    true_label = int(lbl)
                
                # Get prediction from step 0
                img_tensor = img.unsqueeze(0).to(device)
                with torch.no_grad():
                    core_model = model.module if isinstance(model, torch.nn.DataParallel) else model
                    res = core_model.evaluate_all_fusion_types(img_tensor)
                    if method == 'global':
                        out = res['global']['output']
                    elif method == 'local':
                        out = res['local']['output']
                    else:
                        out = res[method]['output']
                probs = torch.softmax(out, dim=1).squeeze().cpu().numpy()
                pred = int(np.argmax(probs))
                
                # Get stored per-image MoRF/LeRF curves
                img_morf_curve = per_image_curves[method_key][img_name]['morf']
                img_lerf_curve = per_image_curves[method_key][img_name]['lerf']
                
                # Set step 0 to validation accuracy if available
                if val_acc is not None and len(img_morf_curve) > 0:
                    img_morf_curve[0] = val_acc
                    img_lerf_curve[0] = val_acc
                
                # Calculate per-image DS
                img_ds = calculate_degradation_score(img_morf_curve, img_lerf_curve)
                
                # Calculate class-specific DS scores (using overall class curves)
                class_ds_scores = []
                for c in range(len(class_names)):
                    if c in all_curves[method_key] and all_curves[method_key][c]['morf'] and all_curves[method_key][c]['lerf']:
                        class_morf = np.array(all_curves[method_key][c]['morf'])
                        class_lerf = np.array(all_curves[method_key][c]['lerf'])
                        # Set step 0 to validation accuracy if available
                        if val_acc is not None:
                            class_morf[0] = val_acc
                            class_lerf[0] = val_acc
                        class_ds = calculate_degradation_score(class_morf, class_lerf)
                        class_ds_scores.append(class_ds)
                    else:
                        class_ds_scores.append(0.0)
                
                # Add to per-image statistics
                stats_dict = {
                    'Image_Name': os.path.basename(img_name),
                    'Method': method_key,
                    'Ground_Truth': class_names[true_label],
                    'Prediction': class_names[pred],
                    'Mean_MoRF': np.mean(img_morf_curve) if img_morf_curve else 0.0,
                    'Mean_LeRF': np.mean(img_lerf_curve) if img_lerf_curve else 0.0,
                    'Mean_DS': img_ds
                }
                # Add class-specific DS scores
                for c, class_name in enumerate(class_names):
                    stats_dict[f'DS_{class_name}'] = class_ds_scores[c]
                per_image_stats.append(stats_dict)
            
            # Save per-image statistics to CSV
            per_image_df = pd.DataFrame(per_image_stats)
            per_image_df.to_csv(os.path.join(fold_save_dir, f'{method_key}_per_image_stats.csv'), index=False)
            
            # Continue with existing DS calculation
            morf_curve = np.array(all_curves[method_key][0]['morf'])
            lerf_curve = np.array(all_curves[method_key][0]['lerf'])
            
            # Set step 0 to validation accuracy if available
            if val_acc is not None:
                morf_curve[0] = val_acc
                lerf_curve[0] = val_acc
            
            ds = calculate_degradation_score(morf_curve, lerf_curve)
            # Class-specific DS
            class_ds_scores = []
            for c in range(len(class_names)):
                class_morf = np.array(all_curves[method_key][c]['morf'])
                class_lerf = np.array(all_curves[method_key][c]['lerf'])
                # Set step 0 to validation accuracy for class curves too
                if val_acc is not None:
                    class_morf[0] = val_acc
                    class_lerf[0] = val_acc
                class_ds = calculate_degradation_score(class_morf, class_lerf)
                class_ds_scores.append(class_ds)
            class_penalty = calculate_class_penalty(class_ds_scores)
            dsc = calculate_class_adjusted_ds(ds, class_penalty, alpha=1.0)
            ds_summary.append({
                'Method': method_key,
                'DS': ds,
                'Class_Penalty': class_penalty,
                'Class_Adjusted_DS': dsc
            })
            # Save full MoRF and LeRF curves for debugging
            morf_curve = np.array(all_curves[method_key][0]['morf'][:num_steps+1])
            lerf_curve = np.array(all_curves[method_key][0]['lerf'][:num_steps+1])
            if len(morf_curve) != num_steps+1 or len(lerf_curve) != num_steps+1:
                print(f"[WARNING] {method_key}: Expected {num_steps+1} steps, got MoRF={len(morf_curve)}, LeRF={len(lerf_curve)}. Truncating.")
            steps = [int(100 * step / num_steps) for step in range(num_steps + 1)]
            min_len = min(len(steps), len(morf_curve), len(lerf_curve))
            morf_lerf_df = pd.DataFrame({
                'Step': steps[:min_len],
                'MoRF': morf_curve[:min_len],
                'LeRF': lerf_curve[:min_len]
            })
            morf_lerf_df.to_csv(os.path.join(fold_save_dir, f'{method_key}_morf_lerf_curves.csv'), index=False)

    ds_df = pd.DataFrame(ds_summary)
    ds_df.to_csv(os.path.join(fold_save_dir, 'degradation_scores_summary.csv'), index=False)

    # Plot MoRF and LeRF curves for each method
    for method in methods:
        for hm_type in heatmap_types:
            method_key = f"{method}_{hm_type}"
            # Load the saved curve (or use morf_curve/lerf_curve directly)
            morf_lerf_path = os.path.join(fold_save_dir, f'{method_key}_morf_lerf_curves.csv')
            if not os.path.exists(morf_lerf_path):
                continue
            morf_lerf_df = pd.read_csv(morf_lerf_path)
            # Set step 0 to validation accuracy if available
            if val_metrics_all_folds is not None:
                try:
                    # Get the correct method name for validation metrics
                    if method in ['global', 'local']:
                        method_cap = method.capitalize()
                    else:
                        method_cap = f"Fusion_{method.capitalize()}"
                    
                    if all(method_cap in fold_metrics for fold_metrics in val_metrics_all_folds):
                        val_accs = [fold_metrics[method_cap]['Accuracy'] for fold_metrics in val_metrics_all_folds]
                        val_acc = np.mean(val_accs)
                        morf_lerf_df.loc[0, 'MoRF'] = val_acc
                        morf_lerf_df.loc[0, 'LeRF'] = val_acc
                        # Save the updated curves
                        morf_lerf_df.to_csv(morf_lerf_path, index=False)
                    else:
                        print(f"Warning: {method_cap} not found in validation metrics")
                except Exception as e:
                    print(f"Warning: Could not set validation accuracy for {method_key}: {str(e)}")
            plt.figure(figsize=(10, 6))
            plt.plot(morf_lerf_df['Step'], morf_lerf_df['MoRF'], label='MoRF', marker='o', linewidth=3)
            plt.plot(morf_lerf_df['Step'], morf_lerf_df['LeRF'], label='LeRF', marker='s', linewidth=3)
            plt.xlabel('Percentage of Blurred Features (%)', fontsize=20, fontweight='bold', labelpad=15)
            plt.ylabel('Target Class Probability', fontsize=20, fontweight='bold', labelpad=15)
            plt.title(f'MoRF vs LeRF Curve: {method_key}', fontsize=24, fontweight='bold', pad=20)
            plt.xticks(np.arange(0, 101, 10), fontsize=16, fontweight='bold')
            plt.yticks(fontsize=16, fontweight='bold')
            legend = plt.legend(fontsize=18, title_fontsize=20)
            # Make legend text bold
            for text in legend.get_texts():
                text.set_fontweight('bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(fold_save_dir, f'{method_key}_morf_lerf_curve.png'), dpi=300)
            plt.close()

    # Collect and plot MoRF/LeRF curves from the summary data
    def plot_all_methods_curves(fold_save_dir, methods, heatmap_types, curve_type='morf'):
        """
        Plot MoRF or LeRF curves for all methods by reading the per-method CSV
        named <method_key>_morf_lerf_curves.csv sitting in fold_save_dir.
        """
        plt.figure(figsize=(12, 8))
        keys      = [f"{m}_{h}" for m in methods for h in heatmap_types]
        palette   = sns.color_palette("husl", n_colors=len(keys))
        color_map = dict(zip(keys, palette))

        for key in keys:
            csv_path = os.path.join(fold_save_dir, f"{key}_morf_lerf_curves.csv")
            if not os.path.exists(csv_path):
                print(f"Missing curve file for {key}, skipping.")
                continue

            df = pd.read_csv(csv_path)
            y  = df['MoRF'] if curve_type=='morf' else df['LeRF']
            display_name = get_display_name(key)
            plt.plot(df['Step'], y,
                    label=display_name,
                    marker='o', linewidth=3,
                    color=color_map[key])

        # match the single-method style
        plt.xlabel('Percentage of Perturbed Features (%)', fontsize=20, fontweight='bold', labelpad=15)
        plt.ylabel('Target Class Probability',        fontsize=20, fontweight='bold', labelpad=15)
        plt.title(f"{curve_type.upper()} Curves for All Methods", fontsize=24, fontweight='bold', pad=20)
        plt.xticks(np.arange(0, 101, 10), fontsize=16, fontweight='bold')
        plt.yticks(fontsize=16, fontweight='bold')
        legend = plt.legend(loc='best', fontsize=18, title_fontsize=20, framealpha=0.9)
        for text in legend.get_texts():
            text.set_fontweight('bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()

        out_path = os.path.join(fold_save_dir, f"all_methods_{curve_type}_curves.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {out_path}")

    plot_all_methods_curves(fold_save_dir, methods, heatmap_types, 'morf')
    plot_all_methods_curves(fold_save_dir, methods, heatmap_types, 'lerf')

    # Create violin plots for DS distributions
    all_ds_data = []
    all_class_ds_data = []
    
    for method in methods:
        for hm_type in heatmap_types:
            method_key = f"{method}_{hm_type}"
            stats_path = os.path.join(fold_save_dir, f'{method_key}_per_image_stats.csv')
            if not os.path.exists(stats_path):
                continue
            
            # Read the stats file
            df = pd.read_csv(stats_path)
            
            # Add overall DS data
            for _, row in df.iterrows():
                all_ds_data.append({
                    'Method': method_key,
                    'DS': row['Mean_DS']
                })
            
            # Add class-specific DS data
            for class_name in class_names:
                ds_col = f'DS_{class_name}'
                if ds_col in df.columns:
                    for _, row in df.iterrows():
                        all_class_ds_data.append({
                            'Method': method_key,
                            'Class': class_name,
                            'DS': row[ds_col]
                        })
    
    # Convert to DataFrames
    ds_df = pd.DataFrame(all_ds_data)
    class_ds_df = pd.DataFrame(all_class_ds_data)
    
    # Plot overall DS violin plot
    plt.figure(figsize=(15, 8))
    
    # Convert method names to display names
    ds_df['Display_Method'] = ds_df['Method'].apply(get_display_name)
    
    # Create a custom color palette
    method_colors = sns.color_palette("husl", n_colors=len(ds_df['Display_Method'].unique()))
    method_color_dict = dict(zip(ds_df['Display_Method'].unique(), method_colors))
    
    # Create violin plot with boxplot inside
    ax = sns.violinplot(data=ds_df, x='Display_Method', y='DS', 
                       hue='Display_Method',  # Use hue instead of palette
                       inner='box',  # Add boxplot inside violin
                       palette=method_color_dict,
                       legend=False)  # Hide legend since x-axis already shows methods
    
    # Customize the plot
    plt.xticks(rotation=45, ha='right', fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    plt.title('Distribution of Degradation Scores Across Methods', 
             fontsize=24, fontweight='bold', pad=20)
    plt.xlabel('Method', fontsize=20, fontweight='bold', labelpad=15)
    plt.ylabel('Degradation Score', fontsize=20, fontweight='bold', labelpad=15)
    
    # Add grid for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(fold_save_dir, 'ds_distribution_violin.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot class-specific DS violin plot
    plt.figure(figsize=(16, 10))  # Reduced width since legend is inside
    
    # Convert method names to display names
    class_ds_df['Display_Method'] = class_ds_df['Method'].apply(get_display_name)
    
    # Create violin plot with boxplot inside
    ax = sns.violinplot(data=class_ds_df, x='Display_Method', y='DS', 
                       hue='Class', inner='box',    # Add boxplot inside violin
                       palette='husl')              # Use husl color palette for better distinction
    
    # Customize the plot
    plt.xticks(rotation=45, ha='right', fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    plt.title('Distribution of Class-Specific Degradation Scores', 
             fontsize=24, fontweight='bold', pad=20)
    plt.xlabel('Method', fontsize=20, fontweight='bold', labelpad=15)
    plt.ylabel('Degradation Score', fontsize=20, fontweight='bold', labelpad=15)
    
    # Customize legend
    legend = plt.legend(title='Class', loc='upper right', fontsize=16, title_fontsize=16, 
              framealpha=0.9)
    # Make legend text bold
    for text in legend.get_texts():
        text.set_fontweight('bold')
    
    # Add grid for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(fold_save_dir, 'class_ds_distribution_violin.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()