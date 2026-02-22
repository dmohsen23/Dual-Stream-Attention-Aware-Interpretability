import torch
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import os
import pandas as pd
from models.xai_metric import XAIMetric
import warnings
warnings.filterwarnings("ignore", message=".*FigureCanvasAgg is non-interactive.*")

class SHAPExplainer:
    def __init__(self, model, device, save_dir=None, class_names=None):
        """
        model      : your dualstream attention model
        device     : torch device
        save_dir   : directory to dump all SHAP PNGs and CSVs
        class_names: dict mapping class indices → name
        """
        self.model = model
        self.device = device
        self.model.eval()
        core = model.module if isinstance(model, torch.nn.DataParallel) else model
        self.num_cls = core.num_cls
        # Default class names if none provided
        if class_names is None:
            self.class_names = {i: f"Class_{i}" for i in range(self.num_cls)}
        else:
            self.class_names = class_names

        # Single flat folder for all outputs
        self.base_save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def preprocess_image(self, image):
        # Convert tensor to numpy and handle dimensions
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if image.ndim == 4:
            image = image[0]
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        return image

    def model_predict(self, x, method):
        # SHAP callback: x as numpy NHWC or HWC
        x_tensor = torch.from_numpy(x).float()
        if x_tensor.ndim == 4:
            # (B,H,W,C) → (B,C,H,W)
            x_tensor = x_tensor.permute(0, 3, 1, 2)
        else:
            # (H,W,C) → (1,C,H,W)
            x_tensor = x_tensor.permute(2, 0, 1).unsqueeze(0)
        x_tensor = x_tensor.to(self.device)
        with torch.no_grad():
            core = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
            res = core.evaluate_all_fusion_types(x_tensor)
        # Pick the right branch or fusion
        out = res[method]['output']
        probs = torch.softmax(out, dim=1)
        return probs.cpu().numpy()

    def calculate_xai_metrics(self, heatmap, gt_mask):
        metric = XAIMetric(heatmap=heatmap, ground_truth_mask=gt_mask)
        return metric.RMA(), metric.RRA(), metric.IoU(), metric.PointingGame()

    def explain_image(self, image, method, gt_mask, image_name,
                      save_dir=None, max_evals=100, batch_size=50):
        """
        Returns:
          - raw_explanation: a shap.Explanation object
          - heatmap: 2D numpy normalized map for metrics
          - (rma, rra, iou, pg) tuple
        """
        img = self.preprocess_image(image)
        masker = shap.maskers.Image("inpaint_telea", img.shape)
        explainer = shap.Explainer(lambda x: self.model_predict(x, method), masker)

        sv: shap.Explanation = explainer(np.expand_dims(img, 0), max_evals=max_evals, batch_size=batch_size, outputs=shap.Explanation.argsort.flip[: self.num_cls])

        # Build heatmap for metrics:
        arr = sv.values[0]                   # Shape (H, W, 3, C)
        # pick class=1 channel (or you can choose true class index)
        heat = arr[..., 1].mean(axis=-1)     # Average over RGB
        heat[heat < 0] = 0
        tot = heat.sum()
        if tot > 0:
            heat /= tot
        # Compute XAI metrics if mask provided
        if gt_mask is not None:
            if isinstance(gt_mask, torch.Tensor):
                gm = gt_mask.cpu().numpy().squeeze()
            else:
                gm = np.array(gt_mask).squeeze()
            try:
                rma, rra, iou, pg = self.calculate_xai_metrics(heat, gm)
            except Exception:
                rma = rra = iou = pg = np.nan
        else:
            rma = rra = iou = pg = np.nan

        # Save results to CSV
        if save_dir:
            csv_path = os.path.join(save_dir, f"SHAP_XAI_Metrics [{method}].csv")
            
            # Create new row data
            new_row = {"Image": image_name, "RMA": rma, "RRA": rra, "IoU": iou, "PointingGame": pg}
            
            # Check if CSV already exists
            if os.path.exists(csv_path):
                # Read existing CSV and append new row
                df = pd.read_csv(csv_path)
                df = df[df["Image"] != "Average"]  # Remove existing average row
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            else:
                # Create new DataFrame
                df = pd.DataFrame([new_row])
            
            # Add average row
            df.loc["Average"] = df.drop("Image", axis=1).mean(numeric_only=True).round(4)
            df.at["Average", "Image"] = "Average"
            
            # Save updated CSV
            df.to_csv(csv_path, index=False)

        return sv, heat, (rma, rra, iou, pg)

    def visualize_shap(self, image, shap_explanation, method, save_path=None):
        """
        Draws the SHAP image plot using the full Explanation object,
        so class regions and legends are preserved.
        """
        img = self.preprocess_image(image)
        # Normalize pixel values for plotting
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        else:
            for c in range(3):
                ch = img[..., c]
                img[..., c] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)

        plt.ioff()
        fig = plt.figure(figsize=(15, 5))
        labels = [self.class_names[i] for i in range(self.num_cls)]
        shap.image_plot(shap_explanation, np.expand_dims(img, 0), labels=labels)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close(fig)
        plt.close('all')

    def explain_batch(self, images, image_names, masks, save_dir=None, methods=['global','local','gate','concat','product'], max_evals=100, batch_size=50):
        """
        Batch-run explain_image; dump all PNGs and CSVs flat into save_dir (or base_save_dir).
        Returns:
          { method: [ (raw_explanation, heatmap, metrics_dict), ... ] }
        """
        out_dir = save_dir or self.base_save_dir
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        if masks is None:
            masks = [None] * len(images)

        results = {}
        for method in methods:
            print(f"\nGenerating SHAP explanations for {method}…")
            per_image = []
            metrics_list = []

            for img, name, m in zip(images, image_names, masks):
                sv, heat, (rma, rra, iou, pg) = self.explain_image(img, method, m, name, save_dir=out_dir, max_evals=max_evals, batch_size=batch_size)

                # Save the visualization flat
                base = os.path.splitext(os.path.basename(name))[0]
                png_name = f"{base}_{method}_shap.png"
                png_path = os.path.join(out_dir, png_name)
                self.visualize_shap(img, sv, method, save_path=png_path)

                per_image.append((sv, heat, {"RMA": rma, "RRA": rra, "IoU": iou, "PointingGame": pg}))
                metrics_list.append({"Image": name, "RMA": rma, "RRA": rra, "IoU": iou, "PointingGame": pg})

            # Write out CSV summary
            df = pd.DataFrame(metrics_list)
            df.loc["Average"] = df.drop("Image", axis=1).mean(numeric_only=True).round(4)
            df.at["Average", "Image"] = "Average"
            csv_path = os.path.join(out_dir, f"SHAP_XAI_Metrics [{method}].csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved SHAP XAI summary for {method} to {csv_path}")

            results[method] = per_image

        return results
