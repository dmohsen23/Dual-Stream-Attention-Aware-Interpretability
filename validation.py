import torch
import time
import numpy as np
import pandas as pd
import os
import torch.nn.functional as F
from performance import Performance
from torch.cuda.amp import autocast

class Model_Validation:
    def __init__(self, model, criterion_global, criterion_local, criterion_fusion, global_w, local_w, fusion_w, device, save_dir, today):
        """
        Initializes the validation model class with necessary parameters.
        """
        self.model = model
        self.criterion_global = criterion_global
        self.criterion_local = criterion_local
        self.criterion_fusion = criterion_fusion
        self.global_w = global_w
        self.local_w = local_w
        self.fusion_w = fusion_w
        self.device = device
        self.save_dir = save_dir
        self.today = today

    def validate(self, val_loader, image_names, class_name, phase="Validation"):
        """
        Performs model validation and calculates accuracy metrics.
        """
        self.model.eval()
        running_loss = 0.0
        y_true, y_prob_global, y_prob_local, y_prob_fusion = [], [], [], []

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 4:                 # ISIC dataset (images, masks, labels, image_ids)
                    images, _, labels, _ = batch    # Ignore masks and image IDs
                elif len(batch) == 3:               # Expected for datasets without masks
                    images, _, labels = batch
                elif len(batch) == 2:               # Expected for Distal Myopathy dataset
                    images, labels = batch
                else:
                    raise ValueError(f"Unexpected batch format: got {len(batch)} elements.")

                images, labels = images.to(self.device, memory_format=torch.channels_last), labels.to(self.device)

                with autocast():
                    g_out, l_out, f_out, _ = self.model(images)
                    loss_global = self.criterion_global(g_out, labels)
                    loss_local = self.criterion_local(l_out, labels)
                    loss_fusion = self.criterion_fusion(f_out, labels)
                    loss = (self.global_w * loss_global + self.local_w * loss_local + self.fusion_w * loss_fusion)

                running_loss += loss.item()

                # Store predictions for metric calculations
                y_true.extend(labels.cpu().numpy())
                y_prob_global.extend(F.softmax(g_out, dim=1).detach().cpu().to(torch.float32).numpy())
                y_prob_local.extend(F.softmax(l_out, dim=1).detach().cpu().to(torch.float32).numpy())
                y_prob_fusion.extend(F.softmax(f_out, dim=1).detach().cpu().to(torch.float32).numpy())

        y_true = np.array(y_true)
        y_prob_global = np.array(y_prob_global)
        y_prob_local = np.array(y_prob_local)
        y_prob_fusion = np.array(y_prob_fusion)

        return running_loss / len(val_loader), y_true, y_prob_global, y_prob_local, y_prob_fusion

    def metric_calculation(self, phase, y_true, y_prob_global, y_prob_local, y_prob_fusion):
        """
        Computes and logs validation/training metrics.
        """
        metric_lists_val = {
            "Global": {"Accuracy": [], "Precision": [], "Recall": [], "F1_score": [], "AUC": []},
            "Local": {"Accuracy": [], "Precision": [], "Recall": [], "F1_score": [], "AUC": []},
            "Fusion": {"Accuracy": [], "Precision": [], "Recall": [], "F1_score": [], "AUC": []}}
        metric_names = ["Accuracy", "Precision", "Recall", "F1_score", "AUC"]

        val_performance, _ = Performance.compute_and_log_metrics(y_true, (y_prob_global, y_prob_local, y_prob_fusion), metric_lists_val, 
                                                               metric_names, phase=phase, save_dir=self.save_dir, today=self.today)

        print(f"{phase} Accuracy >> Global: {val_performance['Global']['Accuracy']:.4f} | "
              f"Local: {val_performance['Local']['Accuracy']:.4f} | "
              f"Fusion: {val_performance['Fusion']['Accuracy']:.4f}")

        return val_performance
