import torch
import os
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast
from torch.nn import functional as F
import time
from performance import Performance
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from incremental_deletion import incremental_deletion_analysis
# Use seaborn style
sns.set_style("darkgrid")

class CrossValidation:
    def __init__(self):
        self.writer = SummaryWriter()
        self.all_fold_true_labels = []
        self.all_fold_pred_probs = {
            'Global': [],
            'Local': [],
            'Fusion_Gate': [],
            'Fusion_Concat': [],
            'Fusion_Product': []
        }
        self.correctly_classified_images = []
        self._seen_correct = set()
        self.best_models = []  # Store best model from each fold

    def run_cross_validation(self, model, criterion_global, criterion_local, criterion_fusion, optimizer_global, optimizer_local, optimizer_fusion, 
                             global_w, local_w, fusion_w, class_name, cv_folds, cv_mask_folds, device, num_epochs=None, save_dir=None, today=None,
                             fold_image_names=None, mask_loader=None, img_size=None):
        
        train_loss_all_folds, val_loss_all_folds = [], []
        train_metrics_all_folds, val_metrics_all_folds = [], []
        cv_results_dir = os.path.join(save_dir, "CV_Results")
        os.makedirs(cv_results_dir, exist_ok=True)

        # Initialize lists to store labels and predictions for ROC curve
        self.all_fold_true_labels = []
        self.all_fold_pred_probs = {
            'Global': [],
            'Local': [],
            'Fusion_Gate': [],
            'Fusion_Concat': [],
            'Fusion_Product': []
        }
        self.best_models = []  # Reset best models list

        evaluate_xai = cv_mask_folds is not None or mask_loader is not None
        if not evaluate_xai:
            print("Warning: No mask loader provided. XAI metrics will not be calculated.")

        for fold, (train_loader, val_loader, original_val_loader) in enumerate(cv_folds):
            print(f'\n======== Fold {fold + 1}/{len(cv_folds)} ========')
            model.apply(reset_weights)
            model.to(device)

            _, fold_val_names = fold_image_names[fold]

            fold_train_metrics = []
            best_val_accuracy = 0
            patience = 10
            no_improvement = 0
            best_model_state = None

            for epoch in range(num_epochs):
                train_start_time = time.time()
                train_loss, train_metrics = self.train_epoch(model, criterion_global, criterion_local, criterion_fusion, optimizer_global, optimizer_local, optimizer_fusion, 
                                                             global_w, local_w, fusion_w, train_loader, device, fold, cv_results_dir, today)
                train_end_time = time.time() - train_start_time

                train_loss_all_folds.append(train_loss)
                fold_train_metrics.append(train_metrics)

                val_start_time = time.time()
                val_loss, val_metrics, y_true, y_prob_global, y_prob_local, y_prob_fusion_gate, y_prob_fusion_concat, y_prob_fusion_product, correct_images = self.validate_epoch(model, criterion_global, criterion_local, criterion_fusion, 
                                                                                                                                     global_w, local_w, fusion_w, val_loader, device, fold, cv_results_dir, 
                                                                                                                                     today, class_name, fold_val_names, cv_mask_folds, mask_loader, img_size)
                
                self.correctly_classified_images.extend(correct_images)

                val_end_time = time.time() - val_start_time

                # Store labels and predictions for ROC curve
                self.all_fold_true_labels.extend(y_true)
                self.all_fold_pred_probs['Global'].extend(y_prob_global)
                self.all_fold_pred_probs['Local'].extend(y_prob_local)
                self.all_fold_pred_probs['Fusion_Gate'].extend(y_prob_fusion_gate)
                self.all_fold_pred_probs['Fusion_Concat'].extend(y_prob_fusion_concat)
                self.all_fold_pred_probs['Fusion_Product'].extend(y_prob_fusion_product)

                print(f"\nEpoch {epoch+1}")
                print(f"Training Epoch {epoch+1} completed in {train_end_time:.2f} seconds")
                print(f"Training Loss: {train_loss:.4f}")
                print(f"Training Accuracy >> Global: {train_metrics['Global']['Accuracy']:.4f} | "
                      f"Local: {train_metrics['Local']['Accuracy']:.4f} | "
                      f"Fusion_Gate: {train_metrics['Fusion_Gate']['Accuracy']:.4f} | "
                      f"Fusion_Concat: {train_metrics['Fusion_Concat']['Accuracy']:.4f} | "
                      f"Fusion_Product: {train_metrics['Fusion_Product']['Accuracy']:.4f}")

                print(f"Validation Epoch {epoch+1} completed in {val_end_time:.2f} seconds")
                print(f"Validation Loss: {val_loss:.4f}")
                print(f"Validation Accuracy >> Global: {val_metrics['Global']['Accuracy']:.4f} | "
                      f"Local: {val_metrics['Local']['Accuracy']:.4f} | "
                      f"Fusion_Gate: {val_metrics['Fusion_Gate']['Accuracy']:.4f} | "
                      f"Fusion_Concat: {val_metrics['Fusion_Concat']['Accuracy']:.4f} | "
                      f"Fusion_Product: {val_metrics['Fusion_Product']['Accuracy']:.4f}")

                if val_metrics['Fusion_Gate']['Accuracy'] > best_val_accuracy:
                    best_val_accuracy = val_metrics['Fusion_Gate']['Accuracy']
                    no_improvement = 0
                    best_model_state = model.state_dict().copy()
                    torch.save(model.state_dict(), os.path.join(cv_results_dir, f"best_model_fold_{fold+1}_{today}.pth"))
                else:
                    no_improvement += 1
                    print(f"No improvement for {no_improvement} epochs")

                if no_improvement >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break

            # Load the best model for this fold
            model.load_state_dict(best_model_state)
            self.best_models.append(best_model_state)

            final_val_loss, final_val_metrics, _, _, _, _, _, _, _ = self.validate_epoch(model, criterion_global, criterion_local, criterion_fusion, global_w, local_w, fusion_w, 
                                                                                        val_loader, device, fold, cv_results_dir, today, class_name, fold_val_names, cv_mask_folds, mask_loader, img_size)
            
            val_loss_all_folds.append(final_val_loss)
            val_metrics_all_folds.append(final_val_metrics)
            train_metrics_all_folds.append(fold_train_metrics[-1])

        self.print_cv_results(train_loss_all_folds, val_loss_all_folds, train_metrics_all_folds, val_metrics_all_folds, cv_results_dir, today)
        
        # Only plot ROC curve if we have collected labels and predictions
        if self.all_fold_true_labels and any(preds for preds in self.all_fold_pred_probs.values()):
            self.plot_and_save_roc_curve(cv_results_dir, class_name, today)
        else:
            print("Warning: No labels and predictions collected for ROC curve plotting")
        self.writer.close()

        # Find the best performing fold and use its model
        best_fold_idx = np.argmax([metrics['Fusion_Gate']['Accuracy'] for metrics in val_metrics_all_folds])
        best_model_state = self.best_models[best_fold_idx]
        model.load_state_dict(best_model_state)

        return np.mean([metrics['Fusion_Gate']['Accuracy'] for metrics in val_metrics_all_folds])

    def train_epoch(self, model, criterion_global, criterion_local, criterion_fusion, optimizer_global, optimizer_local, optimizer_fusion,
                    global_w, local_w, fusion_w, train_loader, device, fold, cv_results_dir, today):
        """ Train a single epoch and compute metrics using `Performance.compute_and_log_metrics()`. """
        model.train()
        running_loss = 0.0
        y_true, y_prob_global, y_prob_local = [], [], []
        y_prob_fusion_gate, y_prob_fusion_concat, y_prob_fusion_product = [], [], []

        for batch in train_loader:
            if len(batch) == 4:
                images, _, labels, _ = batch    # Ignore masks and image IDs
            elif len(batch) == 3:               # Expected for datasets without masks
                images, _, labels = batch
            elif len(batch) == 2:               # Expected for Distal Myopathy dataset
                images, labels = batch
            else:
                raise ValueError(f"Unexpected batch format: got {len(batch)} elements.")

            images, labels = images.to(device, memory_format=torch.channels_last), labels.to(device)
            
            # Zero gradients for all optimizers
            optimizer_global.zero_grad()
            optimizer_local.zero_grad()
            optimizer_fusion.zero_grad()

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                # Get outputs for all fusion types
                core = model.module if isinstance(model, torch.nn.DataParallel) else model
                results = core.evaluate_all_fusion_types(images)
                g_out, l_out = results['global']['output'], results['local']['output']
                
                # Calculate losses for global and local branches
                loss_global = criterion_global(g_out, labels)
                loss_local = criterion_local(l_out, labels)
                
                # Calculate fusion losses for each type
                loss_fusion_gate = criterion_fusion(results['gate']['output'], labels)
                loss_fusion_concat = criterion_fusion(results['concat']['output'], labels)
                loss_fusion_product = criterion_fusion(results['product']['output'], labels)
                
                # Calculate weighted losses
                weighted_loss_global = global_w * loss_global
                weighted_loss_local = local_w * loss_local
                weighted_loss_fusion = fusion_w * (loss_fusion_gate + loss_fusion_concat + loss_fusion_product) / 3
                total_loss = weighted_loss_global + weighted_loss_local + weighted_loss_fusion

            # Backward pass
            total_loss.backward()

            # Step optimizers
            optimizer_global.step()
            optimizer_local.step()
            optimizer_fusion.step()

            running_loss += total_loss.item()

            # Store predictions for metric calculations
            y_true.extend(labels.cpu().numpy())
            y_prob_global.extend(F.softmax(g_out, dim=1).detach().cpu().to(torch.float32).numpy())
            y_prob_local.extend(F.softmax(l_out, dim=1).detach().cpu().to(torch.float32).numpy())
            y_prob_fusion_gate.extend(F.softmax(results['gate']['output'], dim=1).detach().cpu().to(torch.float32).numpy())
            y_prob_fusion_concat.extend(F.softmax(results['concat']['output'], dim=1).detach().cpu().to(torch.float32).numpy())
            y_prob_fusion_product.extend(F.softmax(results['product']['output'], dim=1).detach().cpu().to(torch.float32).numpy())

        y_true = np.array(y_true)
        y_prob_global = np.array(y_prob_global)
        y_prob_local = np.array(y_prob_local)
        y_prob_fusion_gate = np.array(y_prob_fusion_gate)
        y_prob_fusion_concat = np.array(y_prob_fusion_concat)
        y_prob_fusion_product = np.array(y_prob_fusion_product)

        # Compute metrics using "Performance.compute_and_log_metrics()"
        metric_lists_train = {
            "Global": {"Accuracy": [], "Precision": [], "Recall": [], "F1_score": [], "AUC": []},
            "Local": {"Accuracy": [], "Precision": [], "Recall": [], "F1_score": [], "AUC": []},
            "Fusion_Gate": {"Accuracy": [], "Precision": [], "Recall": [], "F1_score": [], "AUC": []},
            "Fusion_Concat": {"Accuracy": [], "Precision": [], "Recall": [], "F1_score": [], "AUC": []},
            "Fusion_Product": {"Accuracy": [], "Precision": [], "Recall": [], "F1_score": [], "AUC": []}}
        metric_names = ["Accuracy", "Precision", "Recall", "F1_score", "AUC"]

        train_metrics, _ = Performance.compute_and_log_metrics(y_true, 
            (y_prob_global, y_prob_local, y_prob_fusion_gate, y_prob_fusion_concat, y_prob_fusion_product),
            metric_lists_train, metric_names, phase=f"Training_Fold_{fold+1}", save_dir=cv_results_dir, today=today)

        return running_loss / len(train_loader), train_metrics

    def print_cv_results(self, train_loss_all_folds, val_loss_all_folds, train_metrics_all_folds, val_metrics_all_folds, cv_results_dir, today):
        mean_train_metrics, std_train_metrics = self.compute_average_metrics(train_metrics_all_folds)
        mean_val_metrics, std_val_metrics = self.compute_average_metrics(val_metrics_all_folds)

        print("\n========= Final Cross-Validation Results =========")
        print(f"\nAverage Training Loss: {np.mean(train_loss_all_folds):.4f} ± {np.std(train_loss_all_folds):.4f} | "
              f"Average Validation Loss: {np.mean(val_loss_all_folds):.4f} ± {np.std(val_loss_all_folds):.4f}")        
        
        print(f"\nTraining Accuracy >> Global: {mean_train_metrics['Global_Accuracy']:.4f} ± {std_train_metrics['Global_Accuracy']:.4f} | "
              f"Local: {mean_train_metrics['Local_Accuracy']:.4f} ± {std_train_metrics['Local_Accuracy']:.4f} | "
              f"Fusion_Gate: {mean_train_metrics['Fusion_Gate_Accuracy']:.4f} ± {std_train_metrics['Fusion_Gate_Accuracy']:.4f} | "
              f"Fusion_Concat: {mean_train_metrics['Fusion_Concat_Accuracy']:.4f} ± {std_train_metrics['Fusion_Concat_Accuracy']:.4f} | "
              f"Fusion_Product: {mean_train_metrics['Fusion_Product_Accuracy']:.4f} ± {std_train_metrics['Fusion_Product_Accuracy']:.4f}")
        
        print(f"\nValidation Accuracy >> Global: {mean_val_metrics['Global_Accuracy']:.4f} ± {std_val_metrics['Global_Accuracy']:.4f} | "
              f"Local: {mean_val_metrics['Local_Accuracy']:.4f} ± {std_val_metrics['Local_Accuracy']:.4f} | "
              f"Fusion_Gate: {mean_val_metrics['Fusion_Gate_Accuracy']:.4f} ± {std_val_metrics['Fusion_Gate_Accuracy']:.4f} | "
              f"Fusion_Concat: {mean_val_metrics['Fusion_Concat_Accuracy']:.4f} ± {std_val_metrics['Fusion_Concat_Accuracy']:.4f} | "
              f"Fusion_Product: {mean_val_metrics['Fusion_Product_Accuracy']:.4f} ± {std_val_metrics['Fusion_Product_Accuracy']:.4f}")

        final_cv_results = pd.DataFrame({
            "Train Mean": mean_train_metrics, "Train Std": std_train_metrics,
            "Val Mean": mean_val_metrics, "Val Std": std_val_metrics})
        final_cv_results.to_csv(os.path.join(cv_results_dir, f"CV_Summary_{today}.csv"), index=True)

    def compute_average_metrics(self, metrics_all_folds):
        """ Compute the average and standard deviation for each metric across all folds. """
        flattened_metrics = []
        for fold_metrics in metrics_all_folds:
            flat_metrics = {}
            for category, metrics in fold_metrics.items():
                for metric_name, value in metrics.items():
                    flat_metrics[f"{category}_{metric_name}"] = value
            flattened_metrics.append(flat_metrics)

        metrics_df = pd.DataFrame(flattened_metrics)
        mean_metrics = round(metrics_df.mean(), 4)
        std_metrics = round(metrics_df.std(), 4)

        return mean_metrics, std_metrics

    def plot_and_save_roc_curve(self, save_dir, class_name, today):
        """ 
        Plot and save the ROC curve for all models (Global, Local, Fusion_Gate, Fusion_Concat, Fusion_Product) on the same plot.
        """
        # Convert lists to numpy arrays
        true_labels = np.array(self.all_fold_true_labels)
        
        # Define colors for each model
        colors = {
            'Global': "#1f58b4",            # Blue
            'Local': '#ff7f0e',             # Orange
            'Fusion_Gate': '#2ca02c',       # Green
            'Fusion_Concat': '#d62728',     # Red
            'Fusion_Product': '#9467bd'     # Purple
        }
        
        # Try to read AUC values from the CV summary file
        auc_values = {}
        cv_summary_path = os.path.join(save_dir, f"CV_Summary_{today}.csv")
        if os.path.exists(cv_summary_path):
            try:
                cv_summary = pd.read_csv(cv_summary_path, index_col=0)
                # Extract AUC values from the summary
                for model_name in ['Global', 'Local', 'Fusion_Gate', 'Fusion_Concat', 'Fusion_Product']:
                    auc_col = f"{model_name}_AUC"
                    if auc_col in cv_summary.index:
                        auc_values[model_name] = cv_summary.loc[auc_col, 'Val Mean']
                    else:
                        auc_values[model_name] = None
            except Exception as e:
                print(f"Warning: Could not read AUC values from CV summary: {e}")
                auc_values = {}
        else:
            print(f"Warning: CV summary file not found at {cv_summary_path}")
            auc_values = {}
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot ROC curves for each model
        for model_name, pred_probs in self.all_fold_pred_probs.items():
            if not pred_probs:  # Skip if no predictions
                continue
                
            pred_probs = np.array(pred_probs)
            
            # Calculate ROC curve for plotting
            if pred_probs.shape[1] == 2:
                # Binary classification
                pred_probs_binary = pred_probs[:, 1]  # Take the probability of the positive class
                true_labels_flat = true_labels.ravel()
                fpr, tpr, _ = roc_curve(true_labels_flat, pred_probs_binary)
            else:
                # Multi-class classification
                classes = np.unique(true_labels)
                y_bin = label_binarize(true_labels, classes=classes)
                n_classes = y_bin.shape[1]
                
                # Compute ROC curve and ROC area for each class
                fpr = dict()
                tpr = dict()
                roc_auc_micro = dict()
                
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], pred_probs[:, i])
                    roc_auc_micro[i] = auc(fpr[i], tpr[i])
                
                # Compute micro-average ROC curve and ROC area
                fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), pred_probs.ravel())
                roc_auc_micro["micro"] = auc(fpr["micro"], tpr["micro"])
                
                # Use micro-average for plotting
                fpr, tpr = fpr["micro"], tpr["micro"]
            
            # Use AUC value from summary file if available, otherwise calculate it
            if model_name in auc_values and auc_values[model_name] is not None:
                roc_auc = auc_values[model_name]
            else:
                # Fallback: calculate AUC using the same method as in performance.py
                if pred_probs.shape[1] == 2:
                    roc_auc = roc_auc_score(true_labels_flat, pred_probs_binary)
                else:
                    roc_auc = roc_auc_score(true_labels, pred_probs, multi_class='ovr', average='weighted')
            
            # Plot the ROC curve with increased line thickness
            ax.plot(fpr, tpr, color=colors[model_name], lw=4, label=f'{model_name} (AUC = {roc_auc:.4f})')
        
        # Plot diagonal line (random classifier) with increased thickness
        ax.plot([0, 1], [0, 1], linestyle='--', lw=3, color='gray', 
               label='Random Classifier', alpha=0.8)
        
        # Plot settings with increased font sizes and bold text
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=18, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=18, fontweight='bold')
        # ax.set_title('ROC Curves for All Models', fontsize=22, fontweight='bold', pad=20)
        legend = ax.legend(loc="lower right", fontsize=18, title_fontsize=20, framealpha=0.9)
        # Make legend text bold
        for text in legend.get_texts():
            text.set_fontweight('bold')
        ax.grid(True, alpha=0.3)
        
        # Increase tick label font sizes and make them bold
        ax.tick_params(axis='both', which='major', labelsize=16)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'roc_auc_all_models_{today}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ROC curves for all models saved to {os.path.join(save_dir, f'roc_auc_all_models_{today}.png')}")

    def validate_epoch(self, model, criterion_global, criterion_local, criterion_fusion, global_w, local_w, fusion_w, val_loader, device, fold,
                       cv_results_dir, today, class_name, image_names, cv_mask_folds=None, mask_loader=None, img_size=None):
        """ Validate a single epoch, compute metrics, and log per-image predictions. """
        model.eval()
        running_loss = 0.0
        y_true, y_prob_global, y_prob_local = [], [], []
        y_prob_fusion_gate, y_prob_fusion_concat, y_prob_fusion_product = [], [], []

        per_image_data = []
        image_index = 0
        correct_images = []
        seen_set = set()
        
        # Initialize mask iterator if available
        mask_iter = None
        if mask_loader is not None:
            mask_iter = iter(mask_loader)
        elif cv_mask_folds is not None and fold < len(cv_mask_folds):
            try:
                _, val_mask_loader, _ = cv_mask_folds[fold]
                mask_iter = iter(val_mask_loader)
            except Exception as e:
                mask_iter = None

        with torch.no_grad():
            for _, batch in enumerate(val_loader):
                if len(batch) == 4:
                    images, masks, labels, _ = batch
                elif len(batch) == 3:
                    images, masks, labels = batch
                elif len(batch) == 2:
                    images, labels = batch
                    # Try to get masks from mask iterator if available
                    if mask_iter is not None:
                        try:
                            mask_batch = next(mask_iter)
                            
                            # Try different possible mask locations
                            if len(mask_batch) >= 2:
                                masks = mask_batch[1]  # Try second element
                            elif len(mask_batch) == 1:
                                masks = mask_batch[0]  # Try first element if only one
                            else:
                                masks = None
                        except Exception as e:
                            masks = None
                    else:
                        masks = None
                else:
                    raise ValueError(f"Unexpected batch format: got {len(batch)} elements.")

                images, labels = images.to(device), labels.to(device)

                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    # Get outputs for all fusion types
                    core = model.module if isinstance(model, torch.nn.DataParallel) else model
                    results = core.evaluate_all_fusion_types(images)
                    g_out, l_out = results['global']['output'], results['local']['output']
                    
                    # Calculate losses for global and local branches
                    loss_global = criterion_global(g_out, labels)
                    loss_local = criterion_local(l_out, labels)
                    
                    # Calculate fusion losses for each type
                    loss_fusion_gate = criterion_fusion(results['gate']['output'], labels)
                    loss_fusion_concat = criterion_fusion(results['concat']['output'], labels)
                    loss_fusion_product = criterion_fusion(results['product']['output'], labels)
                    
                    # Calculate weighted losses
                    weighted_loss_global = global_w * loss_global
                    weighted_loss_local = local_w * loss_local
                    weighted_loss_fusion = fusion_w * (loss_fusion_gate + loss_fusion_concat + loss_fusion_product) / 3
                    total_loss = weighted_loss_global + weighted_loss_local + weighted_loss_fusion

                running_loss += total_loss.item()

                # Convert outputs to probabilities
                prob_g = F.softmax(g_out, dim=1).detach().cpu().to(torch.float32).numpy()
                prob_l = F.softmax(l_out, dim=1).detach().cpu().to(torch.float32).numpy()
                prob_f_gate = F.softmax(results['gate']['output'], dim=1).detach().cpu().to(torch.float32).numpy()
                prob_f_concat = F.softmax(results['concat']['output'], dim=1).detach().cpu().to(torch.float32).numpy()
                prob_f_product = F.softmax(results['product']['output'], dim=1).detach().cpu().to(torch.float32).numpy()

                pred_g = np.argmax(prob_g, axis=1)
                pred_l = np.argmax(prob_l, axis=1)
                pred_f_gate = np.argmax(prob_f_gate, axis=1)
                pred_f_concat = np.argmax(prob_f_concat, axis=1)
                pred_f_product = np.argmax(prob_f_product, axis=1)
                labels_np = labels.cpu().numpy()

                y_true.extend(labels_np)
                y_prob_global.extend(prob_g)
                y_prob_local.extend(prob_l)
                y_prob_fusion_gate.extend(prob_f_gate)
                y_prob_fusion_concat.extend(prob_f_concat)
                y_prob_fusion_product.extend(prob_f_product)

                for i in range(len(labels_np)):
                    if image_index >= len(image_names):
                        break

                    record = {
                        "Image": image_names[image_index],
                        "GroundTruth": class_name[labels_np[i]],
                        "Global_Pred": class_name[pred_g[i]],
                        "Local_Pred": class_name[pred_l[i]],
                        "Fusion_Gate_Pred": class_name[pred_f_gate[i]],
                        "Fusion_Concat_Pred": class_name[pred_f_concat[i]],
                        "Fusion_Product_Pred": class_name[pred_f_product[i]]
                    }

                    for j in range(len(class_name)):
                        cname = class_name[j]
                        record[f"Global_Prob_{cname}"] = prob_g[i][j]
                        record[f"Local_Prob_{cname}"] = prob_l[i][j]
                        record[f"Fusion_Gate_Prob_{cname}"] = prob_f_gate[i][j]
                        record[f"Fusion_Concat_Prob_{cname}"] = prob_f_concat[i][j]
                        record[f"Fusion_Product_Prob_{cname}"] = prob_f_product[i][j]

                    # Check if the image is correctly classified
                    name = image_names[image_index]
                    is_correct = (pred_g[i] == labels_np[i]
                                  and pred_l[i] == labels_np[i]
                                  and pred_f_gate[i] == labels_np[i]
                                  and pred_f_concat[i] == labels_np[i]
                                  and pred_f_product[i] == labels_np[i])
                    
                    if is_correct and name not in self._seen_correct:
                        record["Label"] = "Candidate"
                        if name not in seen_set:
                            self._seen_correct.add(name)
                            # Store image data for incremental deletion
                            mask_for_image = masks[i].cpu().clone() if masks is not None else None
                            
                            correct_images.append({
                                "name": name,
                                "label": class_name[labels_np[i]],
                                "image": images[i].cpu().clone(),
                                "mask": mask_for_image
                            })
                        else:
                            record["Label"] = "Ignore"

                    per_image_data.append(record)
                    image_index += 1

        # Save per-image prediction results
        results_df = pd.DataFrame(per_image_data)
        # Calculate average values for all probability columns
        probability_cols = [col for col in results_df.columns if col.startswith("Global_Prob_") or 
                                                                 col.startswith("Local_Prob_") or 
                                                                 col.startswith("Fusion_Gate_Prob_") or
                                                                 col.startswith("Fusion_Concat_Prob_") or
                                                                 col.startswith("Fusion_Product_Prob_")]
        # Round to 4 decimal places
        avg_row = results_df[probability_cols].mean().round(4).to_dict()

        # Fill non-probability fields for the average row
        avg_row.update({
            "Image": "Average",
            "GroundTruth": "",
            "Global_Pred": "",
            "Local_Pred": "",
            "Fusion_Gate_Pred": "",
            "Fusion_Concat_Pred": "",
            "Fusion_Product_Pred": "",
            "Label": ""
        })

        # Append average row to the DataFrame
        results_df = pd.concat([results_df, pd.DataFrame([avg_row])], ignore_index=True)

        results_path = os.path.join(cv_results_dir, f"Predictions_Fold_{fold+1} [{today}].csv")
        results_df.to_csv(results_path, index=False)
        print(f"Saved detailed prediction results to {results_path}")
        assert image_index == len(image_names), f"[BUG] Expected {len(image_names)} images, got {image_index}"

        y_true = np.array(y_true)
        y_prob_global = np.array(y_prob_global)
        y_prob_local = np.array(y_prob_local)
        y_prob_fusion_gate = np.array(y_prob_fusion_gate)
        y_prob_fusion_concat = np.array(y_prob_fusion_concat)
        y_prob_fusion_product = np.array(y_prob_fusion_product)

        # Compute metrics
        metric_lists_val = {
            "Global": {"Accuracy": [], "Precision": [], "Recall": [], "F1_score": [], "AUC": []},
            "Local": {"Accuracy": [], "Precision": [], "Recall": [], "F1_score": [], "AUC": []},
            "Fusion_Gate": {"Accuracy": [], "Precision": [], "Recall": [], "F1_score": [], "AUC": []},
            "Fusion_Concat": {"Accuracy": [], "Precision": [], "Recall": [], "F1_score": [], "AUC": []},
            "Fusion_Product": {"Accuracy": [], "Precision": [], "Recall": [], "F1_score": [], "AUC": []}
        }
        metric_names = ["Accuracy", "Precision", "Recall", "F1_score", "AUC"]

        val_metrics, _ = Performance.compute_and_log_metrics(y_true, (y_prob_global, y_prob_local, y_prob_fusion_gate, y_prob_fusion_concat, y_prob_fusion_product),
            metric_lists_val, metric_names, phase=f"Validation_Fold_{fold+1}", save_dir=cv_results_dir, today=today)

        return running_loss / len(val_loader), val_metrics, y_true, y_prob_global, y_prob_local, y_prob_fusion_gate, y_prob_fusion_concat, y_prob_fusion_product, correct_images

def reset_weights(m):
    """Resets model weights to avoid data leakage between folds."""
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

if __name__ == "__main__":
    print("This script provides a cross-validation function. Import it into another script to use.")