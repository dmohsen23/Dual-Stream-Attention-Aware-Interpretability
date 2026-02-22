import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class Performance:
    """
    Class for evaluating model performance.
    """
    @staticmethod
    def predictive_performance(y_true, y_prob_global, y_prob_local, y_prob_fusion_gate, y_prob_fusion_concat, y_prob_fusion_product):
        """
        Compute Accuracy, Precision, Recall, F1-score, and AUC for all networks.
        """
        # Convert probabilities to class predictions
        y_pred_global = np.argmax(y_prob_global, axis=1)  
        y_pred_local = np.argmax(y_prob_local, axis=1)  
        y_pred_fusion_gate = np.argmax(y_prob_fusion_gate, axis=1)
        y_pred_fusion_concat = np.argmax(y_prob_fusion_concat, axis=1)
        y_pred_fusion_product = np.argmax(y_prob_fusion_product, axis=1)

        # Compute Accuracy, Precision, Recall, and F1-score
        def compute_metrics(y_true, y_pred):
            acc = round(accuracy_score(y_true, y_pred), 4)
            precision = round(precision_score(y_true, y_pred, average='weighted', zero_division=0), 4)
            recall = round(recall_score(y_true, y_pred, average='weighted', zero_division=0), 4)
            f1 = round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4)
            return acc, precision, recall, f1

        acc_g, prec_g, rec_g, f1_g = compute_metrics(y_true, y_pred_global)
        acc_l, prec_l, rec_l, f1_l = compute_metrics(y_true, y_pred_local)
        acc_fg, prec_fg, rec_fg, f1_fg = compute_metrics(y_true, y_pred_fusion_gate)
        acc_fc, prec_fc, rec_fc, f1_fc = compute_metrics(y_true, y_pred_fusion_concat)
        acc_fp, prec_fp, rec_fp, f1_fp = compute_metrics(y_true, y_pred_fusion_product)

        # Compute AUC Score (Multi-Class)
        try:
            num_classes = len(np.unique(y_true))
            if num_classes == 2:
                # Binary classification — use prob for class 1
                auc_g = round(roc_auc_score(y_true, y_prob_global[:, 1]), 4)
                auc_l = round(roc_auc_score(y_true, y_prob_local[:, 1]), 4)
                auc_fg = round(roc_auc_score(y_true, y_prob_fusion_gate[:, 1]), 4)
                auc_fc = round(roc_auc_score(y_true, y_prob_fusion_concat[:, 1]), 4)
                auc_fp = round(roc_auc_score(y_true, y_prob_fusion_product[:, 1]), 4)
            else:
                # Multi-class
                auc_g = round(roc_auc_score(y_true, y_prob_global, multi_class='ovr', average='weighted'), 4)
                auc_l = round(roc_auc_score(y_true, y_prob_local, multi_class='ovr', average='weighted'), 4)
                auc_fg = round(roc_auc_score(y_true, y_prob_fusion_gate, multi_class='ovr', average='weighted'), 4)
                auc_fc = round(roc_auc_score(y_true, y_prob_fusion_concat, multi_class='ovr', average='weighted'), 4)
                auc_fp = round(roc_auc_score(y_true, y_prob_fusion_product, multi_class='ovr', average='weighted'), 4)
        except ValueError:
            auc_g, auc_l, auc_fg, auc_fc, auc_fp = None, None, None, None, None

        metrics = {
            'Global': {'Accuracy': acc_g, 'Precision': prec_g, 'Recall': rec_g, 'F1_score': f1_g, 'AUC': auc_g},
            'Local': {'Accuracy': acc_l, 'Precision': prec_l, 'Recall': rec_l, 'F1_score': f1_l, 'AUC': auc_l},
            'Fusion_Gate': {'Accuracy': acc_fg, 'Precision': prec_fg, 'Recall': rec_fg, 'F1_score': f1_fg, 'AUC': auc_fg},
            'Fusion_Concat': {'Accuracy': acc_fc, 'Precision': prec_fc, 'Recall': rec_fc, 'F1_score': f1_fc, 'AUC': auc_fc},
            'Fusion_Product': {'Accuracy': acc_fp, 'Precision': prec_fp, 'Recall': rec_fp, 'F1_score': f1_fp, 'AUC': auc_fp}}

        return metrics

    @staticmethod
    def compute_and_log_metrics(y_true, y_probs, metric_lists, metric_names, phase, save_dir, today):
        """
        Compute performance metrics, store values in lists, calculate standard deviation,
        and save results to a CSV file.
        """
        # Compute metrics
        performance = Performance.predictive_performance(y_true, y_probs[0], y_probs[1], y_probs[2], y_probs[3], y_probs[4])

        # Append values to lists
        for model_type in ["Global", "Local", "Fusion_Gate", "Fusion_Concat", "Fusion_Product"]:
            for metric in metric_names:
                value = performance[model_type][metric]
                if value is None:  # Handle missing AUC cases
                    value = 0.0
                metric_lists[model_type][metric].append(value)

        # Compute standard deviations safely
        std_metrics = {
            model_type: {metric: np.std(metric_lists[model_type][metric]) if metric_lists[model_type][metric] else 0.0
                        for metric in metric_names}
            for model_type in ["Global", "Local", "Fusion_Gate", "Fusion_Concat", "Fusion_Product"]}

        # Convert performance dictionary to DataFrame
        performance_df = pd.DataFrame.from_dict(performance, orient='index', columns=metric_names)

        # Save results
        file_path = os.path.join(save_dir, f"{phase}_Performance [{today}].csv")
        performance_df.to_csv(file_path)

        return performance, std_metrics  # Returning standard deviations
    
    @staticmethod
    def compute_training_metrics(metrics_all_epochs, save_dir=None, today=None):

        flattened_metrics = []
        
        for epoch_metrics in metrics_all_epochs:
            flat_metrics = {}
            for category, metrics in epoch_metrics.items():
                for metric_name, value in metrics.items():
                    flat_metrics[f"{category}_{metric_name}"] = value
            flattened_metrics.append(flat_metrics)

        metrics_df = pd.DataFrame(flattened_metrics)
        mean_metrics = metrics_df.mean()
        std_metrics = metrics_df.std()

        # If save_dir is provided, save the results as a CSV file
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            results_df = pd.DataFrame({
                "Metric": mean_metrics.index,
                "Mean": mean_metrics.values,
                "Std Dev": std_metrics.values
            })
            save_path = os.path.join(save_dir, f"Mean_Training_Performance [{today}].csv")
            results_df.to_csv(save_path, index=False)

        return mean_metrics, std_metrics
    
    @staticmethod
    def compute_confusion_matrices(y_true, y_prob_global, y_prob_local, y_prob_fusion_gate, y_prob_fusion_concat, y_prob_fusion_product, class_names=None, save_dir=None, prefix=""):
        """
        Compute confusion matrices for Global, Local, and Fusion predictions.
        
        Args:
            y_true (array): Ground truth labels.
            y_prob_global (array): Probability predictions for Global network.
            y_prob_local (array): Probability predictions for Local network.
            y_prob_fusion_gate (array): Probability predictions for Fusion_Gate network.
            y_prob_fusion_concat (array): Probability predictions for Fusion_Concat network.
            y_prob_fusion_product (array): Probability predictions for Fusion_Product network.
            class_names (list, optional): List of class names for labeling the confusion matrix.
            save_dir (str, optional): Directory to save confusion matrix figures or CSVs.
            prefix (str, optional): Prefix for saved file names.
        
        Returns:
            dict: A dictionary containing the confusion matrices for Global, Local, and Fusion.
        """
        # Convert probabilities to predicted labels
        y_pred_global = np.argmax(y_prob_global, axis=1)
        y_pred_local = np.argmax(y_prob_local, axis=1)
        y_pred_fusion_gate = np.argmax(y_prob_fusion_gate, axis=1)
        y_pred_fusion_concat = np.argmax(y_prob_fusion_concat, axis=1)
        y_pred_fusion_product = np.argmax(y_prob_fusion_product, axis=1)

        cm_global = confusion_matrix(y_true, y_pred_global)
        cm_local = confusion_matrix(y_true, y_pred_local)
        cm_fusion_gate = confusion_matrix(y_true, y_pred_fusion_gate)
        cm_fusion_concat = confusion_matrix(y_true, y_pred_fusion_concat)
        cm_fusion_product = confusion_matrix(y_true, y_pred_fusion_product)

        cm_dict = {'Global': cm_global, 'Local': cm_local, 'Fusion_Gate': cm_fusion_gate, 'Fusion_Concat': cm_fusion_concat, 'Fusion_Product': cm_fusion_product}

        # Optionally, save or plot the confusion matrices
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            for model_type, cm in cm_dict.items():
                # Save numeric matrix as CSV
                cm_csv_path = os.path.join(save_dir, f"{prefix}{model_type}_confusion_matrix.csv")
                pd.DataFrame(cm).to_csv(cm_csv_path, index=False, header=False)

                # Optionally create a heatmap and save as an image
                if class_names:
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d',
                                xticklabels=class_names, yticklabels=class_names, ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('True')
                    plt.title(f'{model_type} Confusion Matrix')
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f"{prefix}{model_type}_confusion_matrix.png"))
                    plt.close(fig)

        return cm_dict
