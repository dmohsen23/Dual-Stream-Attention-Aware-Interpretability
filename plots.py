import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

class Plotter:
    """
    Class for plotting training curves and ROC curves
    """
    @staticmethod
    def plot_performance(epochs, train_loss, train_loss_std, train_acc, train_acc_std, val_loss=None, val_loss_std=None, val_acc=None, val_acc_std=None, is_cv='off'):
        plt.figure(figsize=(10, 8))

        # Plot loss
        plt.subplot(2, 1, 1)
        plt.plot(epochs, train_loss, color='#3498db', label='Training Loss')
        plt.fill_between(epochs, np.array(train_loss) - np.array(train_loss_std), 
                        np.array(train_loss) + np.array(train_loss_std), color='#3498db', alpha=0.2)
        
        if is_cv == 'on':
            plt.plot(epochs, val_loss, color='#3498db', label='Validation Loss')
            plt.fill_between(epochs, np.array(val_loss) - np.array(val_loss_std), 
                            np.array(val_loss) + np.array(val_loss_std), color='#eb984e', alpha=0.2)
        
        plt.title('Training Loss' + (' & Validation Loss' if is_cv=='on' else ''))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot accuracy
        plt.subplot(2, 1, 2)
        plt.plot(epochs, train_acc, color='#eb984e', label='Training Accuracy')
        plt.fill_between(epochs, np.array(train_acc) - np.array(train_acc_std), 
                        np.array(train_acc) + np.array(train_acc_std), color='#3498db', alpha=0.2)
        
        if is_cv == 'on':
            plt.plot(epochs, val_acc, color='#eb984e', label='Validation Accuracy')
            plt.fill_between(epochs, np.array(val_acc) - np.array(val_acc_std), 
                            np.array(val_acc) + np.array(val_acc_std), color='#eb984e', alpha=0.2)
        
        plt.title('Training Accuracy' + (' & Validation Accuracy' if is_cv=='on' else ''))
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        # plt.show()

    @staticmethod
    def plot_roc_curve(y_true, y_probs, class_name, model_name):
        """
        Plots the ROC curve for a multi-class classification model.
        
        Args:
            y_true (array-like): True labels.
            y_probs (array-like): Predicted probabilities for each class.
            model_name (str): Name of the model being evaluated.
            class_name (list): List of class names.
        """
        plt.figure(figsize=(8, 6))

        for i in range(y_probs.shape[1]):  # Iterate over each class
            fpr, tpr, _ = roc_curve(y_true == i, y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            # Replace class number with actual class name
            plt.plot(fpr, tpr, label=f'Class {class_name[i]} (AUC = {roc_auc:.2f})')
            
        plt.plot([0, 1], [0, 1], linestyle='--', color='#7f8c8d')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.grid()
        # plt.show()