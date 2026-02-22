import argparse
from datetime import date
import time
import os
import random
import shutil
import numpy as np
import pandas as pd

# Set CUBLAS environment variable for deterministic behavior
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

from models.BUI_loader import BUI_Dataset
from models.distal_myopathy_loader import Distal_Dataset
from models.attention_plot import save_attention_maps
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from torchvision.models import ResNet50_Weights
from torchvision.models import ResNet18_Weights
from models.dualstream_attention import DualStream_Attention
from validation import Model_Validation
from cross_validation import CrossValidation
from performance import Performance
from plots import Plotter

def set_seed(seed):
    # Python random
    random.seed(seed)
    
    # Numpy random
    np.random.seed(seed)
    
    # PyTorch random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    
    # PyTorch deterministic mode
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set deterministic behavior for transforms
    torch.manual_seed(seed)

torch.use_deterministic_algorithms(False)

today = date.today().strftime("%Y-%m-%d")
start = time.time()

# -------------------------------------------- Configuration --------------------------------------------
dataset = "Distal"  # "ISIC-2017" or "BUI" or "Distal"
if dataset == "BUI":
    dataset_dir = "./BUI_Dataset"
    class_name = {0: 'Benign', 1: 'Malignant'}
    full_dataset_cv = True          # Use the full dataset for cross-validation
else:
    dataset_dir = "./Distal_Dataset"
    class_name = {0: 'Healthy', 1: 'Affected'}
    full_dataset_cv = True          # Use the full dataset for cross-validation
    
img_size = 64
batch_size = 16
epochs = 2
patience_epoch = 10
num_cls = len(class_name)
backbone_model = "resnet50"
out_channels = 512 if backbone_model == "resnet18" else 2048
weights = ResNet18_Weights.IMAGENET1K_V1 if backbone_model == "resnet18" else ResNet50_Weights.IMAGENET1K_V2
cross_validation = 'on'
cv_epochs = 2                      # Number of epochs for cross-validation
n_splits = 2                       # Number of folds for cross-validation
fusion_type = "product"            # "gate," "concat," "product"

# -------------------------------------------- Argument Parser --------------------------------------------
def parse():
    parser = argparse.ArgumentParser(description="Start Training DualStream Attention")

    # Dataset argument (now optional with default)
    parser.add_argument('--dataset', type=str, default=dataset, choices=["BUI", "ISIC-2017", "Distal"],
                        help="Dataset to use: BUI, ISIC-2017, or Distal (default: Distal)")

    # Model parameters
    parser.add_argument('--input_size', type=int, nargs=2, default=[img_size, img_size],
                        help="Input size as two integers (height width)")
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--num_input_channels', type=int, default=3)
    parser.add_argument('--global_net', type=str, default=backbone_model)
    parser.add_argument('--local_net', type=str, default='bagnet33')
    parser.add_argument('--out_channels', type=int, default=out_channels)
    parser.add_argument('--hidden_dim', type=int, default=2048)             # REVISE HERE
    parser.add_argument('--epochs', type=int, default=epochs)
    parser.add_argument('--num_cls', type=int, default=num_cls)
    parser.add_argument('--fusion_type', type=str, default=fusion_type, choices=['gate', 'concat', 'product'])

    # Global branch parameters
    parser.add_argument('--global_weight', type=float, default=0.6)
    parser.add_argument('--global_optim', type=str, default='sgd')
    parser.add_argument('--global_lr', type=float, default=0.001)
    parser.add_argument('--global_weight_decay', type=float, default=0.0001)

    # Local branch parameters
    parser.add_argument('--local_weight', type=float, default=0.1)
    parser.add_argument('--local_optim', type=str, default='sgd')
    parser.add_argument('--local_lr', type=float, default=0.005)
    parser.add_argument('--local_weight_decay', type=float, default=0.0005)

    # Fusion branch parameters
    parser.add_argument('--fusion_weight', type=float, default=0.3)
    parser.add_argument('--fusion_optim', type=str, default='sgd')
    parser.add_argument('--fusion_lr', type=float, default=0.001)
    parser.add_argument('--fusion_weight_decay', type=float, default=0.0001)
    parser.add_argument('--fusion_dropout', type=float, default=0.3)

    # Save directory
    parser.add_argument('--save_dir', type=str, default='saved_models')

    args = parser.parse_args()

    # Convert input_size to tuple
    args.input_size = tuple(args.input_size)

    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Get today's date for the folder structure
    today = date.today().strftime("%Y-%m-%d")

    # Assign parameters based on dataset
    if args.dataset == "BUI":
        args.global_lr = 0.0010326391234229413
        args.fusion_lr = 0.0010895242621177385
        args.fusion_dropout = 0.25265876467678383
        args.global_weight = 0.3
        args.local_weight = 0.3
        args.fusion_weight = 0.4
        args.global_weight_decay = 0.00026178316206500884
        args.fusion_weight_decay = 0.0004399692567575375

    elif args.dataset == "Distal":
        args.global_lr = 0.004551451270627368
        args.fusion_lr = 0.0036595165300542863
        args.fusion_dropout = 0.2892480016038235
        args.global_weight = 0.3
        args.local_weight = 0.3
        args.fusion_weight = 0.4
        args.global_weight_decay = 0.0001153826582408502
        args.fusion_weight_decay = 0.0004171101381897971

    elif args.dataset == "ISIC-2017":
        args.global_lr = 0.0011342864376922249
        args.local_lr = 0.00013562347202116282
        args.fusion_lr = 0.004066198362357901
        args.fusion_dropout = 0.39157470786132015
        args.global_weight = 0.3
        args.local_weight = 0.3
        args.fusion_weight = 0.4
        args.global_weight_decay = 0.0003048255881941921
        args.local_weight_decay = 0.00018625244587133362
        args.fusion_weight_decay = 0.0001949988045311218

    # Set save directory structure: script_dir/saved_models/date/dataset/
    args.save_dir = os.path.join(script_dir, args.save_dir, today, args.dataset)
    os.makedirs(args.save_dir, exist_ok=True)

    return args

# -------------------------------------------- Training Loop --------------------------------------------
def train_epoch(model, data, criterion_global, criterion_local, criterion_fusion, optimizer_global, optimizer_local, optimizer_fusion, 
                global_w, local_w, fusion_w, device, results_dir, today):
    """ Train a single epoch and compute metrics using `Performance.compute_and_log_metrics()`. """
    model.train()
    running_loss = 0.0
    y_true, y_prob_global, y_prob_local = [], [], []
    y_prob_fusion_gate, y_prob_fusion_concat, y_prob_fusion_product = [], [], []

    for batch in data:
        if len(batch) == 4:                     # ISIC dataset (images, masks, labels, image_ids)
            images, _, labels, _ = batch        # Ignore masks and image IDs
        elif len(batch) == 3:                   # Expected for datasets without masks
            images, _, labels = batch
        elif len(batch) == 2:                   # Expected for Distal Myopathy dataset
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
            results = (model.module if isinstance(model, nn.DataParallel) else model).evaluate_all_fusion_types(images)
            g_out, l_out = results['global_output'], results['local_output']
            
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

    # Compute metrics for each fusion type
    fusion_results = {
        'gate': {'probs': y_prob_fusion_gate},
        'concat': {'probs': y_prob_fusion_concat},
        'product': {'probs': y_prob_fusion_product}
    }

    # Compute metrics using `Performance.compute_and_log_metrics()`
    metric_lists_train = {
        "Global": {"Accuracy": [], "Precision": [], "Recall": [], "F1_score": [], "AUC": []},
        "Local": {"Accuracy": [], "Precision": [], "Recall": [], "F1_score": [], "AUC": []},
        "Fusion_Gate": {"Accuracy": [], "Precision": [], "Recall": [], "F1_score": [], "AUC": []},
        "Fusion_Concat": {"Accuracy": [], "Precision": [], "Recall": [], "F1_score": [], "AUC": []},
        "Fusion_Product": {"Accuracy": [], "Precision": [], "Recall": [], "F1_score": [], "AUC": []}}
    metric_names = ["Accuracy", "Precision", "Recall", "F1_score", "AUC"]

    train_performance, _ = Performance.compute_and_log_metrics(
        y_true, 
        (y_prob_global, y_prob_local, y_prob_fusion_gate, y_prob_fusion_concat, y_prob_fusion_product),
        metric_lists_train, 
        metric_names, 
        phase=f"Training", 
        save_dir=results_dir, 
        today=today
    )

    return running_loss / len(data), train_performance, y_true, fusion_results
    
# -------------------------------------------- Main Function --------------------------------------------
def main(args, cross_validation):
    print(f'Loading {dataset} data...')

    # Clear the save directory to avoid loading old weights
    if os.path.exists(args.save_dir):
        # Safely remove existing save_dir, ignoring errors on non-empty dirs
        shutil.rmtree(args.save_dir, ignore_errors=True)
    os.makedirs(args.save_dir, exist_ok=True)

    # Set comprehensive random seeds for reproducibility
    seed = 42
    set_seed(seed)
    
    # Set deterministic behavior for transforms
    torch.manual_seed(seed)
    
    bui_transform = transforms.Compose([transforms.Resize((args.input_size[0], args.input_size[1])),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomRotation(degrees=10, fill=(0,)),
                                        transforms.ColorJitter(brightness=0.2, contrast=0.2),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5]*3, [0.5]*3)])
    
    distal_transform = transforms.Compose([transforms.Resize((args.input_size[0], args.input_size[1])),
                                           transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
                                           transforms.ColorJitter(brightness=0.2, contrast=0.2),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5]*3, [0.5]*3)])
    
    # Load dataset
    if dataset == "BUI":
        data_loader = BUI_Dataset(dataset_dir, args.batch_size, img_size, list(class_name.values()), data_split=0.8, n_splits=n_splits, 
                                 data_transform=bui_transform, use_cv=cross_validation, use_full_dataset_cv=full_dataset_cv, seed=seed)
        loaders = data_loader.prepare_dataloaders()
        train_loader = loaders["train_loader"]
        attn_loader = loaders["attn_loader"]                
        mask_loader = loaders["mask_loader"]                    
        image_names = loaders["image_names"]
        
    else:
        data_loader = Distal_Dataset(dataset_dir, img_size, args.batch_size, test_size=0.2, data_transform=distal_transform,
                                    use_cv=cross_validation, n_splits=n_splits, use_full_dataset_cv=full_dataset_cv, seed=seed)
        loaders = data_loader.prepare_dataloaders()
        train_loader = loaders["train_loader"]
        attn_loader = loaders["attn_loader"]                
        mask_loader = loaders["mask_loader"]                    
        image_names = loaders["image_names"]

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Load model
    model = DualStream_Attention(global_net=args.global_net, local_net=args.local_net, num_cls=args.num_cls, 
                                in_channels=args.num_input_channels, out_channels=args.out_channels, 
                                in_size=args.input_size, global_weight=args.global_weight, 
                                local_weight=args.local_weight, fusion_weight=args.fusion_weight, 
                                dropout=args.fusion_dropout, weights=weights, load_local=True, 
                                use_rgb=True, fusion_type="gate").to(device, memory_format=torch.channels_last)

    # Reset model weights to ensure random initialization
    model.apply(reset_weights)
    model.to(device)

    # If multiple GPUs are available, use DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    # Compile the model for optimization
    #model = torch.compile(model)

    # Define criterion and optimizer
    criterion_global = nn.CrossEntropyLoss()
    criterion_local = nn.CrossEntropyLoss()
    criterion_fusion = nn.CrossEntropyLoss()

    # Create separate parameter groups for each branch
    if isinstance(model, nn.DataParallel):
        global_params = model.module.global_branch.parameters()
        local_params = model.module.local_branch.parameters()
        fusion_params = model.module.fusion_branch.parameters()
    else:
        global_params = model.global_branch.parameters()
        local_params = model.local_branch.parameters()
        fusion_params = model.fusion_branch.parameters()

    # Create optimizers for each branch
    if args.global_optim.lower() == "sgd":
        optimizer_global = optim.SGD(global_params, lr=args.global_lr, momentum=0.9, weight_decay=args.global_weight_decay)
    elif args.global_optim.lower() == "adam":
        optimizer_global = optim.Adam(global_params, lr=args.global_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.global_weight_decay)
    else:
        optimizer_global = optim.AdamW(global_params, lr=args.global_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.global_weight_decay)

    if args.local_optim.lower() == "sgd":
        optimizer_local = optim.SGD(local_params, lr=args.local_lr, momentum=0.9, weight_decay=args.local_weight_decay)
    elif args.local_optim.lower() == "adam":
        optimizer_local = optim.Adam(local_params, lr=args.local_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.local_weight_decay)
    else:
        optimizer_local = optim.AdamW(local_params, lr=args.local_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.local_weight_decay)

    if args.fusion_optim.lower() == "sgd":
        optimizer_fusion = optim.SGD(fusion_params, lr=args.fusion_lr, momentum=0.9, weight_decay=args.fusion_weight_decay)
    elif args.fusion_optim.lower() == "adam":
        optimizer_fusion = optim.Adam(fusion_params, lr=args.fusion_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.fusion_weight_decay)
    else:
        optimizer_fusion = optim.AdamW(fusion_params, lr=args.fusion_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.fusion_weight_decay)

    # Create schedulers for each optimizer
    scheduler_global = lr_scheduler.StepLR(optimizer_global, step_size=5, gamma=0.1)
    scheduler_local = lr_scheduler.StepLR(optimizer_local, step_size=5, gamma=0.1)
    scheduler_fusion = lr_scheduler.StepLR(optimizer_fusion, step_size=5, gamma=0.1)

    os.makedirs(args.save_dir, exist_ok=True)

    scaler = GradScaler()
    print('Starting training...')

    # -------------------------------------------- Cross-Validation Training --------------------------------------------
    if cross_validation == 'on':
        cv_folds = loaders['cv_folds']
        # Pass the mask if the dataset has masks, otherwise pass None
        cv_mask_folds = loaders['cv_mask_folds'] if mask_loader is not None else None
        cv_image_names = loaders['cv_image_names']
        print(f"Total {len(cv_folds)} folds for cross-validation.")
        
        # Call the CrossValidation function
        perform_cv = CrossValidation()
        perform_cv.run_cross_validation(model, criterion_global, criterion_local, criterion_fusion, 
                                      optimizer_global, optimizer_local, optimizer_fusion, 
                                      args.global_weight, args.local_weight, args.fusion_weight, 
                                      list(class_name.values()), cv_folds, cv_mask_folds, device, 
                                      save_dir=args.save_dir, today=today, num_epochs=cv_epochs,
                                      fold_image_names=cv_image_names, mask_loader=mask_loader, img_size=img_size)

    # Terminate training if cross-validation is on
    if cross_validation == 'on':
        print("Cross-validation completed. Terminating training.")
        return

    if cross_validation == 'off':
        
        best_loss = float('inf')
        # Early stopping variables  
        patience = patience_epoch                   # Number of epochs to wait for improvement
        no_improvement_epochs = 0
        
    # ---------------------------------------------------- Training Performance ----------------------------------------------------
        train_loss_list = []
        train_metric_list = []
        train_loss_std_list, train_acc_list, train_acc_std_list = [], [], []

        train_loader = loaders['train_loader']

        model.apply(reset_weights)      # Reset model weights
        model.to(device)                # Move model to device

        # Train the model with "train epoch" function
        for epoch in range(args.epochs):
            start_time = time.time()
            print(f"Epoch {epoch+1}/{args.epochs}")

            # Train the model
            train_loss, train_performance, y_true, fusion_results = train_epoch(model, train_loader, criterion_global, criterion_local, criterion_fusion, optimizer_global, 
                                                                               optimizer_local, optimizer_fusion, args.global_weight, args.local_weight, args.fusion_weight, 
                                                                               device, args.save_dir, today)
            # Calculate training time
            epoch_time = time.time() - start_time
            print(f"Training Epoch {epoch+1} completed in {epoch_time:.2f} seconds")

            # Append the training metrics to the lists
            train_loss_list.append(train_loss)
            train_loss_std_list.append(np.std(train_loss_list))
            train_acc_list.append(train_performance['Fusion_Gate']['Accuracy'])  # Using gate as reference
            train_acc_std_list.append(np.std(train_acc_list))
            train_metric_list.append(train_performance)

            # Print training performance
            print(f"Training Loss: {train_loss:.4f}")
            print(f"Training Accuracy >> Global: {train_performance['Global']['Accuracy']:.4f} | "
                  f"Local: {train_performance['Local']['Accuracy']:.4f} | "
                  f"Gate: {train_performance['Fusion_Gate']['Accuracy']:.4f} | "
                  f"Concat: {train_performance['Fusion_Concat']['Accuracy']:.4f} | "
                  f"Product: {train_performance['Fusion_Product']['Accuracy']:.4f}")
            
            # Save best model
            if train_loss < best_loss:
                best_loss = train_loss
                no_improvement_epochs = 0
                torch.save(model.state_dict(), os.path.join(args.save_dir, f"best_model_{dataset}_[{today}].pth"))
                print('Best model saved!')
            else:
                no_improvement_epochs += 1
                print(f'No improvement for {no_improvement_epochs} epochs.')

            scheduler_global.step()  # Adjust learning rate based on loss
            scheduler_local.step()
            scheduler_fusion.step()

            # Early stopping
            if no_improvement_epochs >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        mean_train_metrics, std_train_metrics = Performance.compute_training_metrics(train_metric_list, save_dir=args.save_dir, today=today)

        print(f"\nTraining Mean Accuracy >>  Global: {mean_train_metrics['Global_Accuracy']:.4f} ± {std_train_metrics['Global_Accuracy']:.4f} | "
              f"Local: {mean_train_metrics['Local_Accuracy']:.4f} ± {std_train_metrics['Local_Accuracy']:.4f} | "
              f"Gate: {mean_train_metrics['Fusion_Gate_Accuracy']:.4f} ± {std_train_metrics['Fusion_Gate_Accuracy']:.4f} | "
              f"Concat: {mean_train_metrics['Fusion_Concat_Accuracy']:.4f} ± {std_train_metrics['Fusion_Concat_Accuracy']:.4f} | "
              f"Product: {mean_train_metrics['Fusion_Product_Accuracy']:.4f} ± {std_train_metrics['Fusion_Product_Accuracy']:.4f}")
        
        print(f"\nTraining completed in {time.time() - start:.2f} seconds")
        
        # --------------------------------------------------- Test the Model ----------------------------------------------------
        print("\nTesting the model with different fusion types...")
        
        # Create a directory for each fusion type
        fusion_types = ["gate", "concat", "product"]
        fusion_results = {}
        
        for fusion_type in fusion_types:
            print(f"\nEvaluating {fusion_type} fusion...")
            
            # Create a subdirectory for this fusion type
            fusion_save_dir = os.path.join(args.save_dir, fusion_type)
            os.makedirs(fusion_save_dir, exist_ok=True)
            
            # Create validator for this fusion type
            test_validator = Model_Validation(model, criterion_global, criterion_local, criterion_fusion, 
                                            args.global_weight, args.local_weight, args.fusion_weight, 
                                            device, fusion_save_dir, today)
            
            # Validate with current fusion type
            val_loss, y_val_true, y_val_prob_global, y_val_prob_local, y_val_prob_fusion = test_validator.validate(
                attn_loader, image_names, args.class_name, "Train_Test", eval_fusion_type=fusion_type)
            
            print(f'Validation Loss for {fusion_type}: {val_loss:.4f}')
            test_validator.metric_calculation("Validation", y_val_true, y_val_prob_global, y_val_prob_local, y_val_prob_fusion)
            
            # Save prediction probabilities
            results_df = pd.DataFrame({
                'Image_Name': image_names,
                'True_Label': [args.class_name[label] for label in y_val_true]
            })
            
            for i, cls_name in enumerate(args.class_name.values()):
                results_df[f'Global_Prob_{cls_name}'] = y_val_prob_global[:, i]
                results_df[f'Local_Prob_{cls_name}'] = y_val_prob_local[:, i]
                results_df[f'Fusion_Prob_{cls_name}'] = y_val_prob_fusion[:, i]
            
            results_df['Global_Prediction'] = [args.class_name[label] for label in np.argmax(y_val_prob_global, axis=1)]
            results_df['Local_Prediction'] = [args.class_name[label] for label in np.argmax(y_val_prob_local, axis=1)]
            results_df['Fusion_Prediction'] = [args.class_name[label] for label in np.argmax(y_val_prob_fusion, axis=1)]
            
            results_df['Status'] = results_df.apply(
                lambda row: "Candidate" if (row['True_Label'] == row['Global_Prediction'] == row['Local_Prediction'] == row['Fusion_Prediction']) 
                else "Ignore", axis=1)
            
            csv_path = os.path.join(fusion_save_dir, f"Prediction_Probabilities_{args.dataset}_{today}.csv")
            results_df.to_csv(csv_path, index=False)
            
            # Compute confusion matrices
            Performance.compute_confusion_matrices(y_val_true, y_val_prob_global, y_val_prob_local, y_val_prob_fusion, 
                                                 list(args.class_name.values()), fusion_save_dir, prefix="Validation")
            
            # Save ROC plot
            performance_plotter = Plotter()
            performance_plotter.plot_roc_curve(y_val_true, y_val_prob_fusion, list(args.class_name.values()), f"Enhanced RadFormer ({fusion_type})")
            plt.savefig(os.path.join(fusion_save_dir, f"roc_curve_{fusion_type}_{today}.png"), dpi=300)
            
            # Save attention maps
            print(f"Saving attention maps for {fusion_type}...")
            save_attention_maps(model, attn_loader, image_names, mask_loader, 
                               os.path.join(fusion_save_dir, "DualStream_Attention_Maps"), device, "Dual-stream", today, eval_fusion_type=fusion_type)
            
            # Store results for comparison
            fusion_results[fusion_type] = {
                'val_loss': val_loss,
                'y_true': y_val_true,
                'y_prob_fusion': y_val_prob_fusion
            }

        # Compare results across fusion types
        print("\nComparison of fusion types:")
        for fusion_type, results in fusion_results.items():
            print(f"\n{fusion_type} fusion:")
            print(f"Validation Loss: {results['val_loss']:.4f}")
            # Add more metrics as needed

    # Traning time
    print(f"Running completed in {time.time() - start:.2f} seconds")

def reset_weights(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

if __name__ == "__main__":
    args = parse()
    main(args, cross_validation)