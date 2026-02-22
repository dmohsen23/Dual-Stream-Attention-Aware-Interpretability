import argparse
from datetime import date
import time
import os
import random
import optuna
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models import ResNet18_Weights, ResNet50_Weights
from torchvision.transforms import functional as TF
import pandas as pd

# Set CUBLAS environment variable for deterministic behavior
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Custom imports
from models.BUI_loader import BUI_Dataset
from models.distal_myopathy_loader import Distal_Dataset
from models.dualstream_attention import DualStream_Attention
from models.attention_plot import save_attention_maps
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set deterministic behavior for transforms
    torch.manual_seed(seed)

torch.use_deterministic_algorithms(True)

today = date.today().strftime("%Y-%m-%d")
start_time_script = time.time()

# ---------------------------------- CONFIG ----------------------------------
dataset = "Distal"  # "BUI" or "Distal"
if dataset == "BUI":
    dataset_dir = "D:/Breast Ultrasound Images dataset/Dataset_BUSI_with_GT"
    class_name = {0: 'Benign Lesion', 1: 'Malignant Lesion'}
else:
    dataset_dir = "D:/Distal Myopathies/Data"
    class_name = {0: 'Healthy', 1: 'Affected'}

tune = 'on'                 # 'on' for hyperparameter tuning, 'off' for standard training
cv   = 'off'                # 'on' for cross-validation, 'off' for normal train/val split
full_dataset_cv = False     # Use the full dataset for cross-validation

# Training parameters
img_size       = 224
batch_size     = 16
epochs         = 60
patience_epoch = 5
num_cls        = len(class_name)
backbone_model = "resnet50"
out_channels   = 512 if backbone_model == "resnet18" else 2048
weights        = ResNet18_Weights.IMAGENET1K_V1 if backbone_model == "resnet18" else ResNet50_Weights.IMAGENET1K_V2
cv_epochs      = 1          # Number of epochs for each cross-validation fold
n_splits       = 5          # Number of folds

# ------------------------------- ARG PARSER ----------------------------------
def parse():
    parser = argparse.ArgumentParser(description="Start Training DualStream Attention")
    parser.add_argument('--input_size', default=(img_size, img_size), type=int)
    parser.add_argument('--batch_size', default=batch_size, type=int)
    parser.add_argument('--num_input_channels', default=3, type=int)
    parser.add_argument('--global_net', default=backbone_model, type=str)
    parser.add_argument('--local_net', default='bagnet33', type=str)
    parser.add_argument('--out_channels', default=out_channels, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--epochs', default=epochs, type=int)
    parser.add_argument('--num_cls', default=num_cls, type=int)
    parser.add_argument('--optim', default='adam', type=str)
    parser.add_argument('--global_lr', default=0.001, type=float, help='LR for the global branch')
    parser.add_argument('--local_lr', default=0.01, type=float, help='LR for the local branch')
    parser.add_argument('--fusion_lr', default=0.001, type=float, help='LR for the fusion branch')
    parser.add_argument('--global_weight', default=0.3, type=float)
    parser.add_argument('--local_weight', default=0.3, type=float)
    parser.add_argument('--fusion_weight', default=0.4, type=float)
    parser.add_argument('--dropout', default=0.0, type=float, help='Dropout rate')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay for optimizer')
    parser.add_argument('--save_dir', default='./checkpoints', type=str)
    parser.add_argument('--save_name', default='dualstream_attention', type=str)
    args = parser.parse_args(args=[])

    # Create base folder and delete if exists
    base_folder = os.path.join(args.save_dir, today, dataset)
    if os.path.exists(base_folder):
        shutil.rmtree(base_folder)
    os.makedirs(base_folder, exist_ok=True)

    args.base_dir = base_folder
    return args

# ---------------------------- PRINT HYPERPARAMS -----------------------------
def print_hyperparameters(params, phase):
    print(f"\n{'='*20} {phase} HYPERPARAMETERS {'='*20}")
    for key, value in params.items():
        print(f"{key}: {value}")
    print('='*60)

# -------------------------- DEFINE HYPERPARAMETERS --------------------------
def define_hyperparameters(trial):
    """Define hyperparameters to optimize with Optuna."""
    args = argparse.Namespace()
    # Fixed parameters (short runs for CV)
    args.input_size = (img_size, img_size)
    args.num_input_channels = 3
    args.global_net = backbone_model
    args.epochs = 60
    args.local_net = 'bagnet33'
    args.out_channels = out_channels
    args.hidden_dim = 128
    args.num_cls = num_cls
    args.batch_size = 16
    args.global_optim = 'sgd'
    args.local_optim = 'sgd'
    args.local_lr = 0.005
    args.local_weight_decay = 0.0005
    args.fusion_optim = 'sgd'
    args.save_dir = './checkpoints'
    args.save_name = 'dualstream_attention'

    # Hyperparameters to optimize
    # 1. Learning Rates - adjust search space based on previous findings
    args.global_lr = trial.suggest_float('global_lr', 1e-4, 0.1, log=True)  # Narrower range
    # args.local_lr = trial.suggest_float('local_lr', 1e-4, 0.1, log=True)   # Lower range for local
    args.fusion_lr = trial.suggest_float('fusion_lr', 1e-4, 0.1, log=True) # Lower range for fusion

    # 2. Architecture and Training
    # args.batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    args.fusion_dropout = trial.suggest_float('fusion_dropout', 0.2, 0.5)  # Increased minimum dropout
    
    # 3. Branch Weights - suggest raw weights first
    raw_global_weight = trial.suggest_float('global_weight', 0.3, 0.5)
    raw_local_weight = trial.suggest_float('local_weight', 0.2, 0.4)
    raw_fusion_weight = trial.suggest_float('fusion_weight', 0.3, 0.5)
    
    # Normalize weights to ensure sum equals 1
    total = raw_global_weight + raw_local_weight + raw_fusion_weight
    args.global_weight = raw_global_weight / total
    args.local_weight = raw_local_weight / total
    args.fusion_weight = raw_fusion_weight / total

    # 4. Optimizer settings for each branch
    # args.global_optim = trial.suggest_categorical('global_optim', ['adamw', 'adam', 'sgd'])
    # args.local_optim = trial.suggest_categorical('local_optim', ['sgd', 'adam', 'adamw'])
    # args.fusion_optim = trial.suggest_categorical('fusion_optim', ['adamw', 'adam', 'sgd'])
    
    # 5. Weight decay for each branch
    args.global_weight_decay = trial.suggest_float('global_weight_decay', 1e-4, 1e-3, log=True)
    # args.local_weight_decay = trial.suggest_float('local_weight_decay', 1e-4, 1e-3, log=True)
    args.fusion_weight_decay = trial.suggest_float('fusion_weight_decay', 1e-4, 1e-3, log=True)

    return args

# ------------------------------ OBJECTIVE FUNC ------------------------------
def objective(trial, cross_validation):
    """Objective: return the mean CV accuracy of the current hyperparam set."""
    args = define_hyperparameters(trial)

    # We only do cross_validation for the objective => final_run=False
    mean_cv_acc = main(args, cross_validation=cross_validation, final_run=False)
    return mean_cv_acc

# --------------------- RUN HYPERPARAM OPTIMIZATION -------------------------
def run_hyperparameter_optimization(args, cross_validation):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, cross_validation), n_trials=30)

    best_trial = study.best_trial
    print("Best trial value (mean CV accuracy):", best_trial.value)
    print("Best hyperparameters:")
    for k, v in best_trial.params.items():
        print(f"  {k}: {v}")
    
    # Print additional optimization statistics
    print("\nOptimization Statistics:")
    print(f"Number of completed trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Number of completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    
    return best_trial

# ----------------------------- TRAIN EPOCH ----------------------------------
def train_epoch(model, data, criterion_global, criterion_local, criterion_fusion, optimizer_global, optimizer_local, optimizer_fusion, 
                global_w, local_w, fusion_w, device, results_dir, today, scaler):
    model.train()
    running_loss = 0.0
    y_true, y_prob_global, y_prob_local, y_prob_fusion = [], [], [], []

    for batch in data:
        if len(batch) == 4:
            images, _, labels, _ = batch
        elif len(batch) == 3:
            images, _, labels = batch
        elif len(batch) == 2:
            images, labels = batch
        else:
            raise ValueError(f"Unexpected batch format: got {len(batch)} elements.")

        images, labels = images.to(device, memory_format=torch.channels_last), labels.to(device)
        
        # Zero gradients for all optimizers
        optimizer_global.zero_grad()
        optimizer_local.zero_grad()
        optimizer_fusion.zero_grad()

        with autocast(device_type='cuda', dtype=torch.bfloat16):
            g_out, l_out, f_out, _ = model(images)
            loss_global = criterion_global(g_out, labels)
            loss_local  = criterion_local(l_out, labels)
            loss_fusion = criterion_fusion(f_out, labels)
            loss = (global_w * loss_global + local_w * loss_local + fusion_w * loss_fusion)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Step optimizers with gradient scaling
        scaler.step(optimizer_global)
        scaler.step(optimizer_local)
        scaler.step(optimizer_fusion)
        scaler.update()

        running_loss += loss.item()

        y_true.extend(labels.cpu().numpy())
        y_prob_global.extend(F.softmax(g_out, dim=1).detach().cpu().to(torch.float32).numpy())
        y_prob_local.extend(F.softmax(l_out, dim=1).detach().cpu().to(torch.float32).numpy())
        y_prob_fusion.extend(F.softmax(f_out, dim=1).detach().cpu().to(torch.float32).numpy())

    y_true         = np.array(y_true)
    y_prob_global  = np.array(y_prob_global)
    y_prob_local   = np.array(y_prob_local)
    y_prob_fusion  = np.array(y_prob_fusion)

    metric_lists = {
        "Global": {"Accuracy": [], "Precision": [], "Recall": [], "F1_score": [], "AUC": []},
        "Local":  {"Accuracy": [], "Precision": [], "Recall": [], "F1_score": [], "AUC": []},
        "Fusion": {"Accuracy": [], "Precision": [], "Recall": [], "F1_score": [], "AUC": []}}
    metric_names = ["Accuracy", "Precision", "Recall", "F1_score", "AUC"]

    train_perf, _ = Performance.compute_and_log_metrics(
        y_true, (y_prob_global, y_prob_local, y_prob_fusion),
        metric_lists, metric_names,
        phase="Training", save_dir=results_dir, today=today)

    return running_loss / len(data), train_perf, y_true, y_prob_fusion

# --------------------------------- MAIN -----------------------------------
def main(args, cross_validation='off', final_run=False):
    """
    If cross_validation == 'on':
        - Perform cross-validation only; return mean CV accuracy.
    If cross_validation == 'off':
        - Train on the full dataset. If final_run==True, also do final plots, etc.
    """
    print(f"Loading {dataset} data...")
    print_hyperparameters(vars(args), phase="USING")

    # Create a unique run folder each time main() is called
    if not hasattr(args, 'base_dir'):
        args.base_dir = os.path.join(args.save_dir, today, dataset)
        os.makedirs(args.base_dir, exist_ok=True)

    timestamp = time.strftime("%H%M%S")
    run_dir   = os.path.join(args.base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    args.save_dir = run_dir

    # Set comprehensive random seeds for reproducibility
    seed = 42
    set_seed(seed)  # Use our comprehensive seed setting function

    # Set deterministic behavior for transforms
    torch.manual_seed(seed)
    
    # Base medical transform with more robust augmentation    
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
        data_loader = BUI_Dataset(dataset_dir, args.batch_size, img_size, list(class_name.values()), data_split=0.8, 
                                 n_splits=n_splits, data_transform=bui_transform, use_cv=cross_validation, use_full_dataset_cv=full_dataset_cv, 
                                 seed=seed)
        loaders = data_loader.prepare_dataloaders()
        train_loader = loaders["train_loader"]
        attn_loader = loaders["attn_loader"]                
        mask_loader = loaders["mask_loader"]                    
        image_names = loaders["image_names"]
        test_loader = loaders["test_loader"]
        test_image_names = loaders["test_image_names"]
        test_mask_loader = loaders["test_mask_loader"]
        
    else:  # Distal
        data_loader = Distal_Dataset(dataset_dir, img_size, args.batch_size, test_size=0.2, data_transform=distal_transform,
                                    use_cv=cross_validation, n_splits=n_splits, use_full_dataset_cv=full_dataset_cv, seed=seed)
        loaders = data_loader.prepare_dataloaders()
        train_loader = loaders["train_loader"]
        attn_loader = loaders["attn_loader"]                
        mask_loader = loaders["mask_loader"]                    
        image_names = loaders["image_names"]
        test_loader = loaders["test_loader"]
        test_image_names = loaders["test_image_names"]
        test_mask_loader = loaders["test_mask_loader"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Initialize model
    model = DualStream_Attention(global_net=args.global_net, local_net=args.local_net, num_cls=args.num_cls, 
                            in_channels=args.num_input_channels, out_channels=args.out_channels,
                            in_size=args.input_size, global_weight=args.global_weight, 
                            local_weight=args.local_weight, fusion_weight=args.fusion_weight, 
                            dropout=args.fusion_dropout, weights=weights, load_local=True, use_rgb=True)
    
    model.apply(reset_weights)
    model = model.to(device, memory_format=torch.channels_last)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
        model = model.to(device, memory_format=torch.channels_last)

    # Compile the model for optimization
    try:
        model = torch.compile(model)
        print("Model compiled successfully")
    except Exception as e:
        print(f"Model compilation failed: {e}")
        print("Proceeding with uncompiled model")

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

    # ------------------- CROSS-VALIDATION BRANCH -------------------
    if cross_validation == 'on':
        """
        1) Perform cross-validation only.
        2) Return the mean cross-validation accuracy for the current hyperparam set.
        3) Do NOT train on the entire dataset here.
        """
        cv_folds = loaders['cv_folds']  # a list of (train_loader, val_loader)
        print(f"[CV] Found {len(cv_folds)} folds. Running for {cv_epochs} epochs each fold...")

        # We have an external CrossValidation utility
        perform_cv = CrossValidation()
        mean_cv_acc = perform_cv.run_cross_validation(
            model=model,
            criterion_global=criterion_global,
            criterion_local=criterion_local,
            criterion_fusion=criterion_fusion,
            optimizer_global=optimizer_global,
            optimizer_local=optimizer_local,
            optimizer_fusion=optimizer_fusion,
            global_w=args.global_weight,
            local_w=args.local_weight,
            fusion_w=args.fusion_weight,
            class_name=list(class_name.values()),
            cv_folds=cv_folds,
            device=device,
            save_dir=args.save_dir,
            today=today,
            num_epochs=cv_epochs,
            image_names=image_names,   # Add image names for attention maps
            mask_loader=mask_loader,   # Add mask loader for attention maps
            img_size=img_size          # Add image size for attention maps
        )

        print(f"\nCross-validation completed. Mean CV accuracy: {mean_cv_acc:.4f}")
        return mean_cv_acc  # Return mean CV accuracy for hyperparameter optimization

    # --------------------- FULL-DATASET TRAINING -------------------
    else:
        """
        If cross_validation == 'off', we do standard training on the full dataset.
        If final_run == True, we also do the final test set evaluation, attention map saving, etc.
        """
        print("Training on full dataset ...")

        best_loss = float('inf')
        patience = patience_epoch
        no_improvement_epochs = 0

        train_loss_list      = []
        train_metric_list    = []
        train_loss_std_list  = []
        train_acc_list       = []
        train_acc_std_list   = []

        # Ensure we re-init weights before training on the entire dataset
        model.apply(reset_weights)
        model.to(device)

        for epoch in range(args.epochs):
            e_start = time.time()
            print(f"Epoch {epoch+1}/{args.epochs}")

            # Train the model
            train_loss, train_performance, y_true, y_prob_fusion = train_epoch(model, train_loader, criterion_global, criterion_local, criterion_fusion, optimizer_global, 
                                                                               optimizer_local, optimizer_fusion, args.global_weight, args.local_weight, args.fusion_weight, 
                                                                               device, args.save_dir, today, scaler=scaler)
            
            e_time = time.time() - e_start
            print(f"Epoch {epoch+1} training took {e_time:.2f} sec")

            train_loss_list.append(train_loss)
            train_loss_std_list.append(np.std(train_loss_list))

            train_acc_list.append(train_performance['Fusion']['Accuracy'])
            train_acc_std_list.append(np.std(train_acc_list))

            train_metric_list.append(train_performance)

            print(f"Training Loss: {train_loss:.4f}")
            print(f"Accuracy >> Global: {train_performance['Global']['Accuracy']:.4f} | "
                  f"Local: {train_performance['Local']['Accuracy']:.4f} | "
                  f"Fusion: {train_performance['Fusion']['Accuracy']:.4f}")

            if train_loss < best_loss:
                best_loss = train_loss
                no_improvement_epochs = 0
                # Save best model
                model_save_path = os.path.join(args.save_dir, f"best_model_{dataset}_{today}.pth")
                torch.save(model.state_dict(), model_save_path)
                print("Best model saved!")
            else:
                no_improvement_epochs += 1
                print(f"No improvement for {no_improvement_epochs} epoch(s).")

            # Step all schedulers
            scheduler_global.step()
            scheduler_local.step()
            scheduler_fusion.step()

            if no_improvement_epochs >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        mean_train_metrics, std_train_metrics = Performance.compute_training_metrics(
            train_metric_list, save_dir=args.save_dir, today=today
        )
        fusion_acc = mean_train_metrics['Fusion_Accuracy']

        # Train Validator
        if full_dataset_cv:
            train_validator = Model_Validation(model, criterion_global, criterion_local, criterion_fusion, args.global_weight, args.local_weight, args.fusion_weight, device, args.save_dir, today)
            train_validator.validate(train_loader, image_names, class_name, "Training")

        print(f"\nFinal Train mean (Fusion) Accuracy: {fusion_acc:.4f} ± {std_train_metrics['Fusion_Accuracy']:.4f}")
        print(f"Full-dataset training completed in {time.time() - start_time_script:.2f} sec")
        
        if final_run:
            if full_dataset_cv:
                print("Full dataset mode enabled. Skipping the testing phase.")
            else:
                print("\n[FINAL RUN] Testing on the test set ...")
                validator = Model_Validation(model, criterion_global, criterion_local, criterion_fusion,
                                            args.global_weight, args.local_weight, args.fusion_weight,
                                            device, args.save_dir, today)
                val_loss, y_val_true, y_val_prob_global, y_val_prob_local, y_val_prob_fusion = validator.validate(test_loader, test_image_names, class_name)
                print(f"[FINAL RUN] Validation Loss: {val_loss:.4f}")
                validator.metric_calculation("Validation", y_val_true,
                                            y_val_prob_global,
                                            y_val_prob_local,
                                            y_val_prob_fusion)
                
                try:
                    # Save prediction probabilities to CSV
                    # Create DataFrame with image names and true labels
                    results_df = pd.DataFrame({
                        'Image_Name': test_image_names,
                        'True_Label': [class_name[label] for label in y_val_true]
                    })
                    
                    # Add global branch probabilities
                    for i, cls_name in enumerate(class_name.values()):
                        results_df[f'Global_Prob_{cls_name}'] = y_val_prob_global[:, i]
                    
                    # Add local branch probabilities
                    for i, cls_name in enumerate(class_name.values()):
                        results_df[f'Local_Prob_{cls_name}'] = y_val_prob_local[:, i]
                    
                    # Add fusion branch probabilities
                    for i, cls_name in enumerate(class_name.values()):
                        results_df[f'Fusion_Prob_{cls_name}'] = y_val_prob_fusion[:, i]
                    
                    # Add predicted labels for each branch
                    results_df['Global_Prediction'] = [class_name[label] for label in np.argmax(y_val_prob_global, axis=1)]
                    results_df['Local_Prediction'] = [class_name[label] for label in np.argmax(y_val_prob_local, axis=1)]
                    results_df['Fusion_Prediction'] = [class_name[label] for label in np.argmax(y_val_prob_fusion, axis=1)]

                    # Add a final column as "Status" to indicate "Candidate" if the prediction is the same for all branches, and "Ignore" if the prediction is different for all branches
                    results_df['Status'] = results_df.apply(
                        lambda row: "Candidate" if (row['True_Label'] == row['Global_Prediction'] == row['Local_Prediction'] == row['Fusion_Prediction']) 
                        else "Ignore", axis=1
                    )
                    
                    # Save to CSV
                    csv_path = os.path.join(args.save_dir, f"Prediction_Probabilities_{dataset}_{today}.csv")
                    results_df.to_csv(csv_path, index=False)
                    
                except Exception as e:
                    print(f"Error saving prediction probabilities to CSV: {str(e)}")
                    print("Debug information:")
                    print(f"Number of image names: {len(test_image_names)}")
                    print(f"Shape of y_val_true: {y_val_true.shape}")
                    print(f"Shape of y_val_prob_global: {y_val_prob_global.shape}")
                    print(f"Shape of y_val_prob_local: {y_val_prob_local.shape}")
                    print(f"Shape of y_val_prob_fusion: {y_val_prob_fusion.shape}")
                    print("Continuing with the rest of the program...")
                
                # Compute confusion matrices
                Performance.compute_confusion_matrices(y_val_true, y_val_prob_global, y_val_prob_local, y_val_prob_fusion,
                                                        list(class_name.values()), args.save_dir, prefix="Final")

            # Plot training performance
            plotter = Plotter()
            plotter.plot_performance(range(1, epoch+2),
                                     train_loss_list, train_loss_std_list,
                                     train_acc_list,  train_acc_std_list,
                                     is_cv=False)
            plt.savefig(os.path.join(args.save_dir, f"{args.save_name}_performance_{today}.png"), dpi=300)

            # Plot ROC
            plotter.plot_roc_curve(y_true, y_prob_fusion, list(class_name.values()), "DualStream_Attention")
            plt.savefig(os.path.join(args.save_dir, f"{args.save_name}_roc_{today}.png"), dpi=300)
            plt.close()

            # --------------------------------------------------- Save Attention Maps ----------------------------------------------------
            if full_dataset_cv:
                used_data = attn_loader
                image_names = image_names
                image_masks = mask_loader
            else:
                used_data = test_loader
                image_names = test_image_names
                image_masks = test_mask_loader

            print("Saving attention maps...")
            # Create directories for attention maps
            attention_dirs = {
                "Dual-stream": os.path.join(args.save_dir, "DualStream_Attention_Maps"),
                "Global": os.path.join(args.save_dir, "Global_Attention_Maps"),
                "Local": os.path.join(args.save_dir, "Local_Attention_Maps")
            }
            
            for mode, save_dir in attention_dirs.items():
                print(f"Saving {mode} attention maps...")
                save_attention_maps(model, used_data, image_names, image_masks, save_dir, device, img_size, mode, today)

        # Return the final (Fusion) accuracy from training on the full dataset
        return fusion_acc

# ----------------------------- RESET WEIGHTS -------------------------------
def reset_weights(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

# ------------------------------- DRIVER ------------------------------------
if __name__ == "__main__":
    args = parse()

    if tune == 'on':
        print("Running hyperparameter tuning with cross-validation for each trial ...")
        best_trial = run_hyperparameter_optimization(args, cross_validation='on')

        print("\n\n[Optuna] Best hyperparameters found:", best_trial.params)

        # Create new args with best hyperparameters
        best_args = define_hyperparameters(best_trial)
        best_args.epochs = 60  # Set epochs for final training
        best_args.save_dir = args.save_dir  # Preserve the save directory
        best_args.base_dir = args.base_dir  # Preserve the base directory

        print("\nStarting final training with best hyperparameters...")
        final_acc = main(best_args, cross_validation='off', final_run=True)

        # Save the best hyperparams + final result
        best_param_file = os.path.join(best_args.save_dir, "best_hyperparams.txt")
        with open(best_param_file, "w") as f:
            f.write(f"Mean CV Accuracy of best trial: {best_trial.value}\n")
            f.write("Best Hyperparameters:\n")
            for k, v in best_trial.params.items():
                f.write(f"  {k}: {v}\n")
            f.write(f"\nFinal Full-Dataset Accuracy (Fusion): {final_acc:.4f}\n")
    else:
        # No hyperparam tuning; just do CV or normal training based on `cv`.
        final_acc = main(args, cross_validation=cv, final_run=True)
        print(f"Finished run with final accuracy: {final_acc:.4f}")