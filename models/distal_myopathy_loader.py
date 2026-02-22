import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split, KFold
from torchvision import datasets
import numpy as np
from PIL import Image

# Dataset for loading masks
class MaskDataset(Dataset):
    def __init__(self, mask_paths, transform=None):
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.mask_paths)

    def __getitem__(self, idx):
        mask_path = self.mask_paths[idx]
        mask = Image.open(mask_path)
        if self.transform:
            mask = self.transform(mask)
        return mask

class Distal_Dataset:
    def __init__(self, data_root, img_size, batch_size, test_size, data_transform, use_cv=False, n_splits=5, use_full_dataset_cv=False, seed=42):
        self.data_root = data_root
        self.img_size = img_size
        self.batch_size = batch_size
        self.test_size = test_size
        self.data_transform = data_transform    # Training/customized transform
        self.seed = seed
        # Ensure use_cv is boolean (supporting "on" as a string)
        self.use_cv = (use_cv == 'on') if isinstance(use_cv, str) else use_cv
        self.n_splits = n_splits
        self.use_full_dataset_cv = use_full_dataset_cv

        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Load the full dataset WITHOUT transforms (base dataset)
        self.full_dataset = datasets.ImageFolder(root=data_root, transform=None)
        self.class_names = self.full_dataset.classes
        self.targets = np.array(self.full_dataset.targets)
        self.image_paths = np.array([sample[0] for sample in self.full_dataset.samples])

    def prepare_dataloaders(self):
        dataloaders = {}

        # Entire dataset indices
        all_indices = np.arange(len(self.full_dataset))

        # Entire dataset with training/customized transform
        train_data = datasets.ImageFolder(root=self.data_root, transform=self.data_transform)
        train_dataset = Subset(train_data, all_indices)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        dataloaders["train_loader"] = train_loader

        # Entire dataset with attention transform
        attn_data = datasets.ImageFolder(root=self.data_root, transform=self._attn_transform())
        attn_dataset = Subset(attn_data, all_indices)
        attn_loader = DataLoader(attn_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        dataloaders["attn_loader"] = attn_loader

        # Unique Image names for entire dataset 
        # First, get all raw names and their full paths
        name_to_paths = {}
        for i in all_indices:
            raw_name = os.path.splitext(os.path.basename(self.image_paths[i]))[0]
            if raw_name not in name_to_paths:
                name_to_paths[raw_name] = []
            name_to_paths[raw_name].append(self.image_paths[i])

        # Now create unique names ensuring we don't lose any images
        unique_names = []
        for i in all_indices:
            raw_name = os.path.splitext(os.path.basename(self.image_paths[i]))[0]
            paths = name_to_paths[raw_name]
            if len(paths) > 1:
                # If there are duplicates, use the index in the paths list
                idx = paths.index(self.image_paths[i])
                unique_names.append(f"{idx+1}_{raw_name}")
            else:
                unique_names.append(raw_name)

        dataloaders["image_names"] = unique_names

        # Mask loader for entire dataset
        mask_paths = self._get_mask_paths()
        if len(mask_paths) > 0:
            mask_dataset = MaskDataset(mask_paths, transform=self._mask_transform())
            mask_loader = DataLoader(mask_dataset, batch_size=1, shuffle=False, num_workers=2)
        else:
            mask_loader = None
        dataloaders["mask_loader"] = mask_loader

        # Cross-validation folds
        if self.use_cv:
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
            cv_folds = []
            cv_image_names = []
            # Decide which indices to use for CV:
            if self.use_full_dataset_cv:
                base_indices = all_indices
            else:
                # Use a train-test split to get training indices only for CV
                train_idx, _ = train_test_split(all_indices, test_size=self.test_size, random_state=self.seed, stratify=self.targets)
                base_indices = train_idx

            for fold_train_idx, fold_val_idx in kf.split(base_indices):
                actual_train_indices = base_indices[fold_train_idx]
                actual_val_indices = base_indices[fold_val_idx]
                train_data_cv = datasets.ImageFolder(root=self.data_root, transform=self.data_transform)
                val_data_cv = datasets.ImageFolder(root=self.data_root, transform=self._val_transform())
                original_val_dataset_cv = datasets.ImageFolder(root=self.data_root, transform=transforms.Compose([transforms.Resize((self.img_size, self.img_size)), transforms.ToTensor()]))

                # Use the same unique naming scheme as the main dataset
                train_image_names = []
                for i in actual_train_indices:
                    raw_name = os.path.splitext(os.path.basename(self.image_paths[i]))[0]
                    paths = name_to_paths[raw_name]
                    if len(paths) > 1:
                        idx = paths.index(self.image_paths[i])
                        train_image_names.append(f"{idx+1}_{raw_name}")
                    else:
                        train_image_names.append(raw_name)

                val_image_names = []
                for i in actual_val_indices:
                    raw_name = os.path.splitext(os.path.basename(self.image_paths[i]))[0]
                    paths = name_to_paths[raw_name]
                    if len(paths) > 1:
                        idx = paths.index(self.image_paths[i])
                        val_image_names.append(f"{idx+1}_{raw_name}")
                    else:
                        val_image_names.append(raw_name)

                cv_image_names.append((train_image_names, val_image_names))

                train_subset_cv = Subset(train_data_cv, actual_train_indices)
                val_subset_cv = Subset(val_data_cv, actual_val_indices)
                original_val_subset_cv = Subset(original_val_dataset_cv, actual_val_indices)

                train_loader_cv = DataLoader(train_subset_cv, batch_size=self.batch_size, shuffle=True, num_workers=2)
                val_loader_cv = DataLoader(val_subset_cv, batch_size=self.batch_size, shuffle=False, num_workers=2)
                original_val_loader_cv = DataLoader(original_val_subset_cv, batch_size=self.batch_size, shuffle=False, num_workers=2)

                cv_folds.append((train_loader_cv, val_loader_cv, original_val_loader_cv))
                
            dataloaders["cv_folds"] = cv_folds
            dataloaders["cv_image_names"] = cv_image_names
        else:
            dataloaders["cv_folds"] = None

        dataloaders["class_names"] = self.class_names
        return dataloaders

    def _val_transform(self):
        """Transform for validation (used in CV folds validation set)."""
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def _attn_transform(self):
        """Transform for attention maps (minimal augmentation)."""
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def _mask_transform(self):
        """Transform for the ground truth masks."""
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def _get_mask_paths(self):
        """Recursively search for masks in the "Binary masks" folder under data_root."""
        mask_dir = os.path.join(self.data_root, "Binary masks")
        if not os.path.isdir(mask_dir):
            return []
        mask_paths = [
            os.path.join(mask_dir, f)
            for f in os.listdir(mask_dir)
            if f.endswith('_mask.png')
        ]
        return sorted(mask_paths)