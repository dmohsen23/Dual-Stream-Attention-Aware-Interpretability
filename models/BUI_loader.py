import os
import copy
import torch
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold

class BUI_Dataset:
    def __init__(self, dataset_dir, batch_size, img_size, desire_order, data_split, n_splits, data_transform, use_cv=False, 
                 use_full_dataset_cv=False, seed=42):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.desire_order = desire_order
        self.data_split = data_split            # e.g., proportion for training (train_size)
        self.n_splits = n_splits
        self.data_transform = data_transform    # Customized (training) transform
        self.use_cv = use_cv
        self.use_full_dataset_cv = use_full_dataset_cv
        self.seed = seed

    def prepare_dataloaders(self):
        dataloaders = {}
        # Load full dataset without transforms
        dataset = datasets.ImageFolder(root=self.dataset_dir, transform=None)
        dataset.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(dataset.classes)}
        # Split images and corresponding masks
        image_samples, mask_samples = self.split_masks_with_correspondence(dataset)
        
        if self.use_full_dataset_cv:
            # --- Full dataset branch (no train/test splitting) ---
            full_images = image_samples
            full_masks = mask_samples
            # Extract full image names (without extension)
            image_names = [os.path.splitext(os.path.basename(image_path))[0] for image_path, _ in full_images]
            
            # Create two pipelines: one with full (augmented) transform and one minimal for attention maps
            train_image_dataset = self.create_dataset_with_transform(dataset, full_images, self.data_transform)
            attn_image_dataset = self.create_dataset_with_transform(dataset, full_images, self._attn_transform())
            
            train_loader = DataLoader(train_image_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
            attn_loader = DataLoader(attn_image_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
            
            # Create mask loader for the full dataset
            mask_dataset = TensorDataset(torch.stack(full_masks))
            mask_loader = DataLoader(mask_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
            
            # Use consistent keys across branches
            dataloaders["train_loader"] = train_loader
            dataloaders["attn_loader"] = attn_loader
            dataloaders["mask_loader"] = mask_loader
            dataloaders["image_names"] = image_names
            
            # Create CV folds on the full dataset if enabled
            if self.use_cv:
                kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
                folds = []
                mask_folds = []
                cv_image_names = []
                full_indices = list(range(len(full_images)))
                for train_idx, val_idx in kfold.split(full_indices):
                    train_subset = [full_images[i] for i in train_idx]
                    val_subset = [full_images[i] for i in val_idx]

                    train_image_names = [os.path.splitext(os.path.basename(image_path))[0] for image_path, _ in train_subset]
                    val_image_names = [os.path.splitext(os.path.basename(image_path))[0] for image_path, _ in val_subset]
                    cv_image_names.append((train_image_names, val_image_names))
                    
                    train_dataset_cv = self.create_dataset_with_transform(dataset, train_subset, self.data_transform)
                    val_dataset_cv = self.create_dataset_with_transform(dataset, val_subset, self._val_transform())
                    original_val_dataset_cv = self.create_dataset_with_transform(dataset, val_subset, transforms.Compose([transforms.Resize((self.img_size, self.img_size)), transforms.ToTensor()]))

                    train_loader_cv = DataLoader(train_dataset_cv, batch_size=self.batch_size, shuffle=True, num_workers=2)
                    val_loader_cv = DataLoader(val_dataset_cv, batch_size=self.batch_size, shuffle=False, num_workers=2)
                    original_val_loader_cv = DataLoader(original_val_dataset_cv, batch_size=self.batch_size, shuffle=False, num_workers=2)
                    # Get corresponding mask subsets
                    train_masks_subset = [mask_samples[i] for i in train_idx]
                    val_masks_subset = [mask_samples[i] for i in val_idx]
                    
                    # Create datasets for masks
                    train_mask_dataset = TensorDataset(torch.stack(train_masks_subset))
                    val_mask_dataset = TensorDataset(torch.stack(val_masks_subset))
                    
                    # Create dataloaders
                    train_mask_loader_cv = DataLoader(train_mask_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
                    val_mask_loader_cv = DataLoader(val_mask_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
                    
                    folds.append((train_loader_cv, val_loader_cv, original_val_loader_cv))
                    mask_folds.append((train_mask_loader_cv, val_mask_loader_cv))
                
                dataloaders["cv_folds"] = folds
                dataloaders["cv_mask_folds"] = mask_folds
                dataloaders["cv_image_names"] = cv_image_names
            else:
                dataloaders["cv_folds"] = None
            
            return dataloaders
        
        else:
            # Standard branch: split dataset into train and test sets
            train_images, test_images, train_masks, test_masks = self.split_masks_into_train_val(image_samples, mask_samples)
            
            # Extract image names for train and test splits
            train_image_names = [os.path.splitext(os.path.basename(image_path))[0] for image_path, _ in train_images]
            test_image_names = [os.path.splitext(os.path.basename(image_path))[0] for image_path, _ in test_images]
            
            # Create mask loaders for train and test splits
            train_mask_dataset = TensorDataset(torch.stack(train_masks))
            train_mask_loader = DataLoader(train_mask_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
            
            test_mask_dataset = TensorDataset(torch.stack(test_masks))
            test_mask_loader = DataLoader(test_mask_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
            
            # Additionally, create loaders over the full dataset (by combining train and test)
            full_images = train_images + test_images
            full_masks = train_masks + test_masks
            full_image_names = [os.path.splitext(os.path.basename(image_path))[0] for image_path, _ in full_images]
            
            full_image_dataset = self.create_dataset_with_transform(dataset, full_images, self.data_transform)
            full_loader = DataLoader(full_image_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
            
            full_attn_dataset = self.create_dataset_with_transform(dataset, full_images, self._attn_transform())
            full_attn_loader = DataLoader(full_attn_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
            
            full_mask_dataset = TensorDataset(torch.stack(full_masks))
            full_mask_loader = DataLoader(full_mask_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
            
            # Create CV folds on the training split if enabled
            if self.use_cv:
                kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
                folds = []
                mask_folds = []
                cv_image_names = []
                train_indices = list(range(len(train_images)))
                for train_idx, val_idx in kfold.split(train_indices):
                    # Get image subsets
                    train_subset = [train_images[i] for i in train_idx]
                    val_subset = [train_images[i] for i in val_idx]

                    train_image_names = [os.path.splitext(os.path.basename(image_path))[0] for image_path, _ in train_subset]
                    val_image_names = [os.path.splitext(os.path.basename(image_path))[0] for image_path, _ in val_subset]
                    cv_image_names.append((train_image_names, val_image_names))
                    
                    # Get corresponding mask subsets
                    train_masks_subset = [train_masks[i] for i in train_idx]
                    val_masks_subset = [train_masks[i] for i in val_idx]
                    
                    # Create datasets for images
                    train_dataset_cv = self.create_dataset_with_transform(dataset, train_subset, self.data_transform)
                    val_dataset_cv = self.create_dataset_with_transform(dataset, val_subset, self._val_transform())
                    original_val_dataset_cv = self.create_dataset_with_transform(dataset, val_subset, transforms.Compose([transforms.Resize((self.img_size, self.img_size)), transforms.ToTensor()]))
                    
                    # Create datasets for masks
                    train_mask_dataset = TensorDataset(torch.stack(train_masks_subset))
                    val_mask_dataset = TensorDataset(torch.stack(val_masks_subset))
                    
                    # Create dataloaders
                    train_loader_cv = DataLoader(train_dataset_cv, batch_size=self.batch_size, shuffle=True, num_workers=2)
                    val_loader_cv = DataLoader(val_dataset_cv, batch_size=self.batch_size, shuffle=False, num_workers=2)
                    original_val_loader_cv = DataLoader(original_val_dataset_cv, batch_size=self.batch_size, shuffle=False, num_workers=2)

                    train_mask_loader_cv = DataLoader(train_mask_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
                    val_mask_loader_cv = DataLoader(val_mask_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)

                    folds.append((train_loader_cv, val_loader_cv, original_val_loader_cv))
                    mask_folds.append((train_mask_loader_cv, val_mask_loader_cv))
                
                dataloaders["cv_folds"] = folds
                dataloaders["cv_mask_folds"] = mask_folds
                dataloaders["cv_image_names"] = cv_image_names
            else:
                dataloaders["cv_folds"] = None
            
            # Return both the split loaders and the full dataset loaders
            dataloaders["train_loader"] = full_loader      # Entire dataset with training transform
            dataloaders["attn_loader"] = full_attn_loader  # Entire dataset with attention transform
            dataloaders["mask_loader"] = full_mask_loader  # Masks for entire dataset
            dataloaders["image_names"] = full_image_names
            # Also keep the split loaders if needed:
            dataloaders["train_mask_loader"] = train_mask_loader
            dataloaders["test_mask_loader"] = test_mask_loader
            dataloaders["train_image_names"] = train_image_names
            dataloaders["test_image_names"] = test_image_names
            
            return dataloaders

    def split_masks_with_correspondence(self, dataset):
        image_samples = []
        mask_samples = {}
        for path, label in dataset.samples:
            if "_mask" in path:
                base_name = os.path.splitext(os.path.basename(path))[0].split("_mask")[0]
                if base_name not in mask_samples:
                    mask_samples[base_name] = []
                mask = Image.open(path).convert("L")
                mask = self._mask_transform()(mask)
                mask_samples[base_name].append(mask)
            else:
                image_samples.append((path, label))
        # Sort images by file name
        image_samples.sort(key=lambda x: os.path.basename(x[0]))
        aligned_masks = []
        for img_path, _ in image_samples:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            if base_name not in mask_samples:
                raise ValueError(f"No masks found for image: {base_name}")
            mask_list = mask_samples[base_name]
            merged_mask = torch.max(torch.stack(mask_list, dim=0), dim=0).values if len(mask_list) > 1 else mask_list[0]
            aligned_masks.append(merged_mask)
        return image_samples, aligned_masks

    def split_masks_into_train_val(self, image_samples, mask_samples):
        indices = list(range(len(image_samples)))
        train_indices, test_indices = train_test_split(indices, train_size=self.data_split, random_state=self.seed)
        train_images = [image_samples[i] for i in train_indices]
        test_images = [image_samples[i] for i in test_indices]
        train_masks = [mask_samples[i] for i in train_indices]
        test_masks = [mask_samples[i] for i in test_indices]
        return train_images, test_images, train_masks, test_masks

    def create_dataset_with_transform(self, base_dataset, samples, transform):
        dataset_copy = copy.deepcopy(base_dataset)
        dataset_copy.samples = samples
        dataset_copy.targets = [label for _, label in samples]
        dataset_copy.transform = transform
        return dataset_copy

    def _val_transform(self):
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def _attn_transform(self):
        # Minimal transform for attention maps (no augmentation)
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def _mask_transform(self):
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])