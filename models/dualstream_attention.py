from __future__ import print_function, division
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from models.attention_gate import AttentionGate
from models.resnet import Resnet50
from models.bagnet import BagNet33, BagNet17, BagNet9

NETWORK_MAPPER = {
    "resnet50": Resnet50,
    "bagnet33": BagNet33,
    "bagnet17": BagNet17,
    "bagnet9": BagNet9}

class Encoder(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout, residual=True, fusion_type="gate"):
        super().__init__()
        
        # Fusing information (attention gate, concat, product)
        if fusion_type == "gate":
            self.attn = AttentionGate(d_model, d_model, hidden_dim)
        elif fusion_type == "concat":
            # Direct concatenation followed by 1x1 conv
            self.fusion = nn.Sequential(
                nn.Conv2d(d_model * 2, d_model, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout))
        elif fusion_type == "product":
            # Direct element-wise multiplication followed by 1x1 conv
            self.fusion = nn.Sequential(
                nn.Conv2d(d_model, d_model, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout))
        
        # Feed-forward network
        self.conv_ff = nn.Sequential(
            nn.Conv2d(d_model, hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv2d(hidden_dim, d_model, kernel_size=1, stride=1, padding=0))
        
        self.norm1 = nn.BatchNorm2d(d_model)
        self.norm2 = nn.BatchNorm2d(d_model)
        self.residual = residual
        self.fusion_type = fusion_type

    def forward(self, x, g):

        # Ensure spatial dimensions match
        if x.shape[-2:] != g.shape[-2:]:
            # Resize g to match x's spatial dimensions
            g = F.interpolate(g, size=x.shape[-2:], mode='bilinear', align_corners=True)

        # Apply different fusion strategies
        if self.fusion_type == "gate":
            att_output, att_map = self.attn(x, g)
        elif self.fusion_type == "concat":
            # Direct concatenation
            concat_features = torch.cat([x, g], dim=1)
            att_output = self.fusion(concat_features)
            att_map = att_output  # For visualization purposes
        elif self.fusion_type == "product":
            # Direct element-wise multiplication
            product_features = x * g
            att_output = self.fusion(product_features)
            att_map = att_output  # For visualization purposes
                
        if self.residual:
            out1 = self.norm1(x + att_output)
        else:
            out1 = self.norm1(att_output)
        
        ff_output = self.conv_ff(out1)
        
        if self.residual:
            out2 = self.norm2(out1 + ff_output)
        else:
            out2 = self.norm2(ff_output)

        return out2, att_map

class AttentionNet(nn.Module):
    def __init__(self, out_features=10, d_model=2048, hidden_dim=2048, residual=True, dropout=0.3, fusion_type="gate"):
        super().__init__()
        self.encoder = Encoder(d_model=d_model, hidden_dim=hidden_dim, residual=residual, dropout=dropout, fusion_type=fusion_type)
        # Calculate feature size dynamically in forward pass
        self.fc = None  # Will be initialized in first forward pass
        self.out_features = out_features
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, global_feature, local_feature, global_pool, local_pool):
        x, attns = self.encoder(global_feature, local_feature)
        global_pool = global_pool.view(global_pool.size(0), -1)
        local_pool = local_pool.view(local_pool.size(0), -1)
        x = x.reshape(x.size(0), -1)
        x_concat = torch.cat((global_pool, local_pool, x), dim=1)
        
        # Initialize fc layer on first forward pass if not already initialized
        if self.fc is None:
            input_features = x_concat.size(1)
            self.fc = nn.Linear(input_features, self.out_features).to(x_concat.device)
            
        out = self.fc(x_concat)
        return out, attns

class DualStream_Attention(nn.Module):
    def __init__(self, global_net="resnet50", local_net="bagnet33", num_cls=3, in_channels=3, out_channels=512, in_size=(224, 224), 
                 global_weight=0.6, local_weight=0.1, fusion_weight=0.3, dropout=0.3, weights=None, load_local=False, use_rgb=True,
                 fusion_type="gate"):
        super(DualStream_Attention, self).__init__()
        
        self.num_cls = num_cls
        self.depth = in_channels
        self.global_weight = global_weight
        self.local_weight = local_weight
        self.fusion_weight = fusion_weight
        self.dropout = dropout
        self.use_rgb = use_rgb
        self.fusion_type = fusion_type
        
        # Validation checks
        if use_rgb and in_channels != 3:
            raise ValueError("In channels must be 3 for RGB images")
        if not use_rgb:
            weights = None
        if fusion_type not in ["gate", "concat", "product"]:
            raise ValueError("fusion_type must be one of: 'gate', 'concat', 'product'")

        # Initialize global branch
        if global_net not in ["resnet18", "resnet34"]:
            self.global_branch = NETWORK_MAPPER[global_net](num_cls=num_cls, out_channels=out_channels, weights=weights)
        elif global_net == "resnet18":
            self.global_branch = NETWORK_MAPPER[global_net](num_cls=num_cls, out_channels=out_channels, weights=weights)

        # Initialize local branch
        self.local_branch = NETWORK_MAPPER[local_net](load_local=load_local, num_cls=num_cls, out_channels=out_channels)

        # Number of features for the fusion branch
        self.num_features = self.global_branch.out_channels
        
        # Initialize fusion branch for training
        self.fusion_branch = AttentionNet(d_model=self.num_features, out_features=num_cls, hidden_dim=2048, dropout=dropout, 
                                         residual=False, fusion_type=fusion_type)
        
        # Initialize fusion branches for evaluation
        self.eval_fusion_branches = nn.ModuleDict({
            "gate": AttentionNet(d_model=self.num_features, out_features=num_cls, hidden_dim=2048, dropout=dropout, 
                                residual=False, fusion_type="gate"),
            "concat": AttentionNet(d_model=self.num_features, out_features=num_cls, hidden_dim=2048, dropout=dropout, 
                                  residual=False, fusion_type="concat"),
            "product": AttentionNet(d_model=self.num_features, out_features=num_cls, hidden_dim=2048, dropout=dropout, 
                                   residual=False, fusion_type="product")
        })
        
        self.cms = None

    def forward(self, x, vis=False, eval_fusion_type=None):
        # Forward pass through global and local branches
        g_out, g_attn, g_pool = self.global_branch(x)
        l_out, l_attn, l_pool = self.local_branch(x)
        
        # During training or if no eval_fusion_type is specified, use the main fusion branch
        if eval_fusion_type is None:
            f_out, attns = self.fusion_branch(g_attn, l_attn, g_pool, l_pool)
        else:
            # During evaluation with specific fusion type
            if eval_fusion_type not in self.eval_fusion_branches:
                raise ValueError(f"Invalid fusion_type: {eval_fusion_type}. Must be one of {list(self.eval_fusion_branches.keys())}")
            f_out, attns = self.eval_fusion_branches[eval_fusion_type](g_attn, l_attn, g_pool, l_pool)

        # Return outputs and attention maps
        attn_data = {
            "g_attn": g_attn,
            "g_pool": g_pool,
            "l_attn": l_attn,
            "l_pool": l_pool,
            "attns": attns}
        
        return g_out, l_out, f_out, attn_data
    
    def evaluate_all_fusion_types(self, x):
        """Evaluate the model with all fusion types and return results"""
        # Get global and local features
        g_out, g_attn, g_pool = self.global_branch(x)
        l_out, l_attn, l_pool = self.local_branch(x)

        # Evaluate each fusion type
        results = {
            'global': {
                'output': g_out,
                'attention': g_attn
            },
            'local': {
                'output': l_out,
                'attention': l_attn
            },
            'gate': {
                'output': self.eval_fusion_branches['gate'](g_attn, l_attn, g_pool, l_pool)[0],
                'attention': self.eval_fusion_branches['gate'](g_attn, l_attn, g_pool, l_pool)[1]
            },
            'concat': {
                'output': self.eval_fusion_branches['concat'](g_attn, l_attn, g_pool, l_pool)[0],
                'attention': self.eval_fusion_branches['concat'](g_attn, l_attn, g_pool, l_pool)[1]
            },
            'product': {
                'output': self.eval_fusion_branches['product'](g_attn, l_attn, g_pool, l_pool)[0],
                'attention': self.eval_fusion_branches['product'](g_attn, l_attn, g_pool, l_pool)[1]
            }
        }
        
        return results

    def get_global_attention_map(self, x):
        _, _, _, attns = self.forward(x)
        global_attn_map = attns["g_attn"]
        return global_attn_map
    
    def get_local_attention_map(self, x):
        _, _, _, attns = self.forward(x)
        local_attn_map = attns["l_attn"]
        return local_attn_map
    
    def get_final_attention_map(self, x):
        _, _, _, attns = self.forward(x)
        final_attn_map = attns["attns"]
        return final_attn_map
    
    def plot_attention_on_image(self, image, attention_map, original_image=None, save_path=None, return_only=False):
        """Plot attention map on the original image."""
        # Ensure attention_map has the correct dimensions
        if attention_map.dim() == 2:
            attention_map = attention_map.unsqueeze(0)  # Add channel dimension
        elif attention_map.dim() == 3:
            attention_map = attention_map.unsqueeze(0)  # Add batch dimension
        elif attention_map.dim() == 4:
            attention_map = attention_map.squeeze(0)  # Remove batch dimension if present
        
        # Ensure attention_map has shape (C, H, W)
        if attention_map.dim() != 3:
            raise ValueError(f"Expected attention_map to have 3 dimensions after processing, got {attention_map.dim()}")
        
        # Average across channels if multiple channels exist
        if attention_map.size(0) > 1:
            attention_map = attention_map.mean(dim=0, keepdim=True)
        
        # Resize attention map to match input image size
        attention_map = F.interpolate(attention_map.unsqueeze(0), size=(image.shape[-2], image.shape[-1]), mode='bilinear', align_corners=True).squeeze(0)

        # Normalize with percentile clipping to remove outliers
        attention_map = attention_map.squeeze(0).detach().cpu().numpy()  # Detach before converting to numpy
        vmin, vmax = np.percentile(attention_map, (1, 99))
        attention_map = np.clip((attention_map - vmin) / (vmax - vmin + 1e-8), 0, 1)

        if return_only:
            return attention_map

        # Create heatmap
        heatmap = cv2.applyColorMap((attention_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = heatmap.astype(np.float32) / 255.0  # Normalize heatmap to 0–1

        # Get base image
        if original_image is not None:
            if isinstance(original_image, torch.Tensor):
                if original_image.dim() == 3:
                    display_image = original_image.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
                else:
                    raise ValueError(f"Expected 3D tensor for original_image, got shape {original_image.shape}")
            else:
                display_image = np.array(original_image)

            # Normalize if needed
            if display_image.max() > 1:
                display_image = display_image / 255.0
        else:
            display_image = image.permute((1, 2, 0)).cpu().numpy()
            display_image = (display_image - display_image.min()) / (display_image.max() - display_image.min() + 1e-8)

        # Resize and normalize display_image to match heatmap
        display_image = cv2.resize(display_image, (attention_map.shape[1], attention_map.shape[0]))

        # Convert to 3 channels if needed
        if display_image.ndim == 2:
            display_image = cv2.cvtColor((display_image * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif display_image.shape[2] == 1:
            display_image = cv2.cvtColor((display_image[:, :, 0] * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif display_image.shape[2] == 4:
            display_image = (display_image[:, :, :3] * 255).astype(np.uint8)
        else:
            display_image = (display_image * 255).astype(np.uint8)

        # Final safety check — ensure both are (H, W, 3)
        if display_image.shape != heatmap.shape:
            print(f"[Shape Mismatch] display_image shape: {display_image.shape}, heatmap shape: {heatmap.shape}")
            raise ValueError("display_image and heatmap must have the same shape before blending")

        # Blend heatmap and image
        display_image = display_image.astype(np.float32) / 255.0  # Normalize
        overlayed_image = cv2.addWeighted(display_image, 0.8, heatmap, 0.2, 0)
        overlayed_image = np.clip(overlayed_image, 0, 1)

        # Save or show
        if save_path:
            overlayed_image_bgr = cv2.cvtColor((overlayed_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, overlayed_image_bgr)

            # Save attention mask
        #     attention_map_mask = cv2.cvtColor((attention_map * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        #     attention_map_save_path = save_path.replace(".png", "_attention.png")
        #     cv2.imwrite(attention_map_save_path, attention_map_mask)
        # else:
            pass

        return attention_map

    def visualize_attention(self, original_image, device, save_paths=None, already_transformed=False):
        was_training = self.training
        self.eval()

        # Determine image size
        if isinstance(original_image, torch.Tensor):
            img_size = (original_image.shape[-2], original_image.shape[-1])
        elif isinstance(original_image, np.ndarray):
            img_size = (original_image.shape[0], original_image.shape[1])
            original_image = transforms.ToPILImage()(original_image)
        else:
            img_size = original_image.size  # PIL.Image

        # Prepare image for model input
        if not already_transformed:
            if isinstance(original_image, torch.Tensor):
                # Tensor already, just resize if needed
                model_input = transforms.Resize(img_size)(original_image)
                if model_input.dim() == 3:
                    model_input = model_input.unsqueeze(0)
            else:
                # Convert PIL or ndarray to tensor and resize
                transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
                model_input = transform(original_image).unsqueeze(0)
        else:
            model_input = original_image if original_image.dim() == 4 else original_image.unsqueeze(0)

        model_input = model_input.to(device)

        with torch.no_grad():
            _, _, _, attns = self.forward(model_input)
            # Plot attention maps
            global_map = self.plot_attention_on_image(model_input[0], attns["g_attn"], original_image=original_image, save_path=save_paths.get("global") if save_paths else None)
            local_map = self.plot_attention_on_image(model_input[0], attns["l_attn"], original_image=original_image, save_path=save_paths.get("local") if save_paths else None)
            # Fusion map
            fusion_map = self.plot_attention_on_image(model_input[0], attns["attns"], original_image=original_image, save_path=save_paths.get("fusion") if save_paths else None)

        self.train(was_training)
        return global_map, local_map, fusion_map

    def load_model(self, weight_file):
        self.load_state_dict(torch.load(weight_file))