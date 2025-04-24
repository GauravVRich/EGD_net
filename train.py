# Multiscale Feature Extractor with Edge Guided Branch and Transformer-based Feature Aggregation
# Dependencies assumed: PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from glob import glob

#  dataset
class CustomDepthEdgeDataset(Dataset):
    def __init__(self, root_dir):
        self.img_paths, self.grad_paths, self.depth_paths, self.edge_paths = [], [], [], []
        scene_dirs = ['Urban', 'Downtown', 'Pillar World']
        for scene in scene_dirs:
            scene_path = os.path.join(root_dir, scene)
            self.img_paths += sorted(glob(os.path.join(scene_path, 'Images', '*.png')))
            self.grad_paths += sorted(glob(os.path.join(scene_path, 'gradients', '*.png')))
            self.depth_paths += sorted(glob(os.path.join(scene_path, 'Depth', '*.npy')))
            self.edge_paths += sorted(glob(os.path.join(scene_path, 'edges', '*.png')))

        self.transform_img = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.transform_gray = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        grad = Image.open(self.grad_paths[idx])
        depth = torch.tensor(np.load(self.depth_paths[idx]), dtype=torch.float32).unsqueeze(0)
        edge = Image.open(self.edge_paths[idx])
        depth_resized = F.interpolate(depth.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)

        return (
            self.transform_img(img),
            self.transform_gray(grad),
            depth_resized,
            self.transform_gray(edge)
        )

from torchvision.models import mobilenet_v2

def get_truncated_mobilenetv2():
    model = mobilenet_v2(weights='IMAGENET1K_V1').features
    return nn.Sequential(*list(model.children())[:7])

class DilatedIRBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=dilation, dilation=dilation, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class EdgeCompactModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

    def forward(self, x):
        return self.encoder(x)

class CAFF(nn.Module):
    def __init__(self, d_channels, e_channels):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        fused_channels = d_channels + e_channels
        self.fc = nn.Sequential(
            nn.Linear(fused_channels, fused_channels // 16),
            nn.ReLU(),
            nn.Linear(fused_channels // 16, fused_channels),
            nn.Sigmoid()
        )

    def forward(self, d, e):
        if e.shape[2:] != d.shape[2:]:
            e = F.interpolate(e, size=d.shape[2:], mode='bilinear', align_corners=False)
        fused = torch.cat([d, e], dim=1)
        b, c, _, _ = fused.size()
        attn = self.fc(self.squeeze(fused).view(b, c)).view(b, c, 1, 1)
        return fused * attn

class TRFA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.edge_proj = nn.Conv2d(in_channels=48, out_channels=dim, kernel_size=1)  # reduce edge channels
        self.conv_reduce = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, context, edge):
        B, C, H, W = context.shape

        # project edge to same channel count as context
        edge = self.edge_proj(edge)

        context = context.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        edge = edge.flatten(2).permute(0, 2, 1)        # [B, HW, C]

        cross_attended = self.transformer(torch.cat([context, edge], dim=1))  # [B, 2*HW, C]

        fused = cross_attended[:, :H * W] + cross_attended[:, H * W:]
        fused = fused.permute(0, 2, 1).view(B, C, H, W)
        return self.conv_reduce(torch.cat([fused, fused], dim=1))


class Decoder(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(mid_channels, 1, 1)
        )

    def forward(self, x):
        return self.decode(x)

class MSFEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = get_truncated_mobilenetv2()
        self.dilated_blocks = nn.Sequential(
            DilatedIRBlock(32, 32, 1),
            DilatedIRBlock(32, 32, 2),
            DilatedIRBlock(32, 32, 3),
            DilatedIRBlock(32, 32, 1),
            DilatedIRBlock(32, 32, 2),
            DilatedIRBlock(32, 32, 3)
        )
        self.ecm = EdgeCompactModule(1, 32)
        self.caff = CAFF(d_channels=16, e_channels=32)
        self.trfa = TRFA(32)
        self.decoder = Decoder(32, 64)
        self.edge_head = nn.Conv2d(16+ 32, 1, kernel_size=1)

    def forward(self, x_img, x_grad):
        d1 = self.backbone[:2](x_img)
        d2 = self.backbone[2:4](d1)
        d3 = self.backbone[4:6](d2)
        d4 = self.backbone[6:](d3)
        d5 = self.dilated_blocks(d4)

        e1 = self.ecm(x_grad)
        #print(d1.shape, e1.shape)
        e2 = self.caff(d1, e1)
        
        # Align e2's size with d5 before sending to TRFA
        if e2.shape[2:] != d5.shape[2:]:
            e2 = F.interpolate(e2, size=d5.shape[2:], mode='bilinear', align_corners=False)

        edge_map = torch.sigmoid(self.edge_head(e2))
        trfa_out = self.trfa(d5, e2)
        depth_map = self.decoder(trfa_out)

        return depth_map, edge_map

def depth_loss(pred, target):
    # Upsample prediction to match target size
    if pred.shape[2:] != target.shape[2:]:
        pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=False)

    # Gradient approximation using finite difference
    def gradient(tensor):
        dx = torch.abs(tensor[:, :, :, 1:] - tensor[:, :, :, :-1])
        dy = torch.abs(tensor[:, :, 1:, :] - tensor[:, :, :-1, :])
        return dx, dy

    pred_dx, pred_dy = gradient(pred)
    target_dx, target_dy = gradient(target)

    grad_loss = F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)
    return F.l1_loss(pred, target) + grad_loss

def edge_loss(pred, target):
    if pred.shape[2:] != target.shape[2:]:
        pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=False)
    return F.binary_cross_entropy(pred, target)

def train(model, dataloader, epochs=10, lr=1e-4, save_path="msfe_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_depth_loss, epoch_edge_loss = 0.0, 0.0
        for x_img, x_grad, depth_gt, edge_gt in dataloader:
            x_img, x_grad, depth_gt, edge_gt = x_img.to(device), x_grad.to(device), depth_gt.to(device), edge_gt.to(device)
            pred_depth, pred_edge = model(x_img, x_grad)
            d_loss = depth_loss(pred_depth, depth_gt)
            e_loss = edge_loss(pred_edge, edge_gt)
            loss = d_loss + 0.1 * e_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_depth_loss += d_loss.item()
            epoch_edge_loss += e_loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] - Depth Loss: {epoch_depth_loss:.4f}, Edge Loss: {epoch_edge_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    dataset = CustomDepthEdgeDataset(root_dir="D:/EGDnet/dataset")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
    model = MSFEModel()
    train(model, dataloader, epochs=20, lr=1e-4, save_path="msfe_model_final.pth")
