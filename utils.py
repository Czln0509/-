import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
from PIL import Image
import torch.nn.functional as F


class FlowerDataset(Dataset):
    def __init__(self, image_ids, labels, root_dir, transform=None, return_filename=False):
        self.image_ids = image_ids
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform
        self.return_filename = return_filename

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        filename = self.image_ids[idx]
        img_path = os.path.join(self.root_dir, filename)

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"警告：无法读取图像 {img_path}，错误：{e}")
            # 创建灰色图像作为备用
            image = Image.new('RGB', (300, 300), color='gray')

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        if self.return_filename:
            return image, label, filename
        else:
            return image, label


def get_transforms():
    weak_train_transform = transforms.Compose([
        transforms.RandomResizedCrop(300, scale=(0.6, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.12)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    strong_train_transform = transforms.Compose([
        transforms.RandomResizedCrop(300, scale=(0.4, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return weak_train_transform, strong_train_transform, val_transform


class MixupCutmix:
    """Mixup和Cutmix数据增强"""
    def __init__(self, mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5, switch_prob=0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        
    def __call__(self, batch):
        if np.random.rand() > self.prob:
            return batch
            
        if np.random.rand() < self.switch_prob:
            return self.mixup(batch)
        else:
            return self.cutmix(batch)
    
    def mixup(self, batch):
        images, targets = batch
        batch_size = images.size(0)
        
        # 生成lambda
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        
        # 随机排列
        indices = torch.randperm(batch_size)
        
        # Mixup图像
        mixed_images = lam * images + (1 - lam) * images[indices]
        
        # 返回混合后的数据和标签信息
        return mixed_images, targets, targets[indices], lam
    
    def cutmix(self, batch):
        images, targets = batch
        batch_size = images.size(0)
        
        # 生成lambda
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        
        # 随机排列
        indices = torch.randperm(batch_size)
        
        # 生成bbox
        W, H = images.size(2), images.size(3)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # 应用cutmix
        mixed_images = images.clone()
        mixed_images[:, :, bbx1:bbx2, bby1:bby2] = images[indices, :, bbx1:bbx2, bby1:bby2]
        
        # 调整lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        return mixed_images, targets, targets[indices], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup损失函数"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def setup_device():
    """设置训练设备"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    return device