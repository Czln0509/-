import os
import json
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import random
from torchvision import transforms
from model import get_model, LabelSmoothingCrossEntropy
from utils import FlowerDataset, get_transforms, setup_device, MixupCutmix, mixup_criterion

# 兼容性导入 - 处理PyTorch版本差异
try:
    from torch.amp import GradScaler, autocast
    def get_autocast_context():
        return autocast('cuda')
    def get_grad_scaler():
        return GradScaler('cuda')
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    def get_autocast_context():
        return autocast()
    def get_grad_scaler():
        return GradScaler()

def set_seed(seed=42):
    """设置随机种子确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def convert_to_serializable(obj):
    """将对象转换为可JSON序列化格式"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, set):
        return list(obj)
    else:
        return obj


def validate_model(model, val_loader, criterion, device):
    """模型验证函数"""
    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels, _ in tqdm(val_loader, desc='验证'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = val_loss / len(val_loader.dataset)
    val_acc = (np.array(all_preds) == np.array(all_labels)).mean()
    val_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    val_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return val_loss, val_acc, val_precision, val_recall, val_f1


def load_split_data(config):
    """
    加载划分后的训练集与测试集
    """
    # 1. 加载划分后的标签文件
    train_label_df = pd.read_csv(config['train_split_label_path'])
    test_label_df = pd.read_csv(config['test_split_label_path'])
    print(f"训练集标签数：{len(train_label_df)}，测试集标签数：{len(test_label_df)}")

    # 2. 校验图像文件存在性
    def validate_data(df, img_dir, data_type):
        valid_filenames = []
        valid_labels = []
        missing_files = []
        augmentation_suffixes = ['_vflip', '_hflip', '_jp', '_tfl', '_sfl', '_dl']

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f'校验{data_type}集图像'):
            base_filename = row['filename']
            base_name = os.path.splitext(base_filename)[0]

            img_path = os.path.join(img_dir, base_filename)
            found = False

            if os.path.exists(img_path):
                valid_filenames.append(base_filename)
                valid_labels.append(row['category_id'])
                found = True
            else:
                for suffix in augmentation_suffixes:
                    augmented_filename = f"{base_name}{suffix}.jpg"
                    augmented_path = os.path.join(img_dir, augmented_filename)
                    if os.path.exists(augmented_path):
                        valid_filenames.append(augmented_filename)
                        valid_labels.append(row['category_id'])
                        found = True
                        break

            if not found:
                missing_files.append(base_filename)

        print(f"{data_type}集有效图像数：{len(valid_filenames)}，缺失图像数：{len(missing_files)}")
        if len(missing_files) > 0:
            print(f"{data_type}集前5个缺失文件：{missing_files[:5]}")
        return valid_filenames, valid_labels

    # 校验训练集与测试集
    train_filenames, train_labels = validate_data(
        train_label_df, config['train_split_img_dir'], '训练'
    )
    test_filenames, test_labels = validate_data(
        test_label_df, config['test_split_img_dir'], '测试'
    )

    # 3. 构建类别映射（仅使用训练集类别）
    train_categories = sorted(set(train_labels))
    label_mapping = {int(cat): idx for idx, cat in enumerate(train_categories)}
    num_classes = len(train_categories)
    print(f"总类别数：{num_classes}，类别映射示例：{dict(list(label_mapping.items())[:3])}")

    # 4. 构建植物信息字典
    plant_info = {}
    for _, row in train_label_df.iterrows():
        category_id = int(row['category_id'])
        if category_id not in plant_info:
            plant_info[category_id] = {
                'chinese_name': row.get('chinese_name', '未知'),
                'english_name': row.get('english_name', 'Unknown')
            }

    # 5. 转换标签为模型输入格式
    train_labels_idx = [label_mapping[cat] for cat in train_labels]

    # 过滤测试集中训练集未出现的类别
    test_labels_idx = []
    valid_test_indices = []
    for i, cat in enumerate(test_labels):
        if cat in label_mapping:
            test_labels_idx.append(label_mapping[cat])
            valid_test_indices.append(i)

    if len(valid_test_indices) < len(test_filenames):
        test_filenames = [test_filenames[i] for i in valid_test_indices]
        print(f"过滤后测试集样本数: {len(test_filenames)}")

    # 6. 获取数据变换
    weak_transform, strong_transform, test_transform = get_transforms()

    # 7. 创建FlowerDataset实例
    init_transform = strong_transform if config.get('use_strong_aug', True) else weak_transform
    train_dataset = FlowerDataset(
        image_ids=train_filenames,
        labels=train_labels_idx,
        root_dir=config['train_split_img_dir'],
        transform=init_transform,
        return_filename=True
    )
    test_dataset = FlowerDataset(
        image_ids=test_filenames,
        labels=test_labels_idx,
        root_dir=config['test_split_img_dir'],
        transform=test_transform,
        return_filename=True
    )

    return train_dataset, test_dataset, label_mapping, plant_info, num_classes, weak_transform, strong_transform


def train(config):
    """优化版训练主函数"""
    # 初始化设置
    set_seed(config.get('random_seed', 42))
    os.makedirs('model', exist_ok=True)

    # 加载设备
    device = setup_device()
    print(f"使用设备：{device}")

    # 1. 加载数据集
    train_dataset, test_dataset, label_mapping, plant_info, num_classes, weak_transform, strong_transform = load_split_data(config)

    # 2. 保存配置文件
    config_data = {
        'label_mapping': {str(k): v for k, v in label_mapping.items()},
        'plant_info': convert_to_serializable(plant_info),
        'num_classes': num_classes,
        'unique_categories': sorted(label_mapping.keys())
    }
    with open('model/config.json', 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=4, ensure_ascii=False)
    print("配置文件已保存至 model/config.json")

    # 3. 创建数据加载器 - 加速优化
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', True),
        persistent_workers=config.get('num_workers', 0) > 0,  # 持久化worker
        prefetch_factor=2 if config.get('num_workers', 0) > 0 else 2  # 预取因子
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', True),
        persistent_workers=config.get('num_workers', 0) > 0,
        prefetch_factor=2 if config.get('num_workers', 0) > 0 else 2
    )
    print(f"训练集批次数量：{len(train_loader)}，测试集批次数量：{len(test_loader)}")

    # 4. 初始化模型、损失函数、优化器
    model = get_model(num_classes, pretrained=config['pretrained'])
    model = model.to(device)
    
    # 模型编译加速（PyTorch 2.0+）
    if config.get('compile_model', False):
        try:
            model = torch.compile(model)
            print("模型编译成功，训练速度将显著提升")
        except Exception as e:
            print(f"模型编译失败（可能是PyTorch版本较低）: {e}")
    
    # 混合精度训练设置
    scaler = None
    if config.get('mixed_precision', False):
        try:
            scaler = get_grad_scaler()
            print("启用混合精度训练，显存使用减少50%，速度提升30%")
        except Exception as e:
            print(f"混合精度训练不可用: {e}")

    # 使用标签平滑损失函数
    criterion = LabelSmoothingCrossEntropy(smoothing=config.get('label_smoothing', 0.1))
    print(f"使用标签平滑交叉熵损失, smoothing={config.get('label_smoothing', 0.1)}")

    # 使用AdamW优化器
    optimizer = optim.AdamW(
        [
            {'params': list(model.features.parameters()), 'lr': config.get('backbone_lr', config['learning_rate'] * 0.5)},
            {'params': list(model.classifier.parameters()), 'lr': config.get('head_lr', config['learning_rate'])}
        ],
        weight_decay=config.get('weight_decay', 0.01)
    )

    # 使用更好的学习率调度器 - 组合策略
    scheduler1 = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # 每10个epoch重启
        T_mult=2,  # 重启周期翻倍
        eta_min=config.get('min_lr', 1e-6)
    )
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3  #学习率容忍
    )
    use_plateau = config.get('use_plateau_scheduler', True)

    # 初始化Mixup/Cutmix增强
    mixup_cutmix = MixupCutmix(
        mixup_alpha=config.get('mixup_alpha', 0.2),
        cutmix_alpha=config.get('cutmix_alpha', 1.0),
        prob=config.get('mixup_prob', 0.5)
    )
    print(f"启用Mixup/Cutmix增强，概率: {config.get('mixup_prob', 0.5)}")

    # 5. 训练循环
    best_test_acc = 0.0
    best_test_f1 = 0.0
    patience_counter = 0

    print("=" * 60)
    print("开始训练 - 反过拟合优化版")
    print("=" * 60)

    for epoch in range(config['num_epochs']):
        epoch_start = time.time()
        if config.get('use_strong_aug', True) and (epoch + 1) == config.get('strong_to_weak_epoch', 20):
            train_dataset.transform = weak_transform
        decay_start = config.get('mixup_decay_start', config.get('strong_to_weak_epoch', 20))
        decay_end = config.get('mixup_decay_end', decay_start + 10)
        if (epoch + 1) >= decay_start:
            t = min(1.0, (epoch + 1 - decay_start) / max(1, (decay_end - decay_start)))
            start_prob = config.get('mixup_prob', 0.5)
            final_prob = config.get('mixup_prob_final', 0.0)
            mixup_cutmix.prob = start_prob * (1 - t) + final_prob * t
        if (epoch + 1) == config.get('unfreeze_epoch', 10):
            for child in model.features.children():
                for p in child.parameters():
                    p.requires_grad = True

        # ---------------------- 训练阶段 ----------------------
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_preds, train_labels_list = [], []
        
        # 梯度累积设置
        accumulation_steps = config.get('gradient_accumulation_steps', 1)
        effective_batch_size = config['batch_size'] * accumulation_steps
        print(f"实际batch_size: {config['batch_size']}, 等效batch_size: {effective_batch_size}")

        for batch_idx, (images, labels, _) in enumerate(
                tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config["num_epochs"]} 训练')):
            images, labels = images.to(device), labels.to(device)

            # 应用Mixup/Cutmix增强
            mixup_result = mixup_cutmix((images, labels))
            
            # 梯度累积：只在累积步骤开始时清零梯度
            if batch_idx % accumulation_steps == 0:
                optimizer.zero_grad()
            
            # 混合精度前向传播
            if scaler is not None:
                with get_autocast_context():
                    if len(mixup_result) == 4:  # 使用了mixup/cutmix
                        mixed_images, targets_a, targets_b, lam = mixup_result
                        outputs = model(mixed_images)
                        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                    else:  # 没有使用mixup/cutmix
                        mixed_images, original_labels = mixup_result
                        outputs = model(mixed_images)
                        loss = criterion(outputs, original_labels)
                
                # 混合精度反向传播 + 梯度累积
                loss = loss / accumulation_steps  # 缩放损失
                scaler.scale(loss).backward()
                
                # 只在累积步骤结束时更新参数
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
            else:
                # 标准精度训练
                if len(mixup_result) == 4:  # 使用了mixup/cutmix
                    mixed_images, targets_a, targets_b, lam = mixup_result
                    outputs = model(mixed_images)
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:  # 没有使用mixup/cutmix
                    mixed_images, original_labels = mixup_result
                    outputs = model(mixed_images)
                    loss = criterion(outputs, original_labels)
                
                # 标准精度训练 + 梯度累积
                loss = loss / accumulation_steps  # 缩放损失
                loss.backward()
                
                # 只在累积步骤结束时更新参数
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
            
            # 准确率计算
            with torch.no_grad():
                _, preds = torch.max(outputs, 1)
                if len(mixup_result) == 4:
                    train_correct += torch.sum(preds == labels).item()
                    train_labels_list.extend(labels.cpu().numpy())
                else:
                    train_correct += torch.sum(preds == original_labels).item()
                    train_labels_list.extend(original_labels.cpu().numpy())
                train_preds.extend(preds.cpu().numpy())

            train_loss += loss.item() * images.size(0)

        # 计算训练集指标
        train_loss_avg = train_loss / len(train_dataset)
        train_acc = train_correct / len(train_dataset)
        train_precision = precision_score(train_labels_list, train_preds, average='macro', zero_division=0)
        train_recall = recall_score(train_labels_list, train_preds, average='macro', zero_division=0)
        train_f1 = f1_score(train_labels_list, train_preds, average='macro', zero_division=0)

        # ---------------------- 测试阶段 ----------------------
        test_loss, test_acc, test_precision, test_recall, test_f1 = validate_model(
            model, test_loader, criterion, device
        )

        # 更新学习率 - 使用组合策略
        if use_plateau:
            scheduler2.step(test_acc)  # 基于测试准确率调整
        else:
            scheduler1.step()  # 使用余弦重启

        current_lr = optimizer.param_groups[0]['lr']

        # ---------------------- 模型保存与早停 ----------------------
        # 基于准确率保存模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_f1 = test_f1
            torch.save(model.state_dict(), 'model/best_model.pth')
            print(f"[Epoch {epoch + 1}] 测试准确率提升至 {test_acc:.4f}，保存最佳模型")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['early_stop_patience']:
                print(f"[Epoch {epoch + 1}] 早停: 连续{patience_counter}个epoch准确率无提升")
                break

        # ---------------------- 日志记录 ----------------------
        epoch_time = time.time() - epoch_start

        # 打印epoch日志
        print(f"""
Epoch {epoch + 1}/{config['num_epochs']} | 耗时: {epoch_time:.2f}s
训练集: 准确率={train_acc:.4f}, F1={train_f1:.4f}, 损失={train_loss_avg:.4f}
测试集: 准确率={test_acc:.4f}, F1={test_f1:.4f}, 损失={test_loss:.4f}
学习率: {current_lr:.6f} | 最佳测试准确率: {best_test_acc:.4f}
无改善epoch: {patience_counter}/{config['early_stop_patience']}
        """.strip())
        print("-" * 80)

    print(f"\n训练完成！")
    print(f"最佳测试准确率: {best_test_acc:.4f}")
    print(f"最佳测试F1: {best_test_f1:.4f}")
    print(f"最佳模型路径: model/best_model.pth")
    print(f"配置文件路径: model/config.json")


if __name__ == '__main__':
    # 高性能优化配置参数 - 追求98%准确率
    config = {
        # 数据集路径
        'train_split_img_dir': r"",
        'test_split_img_dir': r"",
        'train_split_label_path': r".csv",
        'test_split_label_path': r".csv",

        # 训练参数 - 显存友好的加速策略
        'batch_size': 16,  # 显存友好的batch_size
        'num_epochs': 150,  # 适中轮数
        'learning_rate': 0.00002,   # 适中学习率，避免过度震荡
        'min_lr': 5e-7,
        'pretrained': True,
        'early_stop_patience': 20,  # 适度耐心值
        'use_plateau_scheduler': True,   # 使用更稳定的plateau调度器

        # 正则化参数 - 平衡性能和泛化
        'label_smoothing': 0.05,   # 适度标签平滑
        'weight_decay': 0.005,     # 适度权重衰减

        # Mixup/Cutmix参数 - 大幅减少强度加速训练
        'mixup_alpha': 0.05,
        'cutmix_alpha': 0.5,
        'mixup_prob': 0.1,
        'mixup_prob_final': 0.0,
        'mixup_decay_start': 20,
        'mixup_decay_end': 30,

        # 工具设置 - 显存优化的加速策略
        'random_seed': 42,
        'num_workers': 1,  # 减少worker数量节省内存
        'pin_memory': True,  # 启用内存锁定
        'compile_model': False,  # 关闭模型编译节省显存
        'mixed_precision': True,  # 启用混合精度训练（重要！节省50%显存）
        'gradient_accumulation_steps': 4,
        'use_strong_aug': True,
        'strong_to_weak_epoch': 20,
        'head_lr': 0.0001,
        'backbone_lr': 0.00002,
        'unfreeze_epoch': 10
    }

    # 启动训练
    train(config)