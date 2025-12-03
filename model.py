import torch
import torchvision.models as models
import torch.nn as nn


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def get_model(num_classes, pretrained=True, version='v2'):
    """
    优化版EfficientNetV2模型 - 更简洁有效的分类器
    """
    try:
        # 加载预训练模型
        if pretrained:
            model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        else:
            model = models.efficientnet_v2_s(weights=None)

        # 获取原始特征数
        original_in_features = model.classifier[1].in_features

        if version == 'v1':
            # 旧版分类器 - 兼容已训练模型
            model.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(original_in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )
            print(f"模型创建成功: EfficientNetV2-S + 兼容分类器(v1), 类别数: {num_classes}")
        else:
            # 新版高性能分类器 - 平衡容量和正则化
            model.classifier = nn.Sequential(
                nn.Dropout(0.3),  # 适度dropout
                nn.Linear(original_in_features, 1024),  # 增加容量
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),  # 最后层较少dropout
                nn.Linear(256, num_classes)
            )
            print(f"模型创建成功: EfficientNetV2-S + 高性能分类器(v2), 类别数: {num_classes}")

        # 优化冻结策略 - 只冻结前3层
        freeze_layers = 3
        layer_count = 0
        for name, child in model.features.named_children():
            if layer_count < freeze_layers:
                for param in child.parameters():
                    param.requires_grad = False
                print(f"冻结层: {name}")
            layer_count += 1

        # 统计参数
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"可训练参数: {trainable_params:,} / 总参数: {total_params:,}")

        return model

    except Exception as e:
        print(f"模型创建失败: {e}")
        raise