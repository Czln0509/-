import os
import sys
import json
import torch
import pandas as pd
from torchvision import transforms
from tqdm import tqdm
from PIL import Image, ImageFile
import numpy as np
from model import get_model
import time

ImageFile.LOAD_TRUNCATED_IMAGES = True


def setup_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    return device


def robust_image_loader(img_path):
    """极简图像加载函数"""
    try:
        image = Image.open(img_path).convert('RGB')
        return image
    except Exception as e:
        print(f"图像加载失败 {os.path.basename(img_path)}: {e}")
        # 快速创建备用图像
        return Image.new('RGB', (300, 300), color='gray')


class FastPredictor:

    # ===================================================================
    # 在 class FastPredictor: 内部添加此方法
    # ===================================================================

    def predict_batch_all_probs(self, images_batch, use_tta=False, fast_tta=False):
        """
        Streamlit 专用：预测一个批次图像，返回所有类别的 Softmax 概率 (NumPy数组)。
        """
        self.model.eval()

        # 1. TTA处理
        if use_tta:
            # 选择TTA策略
            tta_transforms = self.tta_transforms_fast if fast_tta else self.tta_transforms_full
            all_logits = []

            with torch.no_grad():
                for img in images_batch:
                    # 对每个 TTA 变换应用
                    # 将 PIL Image 转换为 Tensor 列表
                    img_tensors = [t(img) for t in tta_transforms]
                    img_tensors = torch.stack(img_tensors).to(self.device)

                    # 获取 logit 输出
                    outputs = self.model(img_tensors)

                    # 平均 logit (而不是概率，提高准确率)
                    avg_logit = outputs.mean(dim=0, keepdim=True)
                    all_logits.append(avg_logit)

            logits = torch.cat(all_logits, dim=0)

        # 2. 非 TTA 模式
        else:
            # 使用基础 transform 准备 Tensor 批次
            images_tensor = torch.stack([self.transform(img) for img in images_batch]).to(self.device)
            with torch.no_grad():
                logits = self.model(images_tensor)

        # 3. 温度缩放和 Softmax
        # 应用温度缩放
        scaled_logits = logits / self.temperature

        # Softmax 得到概率
        probabilities = self.softmax(scaled_logits)

        # 返回 CPU 上的 numpy 数组
        return probabilities.cpu().numpy()


    def __init__(self, config_path, model_path, device, temperature=0.7):
        """初始化快速预测器 - 添加温度缩放"""
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        self.label_mapping = config['label_mapping']
        self.num_classes = config['num_classes']
        self.reverse_mapping = {int(v): int(k) for k, v in self.label_mapping.items()}
        self.device = device
        self.temperature = temperature  # 温度缩放参数

        # 加载模型 - 兼容旧模型结构
        try:
            # 尝试加载新模型结构
            self.model = get_model(self.num_classes, pretrained=False)
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print("加载新模型结构成功")
        except RuntimeError as e:
            if "size mismatch" in str(e) or "Missing key" in str(e):
                print("检测到旧模型结构，使用兼容模式...")
                # 使用旧模型结构
                self.model = self._get_old_model(self.num_classes)
                self.model.load_state_dict(torch.load(model_path, map_location=device))
                print("兼容模式加载成功")
            else:
                raise e

        self.model = self.model.to(device)
        self.model.eval()

        # 基础预处理
        self.transform = transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(300),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # TTA增强预处理 - 可配置版本
        self.tta_transforms_full = [
            # 原始图像
            transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(300),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            # 水平翻转
            transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(300),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            # 多尺度1
            transforms.Compose([
                transforms.Resize(340),
                transforms.CenterCrop(300),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            # 多尺度2
            transforms.Compose([
                transforms.Resize(310),
                transforms.CenterCrop(300),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        ]

        # 快速TTA - 只用最有效的2种
        self.tta_transforms_fast = [
            # 原始图像
            transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(300),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            # 水平翻转
            transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(300),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        ]

        self.softmax = torch.nn.Softmax(dim=1)

        print(f"快速预测器初始化完成 - 温度缩放: {temperature}")

    def _get_old_model(self, num_classes):
        """旧模型结构 - 兼容已训练的模型"""
        return get_model(num_classes, pretrained=False, version='v1')

    def predict_batch(self, image_batch, use_tta=True, fast_tta=False):
        """批量预测 - 应用温度缩放和TTA"""
        with torch.no_grad():
            if use_tta:
                # 选择TTA策略
                tta_transforms = self.tta_transforms_fast if fast_tta else self.tta_transforms_full
                all_probs = []

                for transform in tta_transforms:
                    images_tensor = torch.stack([transform(img) for img in image_batch]).to(self.device)
                    outputs = self.model(images_tensor)
                    scaled_outputs = outputs / self.temperature
                    probs = self.softmax(scaled_outputs)
                    all_probs.append(probs)

                # 平均所有TTA结果
                avg_probs = torch.mean(torch.stack(all_probs), dim=0)
                confs, preds = torch.max(avg_probs, 1)

            else:
                # 标准预测
                images_tensor = torch.stack([self.transform(img) for img in image_batch]).to(self.device)
                outputs = self.model(images_tensor)
                scaled_outputs = outputs / self.temperature
                probs = self.softmax(scaled_outputs)
                confs, preds = torch.max(probs, 1)

            return preds.cpu().numpy(), confs.cpu().numpy()


def predict_fast(test_dir, output_path, batch_size=32, temperature=0.7, use_tta=True, fast_tta=True,
                 auto_optimize=True):
    """高性能预测函数 - 针对大规模预测优化"""
    # 路径设置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    config_path = os.path.join(project_root, 'model', 'config.json')
    model_path = os.path.join(project_root, 'model', 'best_model.pth')

    # 检查文件存在性
    if not os.path.exists(config_path):
        print(f"错误: 找不到配置文件 {config_path}")
        sys.exit(1)
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        sys.exit(1)

    # 设置设备
    device = setup_device()

    # 内存优化检查
    if auto_optimize:
        if batch_size > 32:
            print(f"🔧 内存优化: 16G内存建议batch_size≤32，当前{batch_size}可能导致OOM")
            if batch_size > 64:
                print(f"自动调整: batch_size {batch_size} → 32")
                batch_size = 32

    # 初始化预测器 - 传入温度参数
    predictor = FastPredictor(config_path, model_path, device, temperature=temperature)

    # 快速扫描图像文件
    print("扫描图像文件中...")
    supported_formats = ('.jpg', '.jpeg', '.png')
    test_images = []

    for f in os.listdir(test_dir):
        if f.lower().endswith(supported_formats):
            test_images.append(f)

    test_images.sort()
    total_images = len(test_images)
    print(f"找到 {total_images} 个测试图像")

    if total_images == 0:
        print("错误: 未找到任何测试图像")
        sys.exit(1)

    # 预测进度估算 - 更准确的时间计算
    if use_tta:
        time_per_image = 0.04 if fast_tta else 0.08  # 快速TTA vs 完整TTA
        tta_info = "快速TTA" if fast_tta else "完整TTA"
    else:
        time_per_image = 0.02
        tta_info = "无TTA"

    estimated_time = total_images * time_per_image
    print(f"预计处理时间({tta_info}): {estimated_time:.1f}秒 ({estimated_time / 60:.1f}分钟)")

    # 自动优化大规模预测
    if auto_optimize and estimated_time > 600:  # 10分钟
        print(f"⚠警告: 预计时间超过10分钟")
        if use_tta and not fast_tta:
            print(f"自动优化: 切换到快速TTA模式")
            fast_tta = True
            estimated_time = total_images * 0.04
            print(f"优化后预计时间: {estimated_time:.1f}秒 ({estimated_time / 60:.1f}分钟)")
        elif use_tta and fast_tta and total_images > 8000:
            print(f"自动优化: 图片数量过多({total_images}张)，关闭TTA以确保在时间限制内完成")
            use_tta = False
            estimated_time = total_images * 0.02
            print(f"优化后预计时间: {estimated_time:.1f}秒 ({estimated_time / 60:.1f}分钟)")
    elif estimated_time > 600:
        print(f"⚠警告: 预计时间超过10分钟，建议使用快速TTA或关闭TTA")
        if not fast_tta and use_tta:
            print(f"建议: 当前完整TTA模式，切换到快速TTA可节省50%时间")

    # 批量预测
    predictions = []
    processed_count = 0
    batch_count = (total_images + batch_size - 1) // batch_size

    print(f"开始批量预测，批次大小: {batch_size}, 总批次: {batch_count}")

    start_time = time.time()

    for batch_start in tqdm(range(0, total_images, batch_size), desc='批量预测'):
        batch_end = min(batch_start + batch_size, total_images)
        batch_files = test_images[batch_start:batch_end]

        # 加载批次图像
        batch_images = []
        valid_files = []

        for img_name in batch_files:
            try:
                img_path = os.path.join(test_dir, img_name)
                image = robust_image_loader(img_path)
                batch_images.append(image)
                valid_files.append(img_name)
            except Exception as e:
                print(f"跳过图像 {img_name}: {e}")
                continue

        if not batch_images:
            continue

        # 批量预测 - 启用TTA
        try:
            batch_preds, batch_confs = predictor.predict_batch(batch_images, use_tta=use_tta, fast_tta=fast_tta)

            # 处理预测结果
            for img_name, pred_idx, confidence in zip(valid_files, batch_preds, batch_confs):
                pred_class = predictor.reverse_mapping[pred_idx]
                predictions.append({
                    'img_name': img_name,
                    'predicted_class': pred_class,
                    'confidence': round(confidence, 4)
                })

            processed_count += len(valid_files)

        except Exception as e:
            print(f"批次预测失败: {e}")
            # 为失败的批次添加默认预测
            for img_name in valid_files:
                predictions.append({
                    'img_name': img_name,
                    'predicted_class': list(predictor.reverse_mapping.values())[0],
                    'confidence': 0.0
                })

        # 定期清理GPU内存
        if device.type == 'cuda' and batch_start % (batch_size * 10) == 0:
            torch.cuda.empty_cache()

    # 最终内存清理
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    end_time = time.time()
    total_time = end_time - start_time

    # 保存结果
    print("保存预测结果...")
    df = pd.DataFrame(predictions)

    # 确保输出目录存在 - 支持相对路径和绝对路径
    if not os.path.isabs(output_path):
        # 如果是相对路径，相对于项目根目录（code的上级目录）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        output_path = os.path.join(project_root, output_path)

    output_dir = os.path.dirname(output_path)
    if output_dir:  # 如果有目录部分
        os.makedirs(output_dir, exist_ok=True)
        print(f"输出目录: {output_dir}")

    df.to_csv(output_path, index=False, encoding='utf-8')

    # 统计信息
    valid_predictions = [p for p in predictions if p['confidence'] > 0]
    if valid_predictions:
        confidences = [p['confidence'] for p in valid_predictions]
        avg_confidence = np.mean(confidences)
        max_confidence = max(confidences)
        min_confidence = min(confidences)

        # 置信度分布
        high_conf = len([c for c in confidences if c >= 0.9])
        medium_conf = len([c for c in confidences if 0.7 <= c < 0.9])
        low_conf = len([c for c in confidences if c < 0.7])
    else:
        avg_confidence = 0
        max_confidence = 0
        min_confidence = 0
        high_conf = medium_conf = low_conf = 0

    print(f"\n预测完成!")
    print(f"总处理图像: {processed_count}/{total_images}")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"平均速度: {total_images / total_time:.1f} 图像/秒")
    print(f"平均置信度: {avg_confidence:.4f}")
    print(f"置信度范围: {min_confidence:.4f} - {max_confidence:.4f}")
    print(f"高置信度(≥0.9): {high_conf} ({high_conf / len(confidences) * 100:.1f}%)")
    print(f"中置信度(0.7-0.9): {medium_conf} ({medium_conf / len(confidences) * 100:.1f}%)")
    print(f"低置信度(<0.7): {low_conf} ({low_conf / len(confidences) * 100:.1f}%)")
    print(f"结果文件: {output_path}")
    print(f"使用温度参数: {temperature}")
    print(f"TTA模式: {'快速TTA(2x)' if fast_tta else '完整TTA(4x)' if use_tta else '无TTA'}")

    # 显示前几个结果
    print(f"\n前5个预测结果:")
    print(df.head().to_string(index=False))

    return df


def predict(test_dir, output_path, batch_size=32, temperature=0.7, use_tta=True, fast_tta=True):
    """主预测函数 - 兼容原有接口"""
    return predict_fast(test_dir, output_path, batch_size, temperature, use_tta, fast_tta)


if __name__ == '__main__':
    # 命令行参数处理
    if len(sys.argv) < 3:
        print("用法: python predict.py '/path/to/test_images' 'results/submission.csv' [批次大小] [温度参数]")
        print("示例: python predict.py '/path/to/test_images' 'results/submission.csv' 32 0.7")
        print("注意: 输出路径相对于项目根目录（code文件夹的上级目录）")
        print("温度参数说明: <1.0 提高置信度, >1.0 降低置信度")
        print("16G内存建议batch_size: 16-32")
        sys.exit(1)

    test_dir = sys.argv[1]
    output_path = sys.argv[2]
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 32  # 16G内存友好的默认值
    temperature = float(sys.argv[4]) if len(sys.argv) > 4 else 0.7

    # 验证输入目录
    if not os.path.exists(test_dir):
        print(f"错误: 测试目录不存在 {test_dir}")
        sys.exit(1)

    print(f"测试集目录: {test_dir}")
    print(f"输出文件: {output_path}")
    print(f"批次大小: {batch_size}")
    print(f"温度参数: {temperature}")

    # 执行快速预测 - 启用快速TTA（10分钟内完成）
    predict_fast(test_dir, output_path, batch_size, temperature, use_tta=True, fast_tta=True)