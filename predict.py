"""
动物情绪识别 - 推理模块 (V5.1)
支持: DenseNet-121 / ResNet-50 / EfficientNet-B3
用法:
    python predict.py <image_path>
    python predict.py --model results/models/best_model_efficientnet_b3.pth <image_path>
"""
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import os
import sys
import argparse

# ============================
# 模型加载 (V5.1)
# ============================
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(current_dir, 'results', 'models')
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pth')
CLASS_NAMES = ['Angry', 'Sad', 'happy']


def load_model(model_path=None):
    """智能加载模型：自动检测权重格式并构建对应架构。
    返回 (model, device, input_size, class_names)
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Predict] 加载模型 on {device}...")

    if not os.path.exists(model_path):
        print(f"[Error] 模型文件 '{model_path}' 不存在，请先训练。")
        sys.exit(1)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_type' in checkpoint:
        model_type = checkpoint['model_type']
        input_size = checkpoint.get('input_size', 224)
        class_names = checkpoint.get('class_names', CLASS_NAMES)
        state_dict = checkpoint['model_state_dict']
        version = checkpoint.get('version', 'unknown')
        print(f"  [检测] {version} 权重: {model_type} (input={input_size})")
    else:
        # 旧版裸 state_dict
        model_type = 'resnet18'
        input_size = 224
        class_names = CLASS_NAMES
        state_dict = checkpoint
        print(f"  [检测] 旧版权重: {model_type} (input=224)")

    # 根据模型类型构建架构 (V5.1 含 Dropout)
    if model_type == 'densenet121':
        model = models.densenet121(weights=None)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, len(class_names)),
        )

    elif model_type == 'resnet50':
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, len(class_names)),
        )

    elif model_type == 'efficientnet_b3':
        model = models.efficientnet_b3(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, len(class_names)),
        )

    # === 向后兼容旧格式 ===
    elif model_type == 'vgg16':
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(class_names))

    elif model_type == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))

    elif model_type == 'resnet18':
        # 旧版 ResNet-18 (向后兼容)
        model = models.resnet18(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.fc.in_features, len(class_names)),
        )

    elif model_type == 'resnet18_cbam':
        # 旧版 CBAM: 使用简化 resnet18 代替
        print("  [警告] CBAM 模型不再支持，使用标准 ResNet-18 替代")
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))

    else:
        raise ValueError(f"未知模型类型: {model_type}")

    # 兼容处理：若 state_dict 的 key 不匹配，尝试加载
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"  [警告] strict load 失败，尝试非严格加载: {e}")
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()
    print(f"  [就绪] {model_type} | classes: {class_names} | input: {input_size}x{input_size}")
    return model, device, input_size, class_names


def predict_image(image_path, model, device, input_size=224, class_names=None):
    """预测单张图片情绪，返回标准字典"""
    if class_names is None:
        class_names = CLASS_NAMES

    if not os.path.exists(image_path):
        print(f"[Error] 图片 '{image_path}' 不存在。")
        return None

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top_prob, top_catid = torch.topk(probabilities, 1)
            predicted_class = class_names[top_catid.item()]

            result = {
                "class": predicted_class,
                "confidence": top_prob.item(),
                "probabilities": {class_names[i]: prob.item() for i, prob in enumerate(probabilities)},
            }
            return result

    except Exception as e:
        print(f"[Error] 预测失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_result(result, image_path):
    """打印格式化预测结果"""
    if not result:
        return
    print(f"\n{'='*10} 预测结果 {'='*10}")
    print(f"图片: {image_path}")
    print(f"情绪: {result['class']}")
    print(f"置信度: {result['confidence']*100:.2f}%")
    print("-" * 30)
    for name, prob in result['probabilities'].items():
        bar = "█" * int(prob * 30)
        print(f"  {name:6s}: {prob*100:5.1f}% {bar}")
    print("=" * 32)


# ============================
# 集成推理 (Ensemble)
# ============================
def load_ensemble():
    """加载全部三个模型，返回列表 [(model, device, input_size, class_names), ...]"""
    models = []
    model_files = [
        ('resnet50',       os.path.join(MODEL_DIR, 'best_model_resnet50.pth')),
        ('efficientnet_b3', os.path.join(MODEL_DIR, 'best_model_efficientnet_b3.pth')),
        ('densenet121',    os.path.join(MODEL_DIR, 'best_model_densenet121.pth')),
    ]

    for name, path in model_files:
        if os.path.exists(path):
            m, dev, sz, cls = load_model(path)
            models.append((m, dev, sz, cls, name))
        else:
            print(f"[集成] 警告: {name} 模型未找到 ({path})")

    if len(models) < 2:
        raise RuntimeError(f"集成需要至少2个模型，只找到 {len(models)} 个。请先训练。")
    print(f"[集成] 已加载 {len(models)} 个模型")
    return models


@torch.no_grad()
def predict_ensemble(image_path, models, weights=None):
    """多模型加权软投票集成。
    
    Args:
        image_path: 图片路径
        models: [(model, device, input_size, class_names, name), ...]
        weights: 各模型权重 (如 val_acc)，None 则等权平均
    
    Returns:
        dict: {"class", "confidence", "probabilities"}
    """
    if not os.path.exists(image_path):
        print(f"[Error] 图片 '{image_path}' 不存在。")
        return None

    all_probs = []
    class_names = None

    for i, (model, device, input_size, cls_names, name) in enumerate(models):
        if class_names is None:
            class_names = cls_names

        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        all_probs.append(probs.cpu())

    # 加权平均 softmax 概率
    if weights is None:
        weights = [1.0] * len(models)
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum()  # 归一化

    weighted_probs = sum(w * p for w, p in zip(weights, all_probs)) / weights.sum()
    top_prob, top_catid = torch.topk(weighted_probs, 1)

    print(f"[集成] 权重: {dict(zip([m[4] for m in models], weights.tolist()))}")

    result = {
        "class": class_names[top_catid.item()],
        "confidence": top_prob.item(),
        "probabilities": {class_names[i]: p.item() for i, p in enumerate(weighted_probs)},
    }
    return result


def evaluate_ensemble(models, test_dir, weights=None):
    """在测试集上评估集成精度。
    
    Args:
        models: [(model, device, input_size, class_names, name), ...]
        test_dir: 测试集目录
        weights: 各模型权重 (如 val_acc)，None 则等权平均
    
    Returns:
        (acc, labels, preds)
    """
    from torch.utils.data import DataLoader
    from torchvision import datasets

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 为每个模型准备对应的dataloader
    loaders = []
    for model, dev, input_size, class_names, name in models:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        test_data = datasets.ImageFolder(test_dir, transform=transform)
        loader = DataLoader(test_data, batch_size=16, shuffle=False, pin_memory=True)
        loaders.append((model, dev, loader, name))

    all_preds = []
    all_labels = None

    # 准备权重
    if weights is None:
        weights = [1.0] * len(models)
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum()
    model_names = [m[4] for m in models]
    print(f"[集成评估] 权重: {dict(zip(model_names, weights.tolist()))}")

    # 逐batch通过所有模型
    batch_iterators = [iter(ldr) for _, _, ldr, _ in loaders]
    first_loader = loaders[0][2]

    for batch_idx, (inputs, labels) in enumerate(first_loader):
        labels = labels.to(device)
        if all_labels is None:
            all_labels = labels
        else:
            all_labels = torch.cat([all_labels, labels])

        batch_probs = []

        for i, (model, dev, loader, name) in enumerate(loaders):
            # 获取对应模型的batch
            try:
                inp, _ = next(batch_iterators[i])
            except StopIteration:
                batch_iterators[i] = iter(loader)
                inp, _ = next(batch_iterators[i])
            inp = inp.to(dev)
            model.eval()
            with torch.no_grad():
                out = model(inp)
                probs = torch.nn.functional.softmax(out, dim=1)
                batch_probs.append(probs.cpu())

        # 加权投票：加权平均概率
        weighted_probs = sum(w * p for w, p in zip(weights, batch_probs)) / weights.sum()
        _, preds = weighted_probs.max(1)
        all_preds.append(preds)

    all_preds = torch.cat(all_preds)
    all_labels = all_labels.cpu()
    acc = 100. * all_preds.eq(all_labels).sum().item() / len(all_labels)
    return acc, all_labels.numpy(), all_preds.numpy()


def load_ensemble_with_weights(val_acc_dict=None):
    """加载三模型并附带权重。
    
    Args:
        val_acc_dict: {'resnet50': 95.78, 'efficientnet_b3': 95.33, 'densenet121': 95.56}
                      None 则等权
    Returns:
        models, weights
    """
    models = load_ensemble()
    if val_acc_dict is None:
        weights = [1.0] * len(models)
    else:
        weights = [val_acc_dict.get(m[4], 1.0) for m in models]
    return models, weights


def find_sample_image():
    """在测试集里确定性地找一张图片，供无参数演示推理使用。"""
    test_dir = os.path.join(current_dir, 'Master-Folder', 'test')
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    for root, dirs, files in os.walk(test_dir):
        dirs.sort()
        for filename in sorted(files):
            if os.path.splitext(filename)[1].lower() in extensions:
                return os.path.join(root, filename)
    return None


# ============================
# CLI 入口
# ============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="动物情绪识别预测 V5.1")
    parser.add_argument('image', nargs='?', help='图片路径')
    parser.add_argument('--model', type=str, default=None, help='模型权重路径 (默认: results/models/best_model.pth)')
    parser.add_argument('--ensemble', action='store_true', help='使用三模型集成预测')
    parser.add_argument('--equal', action='store_true', help='集成时使用等权投票 (默认: 按 val_acc 加权)')
    args = parser.parse_args()
    image_path = args.image
    if image_path is None:
        image_path = find_sample_image()
        if image_path:
            print(f"未指定图片，使用测试集示例: {image_path}")

    if args.ensemble:
        models, weights = load_ensemble_with_weights(
            None if args.equal else {
                'resnet50': 95.33, 'efficientnet_b3': 95.78, 'densenet121': 95.56
            }
        )
        if image_path:
            result = predict_ensemble(image_path, models, weights)
            print_result(result, image_path)
        else:
            print("测试集没有可用图片，请指定图片路径: python predict.py --ensemble <image_path>")
            print("  --equal  使用等权投票 (默认按 val_acc 加权)")
    else:
        model, device, input_size, class_names = load_model(args.model)
        if image_path:
            result = predict_image(image_path, model, device, input_size, class_names)
            print_result(result, image_path)
        else:
            print("测试集没有可用图片，请指定图片路径: python predict.py <image_path>")
