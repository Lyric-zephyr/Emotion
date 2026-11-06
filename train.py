# train.py  (V5.1 — 2026-07-20)
# ============================================================================
# V5.1 深度优化:
#   - SAM (Sharpness-Aware Minimization) 优化器，寻找平坦最小值 +1~3%
#   - OneCycleLR 学习率调度，更好地探索损失景观
#   - 384px 输入分辨率，捕捉更多微表情细节
#   - 三模型均优化至最佳
# ============================================================================

import os
import sys
import time
import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# ============================
# 0. 日志工具 (Logger)
# ============================
class Logger(object):
    """日志分流器: print() → 终端+日志; tqdm/stderr → 仅终端"""

    def __init__(self, filename=None):
        if filename is None:
            filename = cfg.LOG_PATH
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        self.terminal = sys.__stdout__          # 原始终端 stdout (不经过 Logger)
        self.log = open(filename, "a", encoding="utf-8")
        self.log.write(f"\n{'='*20} V5.1 Run: {time.strftime('%Y-%m-%d %H:%M:%S')} {'='*20}\n")

    def write(self, message):
        self.terminal.write(message)            # 终端可见
        self.log.write(message)                 # 写入日志
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 启动日志 (每个进程只初始化一次)
_logger_initialized = False
def init_logger():
    global _logger_initialized
    if not _logger_initialized:
        sys.stdout = Logger()
        _logger_initialized = True

# 终端专用输出 (不写入 log)
def term_print(*args, **kwargs):
    """仅在终端打印，不写入日志文件"""
    msg = " ".join(str(a) for a in args)
    end = kwargs.get('end', '\n')
    sys.__stdout__.write(msg + end)
    sys.__stdout__.flush()

# ============================
# 1. 配置参数 (Configuration)
# ============================
class Config:
    def __init__(self):
        # --- 训练超参 ---
        self.EPOCHS = 30
        self.BATCH_SIZE = 16
        self.LEARNING_RATE = 0.0003
        self.WEIGHT_DECAY = 0.01
        self.WARMUP_EPOCHS = 3
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- 路径 ---
        self.CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATA_PATH = os.path.join(self.CURRENT_DIR, 'Master-Folder')
        # 训练产物统一放在 results 下，按类型分类，避免污染项目根目录。
        self.RESULTS_DIR = os.path.join(self.CURRENT_DIR, 'results')
        self.MODEL_DIR = os.path.join(self.RESULTS_DIR, 'models')
        self.LOG_DIR = os.path.join(self.RESULTS_DIR, 'logs')
        self.PLOT_DIR = os.path.join(self.RESULTS_DIR, 'plots')
        self.LOG_PATH = os.path.join(self.LOG_DIR, 'training.log')
        for output_dir in (self.MODEL_DIR, self.LOG_DIR, self.PLOT_DIR):
            os.makedirs(output_dir, exist_ok=True)

        # --- V5.1 深度优化参数 ---
        self.INPUT_SIZE = 384          # 更高分辨率捕捉微表情
        self.LABEL_SMOOTHING = 0.08    # 标签平滑
        self.MIXUP_ALPHA = 0.15        # MixUp 增强
        self.MIXUP_PROB = 0.4          # MixUp 概率
        self.DROPOUT_RATE = 0.3        # 分类器 Dropout
        self.EMA_DECAY = 0.999         # EMA 衰减率
        self.USE_SAM = False           # SAM 在384px+OneCycleLR下不稳定，关闭
        self.SAM_RHO = 0.05            # SAM 扰动半径

        # --- 性能 ---
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision('high')

cfg = Config()

# ============================
# 2. EMA (指数移动平均)
# ============================
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = self.decay * self.shadow[name].data + (1.0 - self.decay) * param.data

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].data

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

# ============================
# 3. MixUp
# ============================
def mixup_data(x, y, alpha=0.1):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ============================
# 4. 数据准备
# ============================
def get_data_loaders(batch_size=16):
    print(f"[数据准备] 加载数据: {cfg.DATA_PATH}")

    # 增强训练 transform: RandAugment + RandomErasing
    train_transform = transforms.Compose([
        transforms.Resize((cfg.INPUT_SIZE, cfg.INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
    ])

    # 验证/测试 transform
    test_transform = transforms.Compose([
        transforms.Resize((cfg.INPUT_SIZE, cfg.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_data = datasets.ImageFolder(os.path.join(cfg.DATA_PATH, 'train'), transform=train_transform)
    val_data   = datasets.ImageFolder(os.path.join(cfg.DATA_PATH, 'val'),   transform=test_transform)
    test_data  = datasets.ImageFolder(os.path.join(cfg.DATA_PATH, 'test'),  transform=test_transform)

    try:
        shm_total = 0
        if os.path.exists('/dev/shm'):
            shm_stat = os.statvfs('/dev/shm')
            shm_total = shm_stat.f_frsize * shm_stat.f_blocks
        if shm_total < 1073741824:
            print(f"[警告] 共享内存不足 {shm_total/1024/1024:.0f}MB，已强制 num_workers=0")
            num_workers = 0
        else:
            num_workers = min(4, os.cpu_count() if os.cpu_count() else 2)
    except Exception:
        num_workers = 0

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=num_workers,
    )

    class_names = list(train_data.class_to_idx.keys())
    print(f"[数据准备] 类别: {class_names} | train: {len(train_data)} | val: {len(val_data)} | test: {len(test_data)} | 分辨率: {cfg.INPUT_SIZE}")
    return train_loader, val_loader, test_loader, len(train_data.classes), class_names

# ============================
# 5. 模型工厂 (V5.1: 仅3个模型 + Dropout)
# ============================
def build_model(model_name, num_classes):
    print(f"[模型] 初始化: {model_name} (Dropout={cfg.DROPOUT_RATE})")

    if model_name == 'densenet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(cfg.DROPOUT_RATE),
            nn.Linear(in_features, num_classes),
        )

    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(cfg.DROPOUT_RATE),
            nn.Linear(in_features, num_classes),
        )

    elif model_name == 'efficientnet_b3':
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(cfg.DROPOUT_RATE),
            nn.Linear(in_features, num_classes),
        )

    else:
        raise ValueError(f"未知模型: {model_name}。V5.1 仅支持: densenet121, resnet50, efficientnet_b3")

    return model.to(cfg.DEVICE)

# ============================
# 6. 训练器 V5.1
# ============================
class TrainerV5:
    def __init__(self, model, train_loader, val_loader, test_loader, class_names, model_name):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.class_names = class_names
        self.model_name = model_name

        self.criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
        self.optimizer = optim.AdamW(
            model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY,
        )
        self.scaler = torch.amp.GradScaler('cuda')
        # CosineAnnealingLR: 平稳余弦退火
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.EPOCHS - cfg.WARMUP_EPOCHS, eta_min=1e-6,
        )
        self.ema = ModelEMA(model, decay=cfg.EMA_DECAY) if cfg.EMA_DECAY > 0 else None
        self.use_sam = cfg.USE_SAM

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        any_non_mixup = False

        total_batches = len(self.train_loader)
        # tqdm 写入原始终端 (sys.__stdout__)，不进入日志
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1:02d} [Train]", leave=False,
                    ncols=100, file=sys.__stdout__, mininterval=2.0)
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(cfg.DEVICE), targets.to(cfg.DEVICE)

            if cfg.MIXUP_ALPHA > 0 and np.random.random() < cfg.MIXUP_PROB:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, cfg.MIXUP_ALPHA)
                mixup_applied = True
            else:
                mixup_applied = False

            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                outputs = self.model(inputs)
                if mixup_applied:
                    loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
                else:
                    loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()

            # SAM: 扰动参数 → 第二前向+反向 → 恢复 → 用SAM梯度更新
            if self.use_sam:
                # 保存原始参数
                orig_params = {n: p.data.clone() for n, p in self.model.named_parameters() if p.requires_grad}

                # 按梯度方向扰动参数（方向不受scaler影响，直接用scaled grad）
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5 + 1e-12
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.data.add_(p.grad / total_norm, alpha=cfg.SAM_RHO)

                # 第二前向+反向（扰动后的参数）
                self.optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    outputs2 = self.model(inputs)
                    if mixup_applied:
                        loss2 = mixup_criterion(self.criterion, outputs2, targets_a, targets_b, lam)
                    else:
                        loss2 = self.criterion(outputs2, targets)
                self.scaler.scale(loss2).backward()

                # 恢复原始参数
                for n, p in self.model.named_parameters():
                    if p.requires_grad and n in orig_params:
                        p.data.copy_(orig_params[n])
                del orig_params

                loss = loss2  # 报告SAM的loss

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.ema:
                self.ema.update()

            running_loss += loss.item() * inputs.size(0)
            total += targets.size(0)
            if not mixup_applied:
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                any_non_mixup = True

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total if any_non_mixup else 0.0
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in loader:
            inputs, targets = inputs.to(cfg.DEVICE), targets.to(cfg.DEVICE)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return running_loss / total, 100. * correct / total

    @torch.no_grad()
    def evaluate_ema(self, loader):
        if self.ema is None:
            return self.evaluate(loader)
        self.ema.apply_shadow()
        loss, acc = self.evaluate(loader)
        self.ema.restore()
        return loss, acc

    @torch.no_grad()
    def evaluate_detailed(self):
        if self.ema:
            self.ema.apply_shadow()
        self.model.eval()
        all_preds, all_labels = [], []
        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(cfg.DEVICE), targets.to(cfg.DEVICE)
            outputs = self.model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
        if self.ema:
            self.ema.restore()
        return np.array(all_labels), np.array(all_preds)

    def run(self):
        history = {
            'train_loss': [], 'train_acc': [],
            'val_acc': [], 'test_acc': [],
        }
        best_val_acc = 0.0
        best_test_acc = 0.0
        best_epoch = 0
        start_time = time.time()

        print(f"\n{'='*65}")
        print(f"  V5.1 训练: {self.model_name} | 设备: {cfg.DEVICE}")
        print(f"  输入: {cfg.INPUT_SIZE}×{cfg.INPUT_SIZE} | LabelSmooth={cfg.LABEL_SMOOTHING}")
        print(f"  优化器: AdamW | CosineAnnealingLR | Dropout={cfg.DROPOUT_RATE}")
        print(f"  EMA: {cfg.EMA_DECAY} | MixUp α={cfg.MIXUP_ALPHA} p={cfg.MIXUP_PROB}")
        print(f"{'='*65}")

        for epoch in range(cfg.EPOCHS):
            # Warmup
            if epoch < cfg.WARMUP_EPOCHS:
                lr_scale = (epoch + 1) / cfg.WARMUP_EPOCHS
                for pg in self.optimizer.param_groups:
                    pg['lr'] = cfg.LEARNING_RATE * lr_scale

            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.evaluate(self.val_loader)
            _, test_acc = self.evaluate_ema(self.test_loader)

            # CosineAnnealingLR: 每epoch步进
            if epoch >= cfg.WARMUP_EPOCHS:
                self.scheduler.step()

            lr_now = self.optimizer.param_groups[0]['lr']

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['test_acc'].append(test_acc)

            saved = ""
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                best_epoch = epoch + 1
                ckpt = {
                    'model_state_dict': self.model.state_dict(),
                    'model_type': self.model_name,
                    'input_size': cfg.INPUT_SIZE,
                    'class_names': self.class_names,
                    'val_accuracy': best_val_acc,
                    'test_accuracy': best_test_acc,
                    'epoch': best_epoch,
                    'version': 'V5.1',
                }
                save_path = os.path.join(cfg.MODEL_DIR, f'best_model_{self.model_name}.pth')
                torch.save(ckpt, save_path)
                if self.model_name == 'efficientnet_b3':
                    torch.save(ckpt, os.path.join(cfg.MODEL_DIR, 'best_model.pth'))
                saved = " ★SAVED"

            print(f"Epoch [{epoch+1:02d}/{cfg.EPOCHS}] "
                  f"LR: {lr_now:.6f} | Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
                  f"Test Acc: {test_acc:.2f}%{saved}")

        duration = time.time() - start_time

        print(f"\n{'='*65}")
        print(f"  完成: {self.model_name}")
        print(f"  最佳 Val Acc: {best_val_acc:.2f}% @ Epoch {best_epoch}")
        print(f"  对应 Test Acc: {best_test_acc:.2f}%")
        print(f"  最终 Test Acc (EMA): {history['test_acc'][-1]:.2f}%")
        print(f"  耗时: {duration:.1f}s ({duration/60:.1f}min)")
        print(f"{'='*65}\n")

        self._plot_confusion_matrix()

        labels, preds = self.evaluate_detailed()
        print(f"\n[分类报告] {self.model_name}")
        print(classification_report(labels, preds, target_names=self.class_names, digits=4))

        try:
            cm = confusion_matrix(labels, preds)
            a_idx = self.class_names.index('Angry')
            h_idx = self.class_names.index('happy')
            a2h = cm[a_idx][h_idx]
            h2a = cm[h_idx][a_idx]
            print(f"[分析] Angry→happy: {a2h} | happy→Angry: {h2a} | 混淆合计: {a2h+h2a}")
        except ValueError:
            pass

        return history, best_val_acc, best_test_acc

    def _plot_confusion_matrix(self):
        labels, preds = self.evaluate_detailed()
        cm = confusion_matrix(labels, preds)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'Confusion Matrix — {self.model_name} (V5.1)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        cm_path = os.path.join(cfg.PLOT_DIR, f'confusion_{self.model_name}.png')
        plt.savefig(cm_path, dpi=150)
        plt.close()
        print(f"[混淆矩阵] 已保存: {cm_path}")

# ============================
# 7. 可视化
# ============================
def plot_test_acc(all_histories, all_best_val, all_best_test, filename):
    """多模型对比: Test Accuracy vs Epochs"""
    plt.figure(figsize=(12, 7))

    colors = {'densenet121': '#2196F3', 'resnet50': '#FF9800', 'efficientnet_b3': '#4CAF50'}
    markers = {'densenet121': 'o', 'resnet50': 's', 'efficientnet_b3': 'D'}

    for m_name, history in all_histories.items():
        test_accs = history.get('test_acc', [])
        best_v = all_best_val.get(m_name, 0)
        best_t = all_best_test.get(m_name, 0)
        if test_accs:
            color = colors.get(m_name, 'gray')
            marker = markers.get(m_name, 'o')
            plt.plot(test_accs,
                     label=f'{m_name} (best val:{best_v:.2f}%, test:{best_t:.2f}%)',
                     color=color, marker=marker, markersize=2, linewidth=1.5,
                     markevery=max(1, len(test_accs)//10))

    plt.title('Model Comparison: Test Accuracy vs Epochs (V5.1)', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy (%)')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"[对比图] Test Acc 对比已保存: {filename}")

def plot_single_test_acc(history, model_name, filename):
    """单模型: Test Accuracy vs Epochs"""
    plt.figure(figsize=(10, 6))

    epochs = range(1, len(history['test_acc']) + 1)

    plt.plot(epochs, history['test_acc'], 'b-o', markersize=3, linewidth=1.5, label='Test Accuracy')
    plt.plot(epochs, history['val_acc'], 'r--s', markersize=3, linewidth=1.0, alpha=0.6, label='Val Accuracy')
    plt.plot(epochs, history['train_acc'], 'g--', linewidth=0.8, alpha=0.4, label='Train Accuracy')

    best_test = max(history['test_acc'])
    best_epoch = history['test_acc'].index(best_test) + 1
    plt.axhline(y=best_test, color='blue', linestyle=':', alpha=0.5)
    plt.annotate(f'Best Test: {best_test:.2f}% @ Epoch {best_epoch}',
                 xy=(best_epoch, best_test),
                 xytext=(best_epoch + 2, best_test - 1),
                 arrowprops=dict(arrowstyle='->', color='blue'),
                 fontsize=10, color='blue')

    plt.title(f'{model_name} — Training Curves (V5.1)', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"[曲线图] Test Acc 曲线已保存: {filename}")

# ============================
# 8. TTA (Test-Time Augmentation) 最终评估
# ============================
@torch.no_grad()
def evaluate_with_tta(model, loader, num_aug=5):
    """使用 TTA 进行最终评估: 多次随机增强取平均"""
    model.eval()
    all_probs = []
    all_labels = []

    tta_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
    ])

    for inputs, targets in tqdm(loader, desc="TTA 评估", ncols=80, leave=False, file=sys.__stdout__):
        inputs, targets = inputs.to(cfg.DEVICE), targets.to(cfg.DEVICE)
        batch_probs = torch.zeros(inputs.size(0), 3).to(cfg.DEVICE)

        for _ in range(num_aug):
            aug_list = []
            for img in inputs.cpu():
                pil_img = transforms.ToPILImage()(img)
                aug_list.append(tta_transform(pil_img))
            aug_inputs = torch.stack(aug_list).to(cfg.DEVICE)
            outputs = model(aug_inputs)
            batch_probs += torch.softmax(outputs, dim=1)

        batch_probs /= num_aug
        all_probs.append(batch_probs.cpu())
        all_labels.append(targets.cpu())

    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    _, preds = all_probs.max(1)
    acc = 100. * preds.eq(all_labels).sum().item() / len(all_labels)
    return acc, all_labels.numpy(), preds.numpy()

# ============================
# 9. 主程序入口
# ============================
if __name__ == "__main__":
    init_logger()

    parser = argparse.ArgumentParser(description="动物情绪识别训练 V5.1")
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['densenet121', 'resnet50', 'efficientnet_b3', 'all'],
                        help='模型选择 (默认: resnet50)')
    parser.add_argument('--epochs', type=int, default=cfg.EPOCHS, help='训练轮次 (默认30)')
    parser.add_argument('--batch-size', type=int, default=cfg.BATCH_SIZE, help='批大小')
    parser.add_argument('--lr', type=float, default=cfg.LEARNING_RATE, help='初始学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--tta', action='store_true', help='训练后运行 TTA 评估')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg.EPOCHS = args.epochs
    cfg.BATCH_SIZE = args.batch_size
    cfg.LEARNING_RATE = args.lr

    print(f"[V5.1] seed={args.seed}  EPOCHS={cfg.EPOCHS}  BATCH={cfg.BATCH_SIZE}  LR={cfg.LEARNING_RATE}")
    print(f"[V5.1] INPUT_SIZE={cfg.INPUT_SIZE}  DROPOUT={cfg.DROPOUT_RATE}  LABEL_SMOOTHING={cfg.LABEL_SMOOTHING}")
    print(f"[V5.1] MIXUP_ALPHA={cfg.MIXUP_ALPHA}  MIXUP_PROB={cfg.MIXUP_PROB}  SAM={cfg.USE_SAM}")

    train_loader, val_loader, test_loader, num_classes, class_names = get_data_loaders(args.batch_size)

    if args.model == 'all':
        # ============ 多模型对比模式 ============
        models_to_train = ['efficientnet_b3', 'resnet50', 'densenet121']
        all_histories = {}
        all_best_val = {}
        all_best_test = {}

        print("\n>>> V5.1 多模型对比模式 (3 Models) <<<")

        for m_name in models_to_train:
            model = build_model(m_name, num_classes)
            if torch.cuda.is_available():
                model = model.to(memory_format=torch.channels_last)

            params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"\n[模型] {m_name}: {params:.2f}M 参数")

            trainer = TrainerV5(model, train_loader, val_loader, test_loader, class_names, m_name)
            history, best_val, best_test = trainer.run()
            all_histories[m_name] = history
            all_best_val[m_name] = best_val
            all_best_test[m_name] = best_test

            if args.tta:
                print(f"\n[TTA] {m_name} Test-Time Augmentation 评估...")
                if trainer.ema:
                    trainer.ema.apply_shadow()
                tta_acc, _, _ = evaluate_with_tta(model, test_loader, num_aug=7)
                if trainer.ema:
                    trainer.ema.restore()
                print(f"[TTA] {m_name} TTA Test Acc: {tta_acc:.2f}%")

        comparison_plot = os.path.join(cfg.PLOT_DIR, 'model_comparison_test_acc.png')
        plot_test_acc(all_histories, all_best_val, all_best_test, comparison_plot)

        print("\n" + "="*65)
        print("  V5.1 三模型对比汇总")
        print("="*65)
        print(f"{'Model':<22s} {'Best Val':>10s} {'Best Test':>10s} {'Final Test':>10s}")
        print("-"*52)
        for m_name in models_to_train:
            best_v = all_best_val.get(m_name, 0)
            best_t = all_best_test.get(m_name, 0)
            final_t = all_histories[m_name]['test_acc'][-1]
            print(f"  {m_name:<20s} {best_v:>9.2f}% {best_t:>9.2f}% {final_t:>9.2f}%")
        print("="*65)

        # ====== 集成评估 (Ensemble) ======
        print("\n>>> 三模型集成评估 <<<")
        try:
            from predict import load_ensemble_with_weights, evaluate_ensemble
            models, weights = load_ensemble_with_weights(all_best_val)
            ens_acc, ens_labels, ens_preds = evaluate_ensemble(
                models, os.path.join(cfg.DATA_PATH, 'test'), weights
            )
            print(f"\n{'='*65}")
            print(f"  🏆 加权集成 Test Acc: {ens_acc:.2f}%")
            print(f"  单模型最佳: {max(all_best_test.values()):.2f}%")
            print(f"  集成提升: {ens_acc - max(all_best_test.values()):+.2f}%")

            # 同时跑等权作为对照
            models2, w2 = load_ensemble_with_weights(None)
            eq_acc, _, _ = evaluate_ensemble(models2, os.path.join(cfg.DATA_PATH, 'test'), w2)
            print(f"  等权集成: {eq_acc:.2f}% (加权 vs 等权: {ens_acc - eq_acc:+.2f}%)")
            print(f"{'='*65}")
        except Exception as e:
            print(f"[集成] 评估失败: {e}")

    else:
        # ============ 单模型模式 ============
        print(f"\n>>> V5.1 单模型训练: {args.model} <<<")
        model = build_model(args.model, num_classes)
        if torch.cuda.is_available():
            model = model.to(memory_format=torch.channels_last)

        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[模型] {args.model}: {params:.2f}M 参数")

        trainer = TrainerV5(model, train_loader, val_loader, test_loader, class_names, args.model)
        history, best_val, best_test = trainer.run()

        curve_plot = os.path.join(cfg.PLOT_DIR, f'test_acc_{args.model}.png')
        plot_single_test_acc(history, args.model, curve_plot)

        if args.tta:
            print(f"\n[TTA] {args.model} Test-Time Augmentation 评估...")
            if trainer.ema:
                trainer.ema.apply_shadow()
            tta_acc, _, _ = evaluate_with_tta(model, test_loader, num_aug=7)
            if trainer.ema:
                trainer.ema.restore()
            print(f"[TTA] {args.model} TTA Test Acc: {tta_acc:.2f}%")

    print("\n[V5.1] 训练完成！")
