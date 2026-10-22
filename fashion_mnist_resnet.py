import os
import sys
import time
import copy
import json
import struct
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi font/mã hóa cp1252
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
        sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)
    except Exception:
        pass

# ==========================================
# 1. CẤU HÌNH THIẾT BỊ VÀ SIÊU THAM SỐ
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--> Đang sử dụng thiết bị: {device}")

BATCH_SIZE = 128
LEARNING_RATE = 0.002
EPOCHS = 30
PATIENCE = 10  # Dừng sớm nếu sau 10 epoch mà độ chính xác Validation không tăng
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Fashion MNIST")
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
os.makedirs(RESULT_DIR, exist_ok=True)

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# ==========================================
# 2. HÀM NẠP DỮ LIỆU FASHION MNIST
# ==========================================
def load_idx_images(filepath):
    """Đọc file ảnh định dạng idx3-ubyte"""
    with open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows * cols)
        return images

def load_idx_labels(filepath):
    """Đọc file nhãn định dạng idx1-ubyte"""
    with open(filepath, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
        return labels

class FashionDataset(Dataset):
    def __init__(self, images, labels):
        # Chuẩn hóa về [0.0, 1.0] và giữ nguyên dạng ảnh 1 kênh 28x28
        self.images = torch.tensor(images, dtype=torch.float32).view(-1, 1, 28, 28) / 255.0
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def get_datasets(data_dir):
    train_img_path = os.path.join(data_dir, "train-images-idx3-ubyte")
    train_lbl_path = os.path.join(data_dir, "train-labels-idx1-ubyte")
    test_img_path = os.path.join(data_dir, "t10k-images-idx3-ubyte")
    test_lbl_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte")

    train_csv_path = os.path.join(data_dir, "fashion-mnist_train.csv")
    test_csv_path = os.path.join(data_dir, "fashion-mnist_test.csv")

    if os.path.exists(train_img_path) and os.path.exists(train_lbl_path) and \
       os.path.exists(test_img_path) and os.path.exists(test_lbl_path):
        print("--> Tải dữ liệu từ file nhị phân IDX...")
        train_images = load_idx_images(train_img_path)
        train_labels = load_idx_labels(train_lbl_path)
        test_images = load_idx_images(test_img_path)
        test_labels = load_idx_labels(test_lbl_path)
    elif os.path.exists(train_csv_path) and os.path.exists(test_csv_path):
        print("--> Tải dữ liệu từ file CSV...")
        df_train = pd.read_csv(train_csv_path)
        train_labels = df_train['label'].values
        train_images = df_train.drop('label', axis=1).values

        df_test = pd.read_csv(test_csv_path)
        test_labels = df_test['label'].values
        test_images = df_test.drop('label', axis=1).values
    else:
        raise FileNotFoundError(f"Không tìm thấy dữ liệu Fashion MNIST trong thư mục: {data_dir}")

    full_train_dataset = FashionDataset(train_images, train_labels)
    test_dataset = FashionDataset(test_images, test_labels)

    val_size = 10000
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"--> Kích thước tập Training: {len(train_dataset):,} mẫu")
    print(f"--> Kích thước tập Validation: {len(val_dataset):,} mẫu")
    print(f"--> Kích thước tập Testing: {len(test_dataset):,} mẫu")

    return train_dataset, val_dataset, test_dataset

# ==========================================
# 3. TĂNG CƯỜNG DỮ LIỆU CHO MẠNG CNN (RESNET)
# ==========================================
class ResNetAugmentation(nn.Module):
    """
    Tăng cường dữ liệu không gian cho ảnh 2D:
    - RandomHorizontalFlip: Lật ngang đối xứng
    - RandomCrop: Dịch chuyển nhẹ 2 pixel với biên cạnh
    - RandomErasing: Che khuất ngẫu nhiên một vùng nhỏ
    """
    def __init__(self):
        super(ResNetAugmentation, self).__init__()
        self.transforms = nn.Sequential(
            T.RandomHorizontalFlip(p=0.5),
            T.RandomCrop(28, padding=2, padding_mode='edge'),
            T.RandomErasing(p=0.1, scale=(0.02, 0.08), value=0)
        )

    def forward(self, x):
        return self.transforms(x)

# ==========================================
# 4. ĐỊNH NGHĨA KIẾN TRÚC RESNET THU NHỎ (SMALL RESNET)
# ==========================================
class ResidualBlock(nn.Module):
    """
    Khối Residual cơ bản:
    x -> [Conv3x3 -> BatchNorm -> GELU -> Conv3x3 -> BatchNorm] + Shortcut(x) -> GELU
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.GELU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.GELU()

        # Đường tắt (Skip / Shortcut Connection)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act2(out + residual)
        return out

class SmallResNet(nn.Module):
    """
    Kiến trúc ResNet thu nhỏ dành riêng cho ảnh 28x28 (Fashion MNIST):
    - Tầng chuẩn bị (Prep Layer): Conv 1 -> 32
    - Stage 1 (32 -> 64, stride 2): kích thước ảnh 28x28 -> 14x14
    - Stage 2 (64 -> 128, stride 2): kích thước ảnh 14x14 -> 7x7
    - Stage 3 (128 -> 256, stride 2): kích thước ảnh 7x7 -> 4x4
    - Global Adaptive Average Pooling -> Dropout -> Linear Classifier
    """
    def __init__(self, num_classes=10, dropout_rate=0.2):
        super(SmallResNet, self).__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        self.layer1 = ResidualBlock(32, 64, stride=2)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.prep(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.global_pool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

# ==========================================
# 5. HÀM ĐÁNH GIÁ (EVALUATION FUNCTION)
# ==========================================
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total
    accuracy = (correct / total) * 100.0
    return avg_loss, accuracy

# ==========================================
# 6. CHƯƠNG TRÌNH HUẤN LUYỆN CHÍNH (MAIN)
# ==========================================
def main():
    # 1. Tải dữ liệu
    train_dataset, val_dataset, test_dataset = get_datasets(DATA_DIR)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Khởi tạo mô hình Small ResNet & Augmentation
    model = SmallResNet(num_classes=10, dropout_rate=0.2).to(device)
    data_augmentation = ResNetAugmentation().to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nKiến trúc mô hình: Small ResNet (Tổng số tham số: {total_params:,})")
    print(model)

    # 3. Hàm mất mát (Label Smoothing CrossEntropy) & Optimizer AdamW + CosineAnnealingLR
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    # 4. Quá trình huấn luyện với cơ chế Early Stopping
    print("\n" + "="*112)
    print(f"{'Epoch':^7} | {'Thời gian':^12} | {'Train Loss':^12} | {'Train Acc (%)':^15} | {'Val Loss':^12} | {'Val Acc (%)':^15} | {'Early Stopping':^22}")
    print("="*112)

    total_start_time = time.time()
    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    best_model_weights = copy.deepcopy(model.state_dict())
    history = []

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()

        # Chế độ huấn luyện
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Áp dụng Tăng cường dữ liệu trên tập Training
            images = data_augmentation(images)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass & optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        epoch_duration = time.time() - epoch_start_time

        # Tính toán kết quả epoch
        epoch_train_loss = running_train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100.0

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        # Lưu lại lịch sử
        history.append({
            "epoch": epoch,
            "duration_sec": round(epoch_duration, 2),
            "train_loss": round(epoch_train_loss, 4),
            "train_acc": round(epoch_train_acc, 2),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2)
        })

        # Cơ chế Early Stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            best_model_weights = copy.deepcopy(model.state_dict())
            status = f"-> Best ({best_val_acc:.2f}%)"
        else:
            epochs_no_improve += 1
            status = f"Không tăng ({epochs_no_improve}/{PATIENCE})"

        print(f"{epoch:^7} | {epoch_duration:^9.2f} s | {epoch_train_loss:^12.4f} | {epoch_train_acc:^15.2f} | {val_loss:^12.4f} | {val_acc:^15.2f} | {status:^22}")

        if epochs_no_improve >= PATIENCE:
            print("\n" + "-"*112)
            print(f"[Early Stopping Kích Hoạt] Dừng huấn luyện tại Epoch {epoch} vì sau {PATIENCE} epoch liên tiếp độ chính xác Validation không tăng.")
            print("-"*112)
            break

    total_training_time = time.time() - total_start_time
    print("="*112)
    print(f"Tổng thời gian huấn luyện: {total_training_time:.2f} giây.")
    print(f"Mô hình đạt độ chính xác Validation cao nhất tại Epoch {best_epoch}: {best_val_acc:.2f}%")
    print("-> Đang nạp lại trọng số tốt nhất (Best Model) để đánh giá trên tập TEST...")
    model.load_state_dict(best_model_weights)

    # 5. Đánh giá trên tập TEST
    print("\n" + "="*50)
    print("        KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST")
    print("="*50)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"-> Độ mất mát trên tập Test (Test Loss)   : {test_loss:.4f}")
    print(f"-> Độ chính xác trên tập Test (Test Acc) : {test_acc:.2f} %")
    print("="*50)

    # 6. LƯU MÔ HÌNH VÀ KẾT QUẢ VÀO THƯ MỤC RESULT
    print("\n--> Đang lưu kết quả huấn luyện và trọng số mô hình vào thư mục 'result/'...")
    
    # 6.1. Lưu trọng số mô hình tốt nhất (.pth)
    model_save_path = os.path.join(RESULT_DIR, "best_resnet_model.pth")
    torch.save({
        "epoch": best_epoch,
        "model_state_dict": best_model_weights,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_loss": test_loss
    }, model_save_path)
    print(f" [✓] Đã lưu trọng số mô hình tốt nhất: {model_save_path}")

    # 6.2. Lưu lịch sử huấn luyện ra file CSV
    csv_save_path = os.path.join(RESULT_DIR, "resnet_training_history.csv")
    df_history = pd.DataFrame(history)
    df_history.to_csv(csv_save_path, index=False, encoding='utf-8')
    print(f" [✓] Đã lưu lịch sử huấn luyện: {csv_save_path}")

    # 6.3. Lưu tóm tắt kết quả ra file JSON
    summary_save_path = os.path.join(RESULT_DIR, "resnet_summary.json")
    summary_data = {
        "model_architecture": "SmallResNet",
        "total_parameters": total_params,
        "batch_size": BATCH_SIZE,
        "initial_learning_rate": LEARNING_RATE,
        "epochs_trained": len(history),
        "best_epoch": best_epoch,
        "best_val_acc_percent": round(best_val_acc, 2),
        "test_loss": round(test_loss, 4),
        "test_acc_percent": round(test_acc, 2),
        "total_training_time_sec": round(total_training_time, 2)
    }
    with open(summary_save_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)
    print(f" [✓] Đã lưu báo cáo tổng kết JSON: {summary_save_path}")
    print("="*50)

if __name__ == "__main__":
    main()
