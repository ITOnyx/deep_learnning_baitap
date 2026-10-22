import os
import sys
import time
import copy
import json
import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
# 1. CẤU HÌNH THIẾT BỊ VÀ THAM SỐ HUẤN LUYỆN
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 50
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
        # Chuẩn hóa về [0.0, 1.0] và định dạng ảnh 1 kênh 28x28
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

    # Chia tập train (60.000 mẫu) thành Training (50.000 mẫu) và Validation (10.000 mẫu)
    val_size = 10000
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"--> Kích thước tập Training   : {len(train_dataset):,} mẫu")
    print(f"--> Kích thước tập Validation : {len(val_dataset):,} mẫu")
    print(f"--> Kích thước tập Testing    : {len(test_dataset):,} mẫu")

    return train_dataset, val_dataset, test_dataset

# ==========================================
# 3. KĨ THUẬT TĂNG CƯỜNG DỮ LIỆU (DATA AUGMENTATION)
# ==========================================
class AlexNetAugmentation(nn.Module):
    """
    Module tăng cường dữ liệu chuyên biệt cho ảnh thời trang và mô hình AlexNet:
    1. RandomHorizontalFlip (p=0.5): Lật ngang đối xứng (áo, quần, giày dép, túi xách).
    2. RandomCrop (28, padding=2, padding_mode='edge'): Dịch chuyển ngẫu nhiên 1-2 pixel
       giúp mô hình học được tính bất biến tịnh tiến (translation invariance).
    3. RandomRotation (degrees=10): Xoay góc nhẹ trong khoảng [-10, 10] độ.
    4. RandomErasing (p=0.2, scale=(0.02, 0.08), value=0): Kỹ thuật Cutout, che ngẫu nhiên
       một vùng nhỏ để mạng không bị phụ thuộc vào một chi tiết duy nhất (chống Overfitting).
    * Lưu ý: Chỉ áp dụng trên tập huấn luyện (Train), KHÔNG áp dụng trên Validation và Test.
    """
    def __init__(self, flip_p=0.5, crop_pad=2, rot_deg=10, erase_p=0.2):
        super(AlexNetAugmentation, self).__init__()
        self.transforms = nn.Sequential(
            T.RandomHorizontalFlip(p=flip_p),
            T.RandomCrop(28, padding=crop_pad, padding_mode='edge'),
            T.RandomRotation(degrees=rot_deg),
            T.RandomErasing(p=erase_p, scale=(0.02, 0.08), value=0)
        )

    def forward(self, x):
        return self.transforms(x)

# ==========================================
# 4. ĐỊNH NGHĨA KIẾN TRÚC MẠNG ALEXNET NÂNG CẤP VỚI DROPOUT
# ==========================================
class AlexNet(nn.Module):
    """
    Kiến trúc AlexNet nâng cấp cho Fashion MNIST (ảnh 28x28 x 1 kênh):
    Bao gồm 5 lớp tích chập (Conv) và 3 lớp Fully Connected (FC) với Dropout đa tầng:
    
    A. Khối trích xuất đặc trưng (Feature Extractor):
       - Conv 1: Conv2d(1 -> 64, 3x3, pad 1) -> BatchNorm2d -> ReLU -> MaxPool2d(2, 2) [14x14]
       - Conv 2: Conv2d(64 -> 128, 3x3, pad 1) -> BatchNorm2d -> ReLU -> MaxPool2d(2, 2) [7x7]
         + Thêm lớp Dropout2d (Spatial Dropout p=0.15) để loại bỏ ngẫu nhiên kênh đặc trưng.
       - Conv 3: Conv2d(128 -> 256, 3x3, pad 1) -> BatchNorm2d -> ReLU [7x7]
       - Conv 4: Conv2d(256 -> 256, 3x3, pad 1) -> BatchNorm2d -> ReLU [7x7]
       - Conv 5: Conv2d(256 -> 128, 3x3, pad 1) -> BatchNorm2d -> ReLU -> MaxPool2d(2, 2) [3x3]
         + Thêm lớp Dropout2d (Spatial Dropout p=0.2) chống Overfitting ở tầng đặc trưng cao cấp.

    B. Khối phân loại (Classifier) với 2 lớp Dropout truyền thống của AlexNet:
       - Flatten -> 128 * 3 * 3 = 1152 đặc trưng
       - Linear(1152 -> 1024) -> BatchNorm1d(1024) -> ReLU
       - LỚP DROPOUT 1 (p=0.5): Ngắt kết nối 50% nơ-ron giữa FC1 và FC2.
       - Linear(1024 -> 512) -> BatchNorm1d(512) -> ReLU
       - LỚP DROPOUT 2 (p=0.5): Ngắt kết nối 50% nơ-ron giữa FC2 và FC3.
       - Linear(512 -> 10 classes)
    """
    def __init__(self, num_classes=10, conv_dropout=0.2, fc_dropout=0.5):
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(
            # Lớp Conv 1
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28 -> 14x14

            # Lớp Conv 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14 -> 7x7
            nn.Dropout2d(p=conv_dropout * 0.75),    # Dropout2d tầng Conv 2

            # Lớp Conv 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Lớp Conv 4
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Lớp Conv 5
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 7x7 -> 3x3
            nn.Dropout2d(p=conv_dropout),           # Dropout2d tầng Conv 5
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=fc_dropout),               # Lớp Dropout 1 (p=0.5)

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=fc_dropout),               # Lớp Dropout 2 (p=0.5)

            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==========================================
# 5. HÀM ĐÁNH GIÁ (EVALUATION FUNCTION)
# ==========================================
def evaluate(model, data_loader, criterion, device):
    """Đánh giá độ mất mát và độ chính xác trên tập dữ liệu"""
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
# 6. HÀM VẼ ĐỒ THỊ HUẤN LUYỆN
# ==========================================
def plot_training_history(history, save_path):
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Đồ thị Loss
    ax1.plot(epochs, train_loss, label='Train Loss', color='#1f77b4', linewidth=2)
    ax1.plot(epochs, val_loss, label='Val Loss', color='#ff7f0e', linewidth=2, linestyle='--')
    ax1.set_title('Giá trị mất mát (Loss) qua từng Epoch - AlexNet', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Đồ thị Accuracy
    ax2.plot(epochs, train_acc, label='Train Accuracy', color='#2ca02c', linewidth=2)
    ax2.plot(epochs, val_acc, label='Val Accuracy', color='#d62728', linewidth=2, linestyle='--')
    ax2.set_title('Độ chính xác (Accuracy %) qua từng Epoch - AlexNet', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f" [✓] Đã lưu biểu đồ huấn luyện: {save_path}")

# ==========================================
# 7. CHƯƠNG TRÌNH HUẤN LUYỆN CHÍNH (TRAINING)
# ==========================================
def train_alexnet(epochs=EPOCHS):
    print("=" * 95)
    print("  CHƯƠNG TRÌNH HUẤN LUYỆN ALEXNET NÂNG CẤP VỚI DROPOUT & DATA AUGMENTATION")
    print("=" * 95)
    print(f"--> Đang sử dụng thiết bị tính toán: {device}")
    if torch.cuda.is_available():
        print(f"--> GPU: {torch.cuda.get_device_name(0)}")

    # 1. Tải dữ liệu
    train_dataset, val_dataset, test_dataset = get_datasets(DATA_DIR)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Khởi tạo mô hình AlexNet & Module Tăng cường dữ liệu
    model = AlexNet(num_classes=10, conv_dropout=0.2, fc_dropout=0.5).to(device)
    data_augmentation = AlexNetAugmentation(flip_p=0.5, crop_pad=2, rot_deg=10, erase_p=0.2).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nKiến trúc mô hình AlexNet (Tổng số tham số: {total_params:,}):")
    print(model)

    print("\n[+] Kĩ thuật đã bổ sung để tối ưu hóa độ chính xác:")
    print(" 1. Lớp Dropout:")
    print("    - Spatial Dropout2d(p=0.15) sau Conv2 và Dropout2d(p=0.2) sau Conv5.")
    print("    - 2 lớp Dropout(p=0.5) kinh điển trong Classifier giữa các tầng FC.")
    print(" 2. Kĩ thuật Tăng cường dữ liệu (Data Augmentation trên tập Train):")
    print("    - RandomHorizontalFlip(p=0.5): Lật ngang ảnh ngẫu nhiên.")
    print("    - RandomCrop(28, padding=2): Dịch chuyển ảnh nhẹ có bù biên cạnh.")
    print("    - RandomRotation(degrees=10): Xoay góc nhỏ +/- 10 độ.")
    print("    - RandomErasing(p=0.2): Che khuất ngẫu nhiên một phần ảnh nhỏ.")
    print(" 3. Lịch trình học (Learning Rate Scheduler): CosineAnnealingLR.")

    # 3. Hàm mất mát, Bộ tối ưu hóa AdamW và CosineAnnealingLR
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # 4. Quá trình huấn luyện trong số epoch yêu cầu
    print("\n" + "=" * 105)
    print(f"{'Epoch':^7} | {'Thời gian (s)':^14} | {'Train Loss':^12} | {'Train Acc (%)':^15} | {'Val Loss':^12} | {'Val Acc (%)':^15} | {'Ghi chú':^15}")
    print("=" * 105)

    total_start_time = time.time()
    best_val_acc = 0.0
    best_epoch = 0
    best_model_weights = copy.deepcopy(model.state_dict())
    history = []

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        # Chế độ huấn luyện
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # ÁP DỤNG DATA AUGMENTATION TRÊN TẬP HUẤN LUYỆN
            images = data_augmentation(images)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        epoch_duration = time.time() - epoch_start_time

        # Tính toán loss và accuracy trên tập training
        epoch_train_loss = running_train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100.0

        # Đánh giá trên tập validation (không tăng cường dữ liệu)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        # Lưu lịch sử từng epoch
        history.append({
            "epoch": epoch,
            "duration_sec": round(epoch_duration, 2),
            "train_loss": round(epoch_train_loss, 4),
            "train_acc": round(epoch_train_acc, 2),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2)
        })

        # Ghi nhận trạng thái tốt nhất
        note = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_weights = copy.deepcopy(model.state_dict())
            note = f"★ Best ({best_val_acc:.2f}%)"

        # In kết quả sau mỗi epoch
        print(f"{epoch:^7} | {epoch_duration:^14.2f} | {epoch_train_loss:^12.4f} | {epoch_train_acc:^15.2f} | {val_loss:^12.4f} | {val_acc:^15.2f} | {note:^15}")

    total_training_time = time.time() - total_start_time
    print("=" * 105)
    print(f"Hoàn thành huấn luyện {epochs} epoch trong: {total_training_time:.2f} giây (~{total_training_time/60:.2f} phút).")
    print(f"--> Độ chính xác Validation cao nhất đạt được: {best_val_acc:.2f}% (tại Epoch {best_epoch}).")

    # 5. Đánh giá trên tập Test với trọng số tốt nhất
    print(f"\n--> Đang nạp lại trọng số mô hình tốt nhất (Best Weights tại Epoch {best_epoch}) để kiểm tra trên tập TEST...")
    model.load_state_dict(best_model_weights)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print("\n" + "=" * 65)
    print("              KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST")
    print("=" * 65)
    print(" Mô hình sử dụng                    : AlexNet (Dropout + Augmentation)")
    print(f" Số epoch đã huấn luyện             : {epochs}")
    print(f" Giá trị mất mát trên tập Test (Loss): {test_loss:.4f}")
    print(f" Độ chính xác trên tập Test (Acc)   : {test_acc:.2f} %")
    print(f" Độ chính xác lớn nhất đạt được (Val): {best_val_acc:.2f} % (Epoch {best_epoch})")
    print("=" * 65)

    # 6. Lưu kết quả và trọng số
    model_save_path = os.path.join(RESULT_DIR, "best_alexnet_model.pth")
    csv_save_path = os.path.join(RESULT_DIR, "alexnet_training_history.csv")
    json_save_path = os.path.join(RESULT_DIR, "alexnet_summary.json")
    chart_save_path = os.path.join(RESULT_DIR, "alexnet_loss_acc_curve.png")

    torch.save({
        "epoch": best_epoch,
        "model_state_dict": best_model_weights,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_loss": test_loss
    }, model_save_path)
    print(f"\n [✓] Đã lưu trọng số mô hình tốt nhất: {model_save_path}")

    df_history = pd.DataFrame(history)
    df_history.to_csv(csv_save_path, index=False, encoding='utf-8')
    print(f" [✓] Đã lưu lịch sử huấn luyện: {csv_save_path}")

    summary_data = {
        "model_name": "AlexNet (Dropout + Data Augmentation)",
        "model_type": "alexnet",
        "total_parameters": total_params,
        "batch_size": BATCH_SIZE,
        "initial_learning_rate": LEARNING_RATE,
        "epochs_trained": epochs,
        "best_epoch": best_epoch,
        "best_val_acc_percent": round(best_val_acc, 2),
        "test_loss": round(test_loss, 4),
        "test_acc_percent": round(test_acc, 2),
        "total_training_time_sec": round(total_training_time, 2)
    }
    with open(json_save_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)
    print(f" [✓] Đã lưu báo cáo tổng kết JSON: {json_save_path}")

    plot_training_history(history, chart_save_path)

    return summary_data


if __name__ == "__main__":
    train_alexnet(epochs=EPOCHS)
