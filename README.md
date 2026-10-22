# Fashion MNIST Deep Learning Implementations

Bài tập thực hành huấn luyện và so sánh các kiến trúc mạng nơ-ron học sâu (Deep Learning) trên tập dữ liệu **Fashion MNIST** sử dụng thư viện **PyTorch**.

## 📌 Các mô hình triển khai

- **MLP (Fully Connected Neural Network)** (`fashion_mnist_nn.py`): Mạng nhiều tầng kết nối đầy đủ (Multi-Layer Perceptron).
- **LeNet-5** (`fashion_mnist_lenet5.py` / `fashion_mnist_cnn.py`): Kiến trúc mạng CNN kinh điển của Yann LeCun.
- **AlexNet** (`fashion_mnist_alexnet.py`): Kiến trúc CNN tùy chỉnh phù hợp với ảnh 28x28 của Fashion MNIST.
- **ResNet** (`fashion_mnist_resnet.py`): Kiến trúc mạng nơ-ron còn dư (Residual Network) với Residual Blocks và Skip Connections.

## 📊 Kết quả thực nghiệm

Chi tiết lịch sử huấn luyện, biểu đồ hàm mất mát (loss) và độ chính xác (accuracy) được lưu trong thư mục `result/`:
- `result/*_loss_acc_curve.png`: Biểu đồ loss và accuracy qua từng epoch.
- `result/*_training_history.csv`: Bảng log kết quả huấn luyện chi tiết theo từng epoch.
- `result/*_summary.json`: Báo cáo tóm tắt chỉ số đánh giá (Accuracy, Precision, Recall, F1-Score).

## 🚀 Hướng dẫn chạy

1. Cài đặt các thư viện cần thiết:
```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn
```

2. Tải tập dữ liệu Fashion MNIST đặt vào thư mục `Fashion MNIST/` hoặc để script tự động đọc file csv / idx-ubyte.

3. Chạy từng mô hình:
```bash
# Chạy mô hình MLP
python fashion_mnist_nn.py

# Chạy mô hình LeNet-5
python fashion_mnist_lenet5.py

# Chạy mô hình AlexNet
python fashion_mnist_alexnet.py

# Chạy mô hình ResNet
python fashion_mnist_resnet.py
```
