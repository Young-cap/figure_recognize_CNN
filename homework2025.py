import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision.utils import make_grid

# 1. 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 设置路径和超参数
data_dir = './data/EuroSAT/2750'
batch_size = 64
num_epochs = 10
learning_rate = 0.001
model_path = './saved_models/eurosat_cnn.pth'

# 3. 数据预处理与加载
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

full_dataset = ImageFolder(root=data_dir, transform=transform)
class_names = full_dataset.classes

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# 4. CNN 模型定义
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleCNN(num_classes=len(class_names)).to(device)

# 5. 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 6. 训练模型
train_acc_list, train_loss_list = [], []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    train_acc_list.append(accuracy)
    train_loss_list.append(avg_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {avg_loss:.4f}  Accuracy: {accuracy:.4f}")

# 7. 保存模型
os.makedirs(os.path.dirname(model_path), exist_ok=True)
torch.save(model.state_dict(), model_path)
print(f"✅ 模型已保存至: {model_path}")

# 8. 可视化训练过程
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_loss_list, label='Loss')
plt.title("Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(train_acc_list, label='Accuracy')
plt.title("Train Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.show()

# 9. 可视化测试图像的预测
model.eval()
with torch.no_grad():
    images, labels = next(iter(val_loader))
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

    # 显示前 8 张图像及预测
    plt.figure(figsize=(12, 6))
    img_grid = make_grid(images[:8].cpu(), nrow=4)
    plt.imshow(img_grid.permute(1, 2, 0))
    titles = [f"P: {class_names[preds[i]]}\nT: {class_names[labels[i]]}" for i in range(8)]
    for i in range(8):
        plt.text((i % 4) * 64 + 5, (i // 4) * 64 + 60, titles[i], fontsize=8, color='white', backgroundcolor='black')
    plt.axis('off')
    plt.show()
