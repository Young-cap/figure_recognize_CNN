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


# ⚠️ 如果你已经在前面定义了 CNN，就不需要重复定义
# 假设你的模型结构是 CNN，像这样：
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 设置类别名（EuroSAT 共有 10 个类）
class_names = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

# 设置 device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建模型并加载权重
model = CNN(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load('eurosat_cnn.pth'))
model.eval()

# 可视化函数
def imshow(img, title=None):
    img = img.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

# 加载数据（确保 transform 与训练时一致）
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
test_dataset = datasets.ImageFolder(root='./data/EuroSAT/2750', transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=True)

# 获取一批测试图像
images, labels = next(iter(test_loader))
images = images.to(device)
labels = labels.to(device)

# 获取模型预测结果
with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

# 可视化前 6 张图像
for i in range(6):
    imshow(images[i].cpu(), title=f"P: {class_names[predicted[i]]}, T: {class_names[labels[i]]}")
