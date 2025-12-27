import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ====================== 1. 数据加载 ======================
train_dir = "dataset/train"
val_dir = "dataset/val"
test_dir = "dataset/test"

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

class_names = train_dataset.classes
print(f"类别：{class_names}（0={class_names[0]}, 1={class_names[1]}）")
print(f"数据量：训练集{len(train_dataset)}张，验证集{len(val_dataset)}张，测试集{len(test_dataset)}张")


# ====================== 2. 构建模型 ======================
class CatDogModel(nn.Module):
    def __init__(self):
        super(CatDogModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        for param in list(self.resnet.parameters())[:-10]:
            param.requires_grad = False
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.resnet(x)


model = CatDogModel()
device = torch.device("cpu")
model = model.to(device)
print("模型创建完成！")

# ====================== 3. 损失函数和优化器 ======================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ====================== 4. 训练模型 ======================
epochs = 5
train_losses = []
val_losses = []
val_accuracies = []

print("\n开始训练...")
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    train_loss /= len(train_dataset)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss /= len(val_dataset)
    val_acc = (correct / total) * 100
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"第{epoch + 1}/{epochs}轮")
    print(f"训练损失：{train_loss:.4f} | 验证损失：{val_loss:.4f} | 验证准确率：{val_acc:.2f}%")
    print("-" * 50)

# ====================== 5. 注释掉绘图代码（关键！） ======================
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(range(1, epochs + 1), train_losses, label="训练损失")
# plt.plot(range(1, epochs + 1), val_losses, label="验证损失")
# plt.title("损失变化")
# plt.xlabel("轮次")
# plt.ylabel("损失")
# plt.legend()
# plt.subplot(1, 2, 2)
# plt.plot(range(1, epochs + 1), val_accuracies, label="验证准确率", color="green")
# plt.title("准确率变化")
# plt.xlabel("轮次")
# plt.ylabel("准确率(%)")
# plt.legend()
# plt.tight_layout()
# plt.show()


# ====================== 6. 测试集评估 ======================
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
test_acc = (test_correct / test_total) * 100
print(f"\n测试集最终准确率：{test_acc:.2f}%")

# ====================== 7. 保存模型（确保执行到这里） ======================
torch.save(model.state_dict(), "cat_dog_model.pth")
print("模型已保存为：cat_dog_model.pth")  # 这行必须显示！