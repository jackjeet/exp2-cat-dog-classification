import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import os

# 解决中文显示问题
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]


# 1. 定义模型结构（与训练时一致）
class CatDogModel(torch.nn.Module):
    def __init__(self):
        super(CatDogModel, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.14.1', 'resnet18', weights=None)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Linear(in_features, 2)

    def forward(self, x):
        return self.resnet(x)


# 2. 加载训练好的模型
model = CatDogModel()
device = torch.device("cpu")
model.load_state_dict(torch.load("cat_dog_model.pth", map_location=device))
model.eval()  # 切换到评估模式

# 3. 加载测试集数据
test_dir = "dataset/test"
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
class_names = test_dataset.classes  # ['cat', 'dog']

# 4. 收集测试集的真实标签和预测结果
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# 5. 输出分类报告（每个类别的详细指标）
print("=== 分类报告 ===")
print(classification_report(
    all_labels, all_preds,
    target_names=class_names,
    digits=2
))

# 6. 绘制混淆矩阵并保存（不弹窗，直接存为图片）
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel('预测类别')
plt.ylabel('真实类别')
plt.title('混淆矩阵')
plt.savefig('confusion_matrix.png', bbox_inches='tight')  # 保存到当前目录
plt.close()  # 关闭图像，释放资源
print("混淆矩阵已保存为：confusion_matrix.png")

# 7. 可视化错误分类的图片并保存
misclassified_indices = [i for i, (p, l) in enumerate(zip(all_preds, all_labels)) if p != l]
if misclassified_indices:
    plt.figure(figsize=(15, 4))
    for i, idx in enumerate(misclassified_indices[:5]):  # 显示前5张错误样本
        img_path, true_label = test_dataset.samples[idx]
        img = Image.open(img_path).convert("RGB")
        pred_label = all_preds[idx]

        plt.subplot(1, 5, i + 1)
        plt.imshow(img)
        plt.title(f"真实: {class_names[true_label]}\n预测: {class_names[pred_label]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('misclassified_samples.png', bbox_inches='tight')  # 保存到当前目录
    plt.close()
    print("错误分类样本已保存为：misclassified_samples.png")
else:
    print("所有测试样本均分类正确，无错误样本可展示！")