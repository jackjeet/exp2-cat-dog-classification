import torch
from torchvision import transforms
from PIL import Image


# 1. 定义模型结构（和训练时一致）
class CatDogModel(torch.nn.Module):
    def __init__(self):
        super(CatDogModel, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.14.1', 'resnet18', weights=None)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Linear(in_features, 2)

    def forward(self, x):
        return self.resnet(x)


# 2. 加载模型权重
model = CatDogModel()
device = torch.device("cpu")
model.load_state_dict(torch.load("cat_dog_model.pth", map_location=device))
model.eval()  # 切换到预测模式

# 3. 输入测试图片路径（已改为test.jpg）
image_path = "test.png"

# 4. 图片预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)

# 5. 预测并输出结果
with torch.no_grad():
    output = model(image_tensor)
    _, predicted = torch.max(output, 1)

class_names = ['猫', '狗']
print(f"图片 '{image_path}' 的预测结果：{class_names[predicted.item()]}")