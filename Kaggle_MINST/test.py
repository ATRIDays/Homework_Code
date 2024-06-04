import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.nn import functional as F
from torch import nn, optim
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import numpy as np

# 加载MNIST数据集
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.
test_dataset = datasets.MNIST(root='./data', train=False, download=True,
                              transform=transform)

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64,
                                          shuffle=False)


# 定义一个简单的CNN模型用于特征提取
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 7 * 7,
                            256)  # Adjusted for the output from conv layers

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)  # Flatten
        x = F.relu(self.fc(x))
        return x


# 训练模型以提取特征
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FeatureExtractor().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()

num_epochs = 5
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item()}')

# 提取特征
features_train = []
labels_train = []
with torch.no_grad():
    for images, labels in train_loader:
        images = images.to(device)
        feats = model(images)
        features_train.append(feats.cpu().numpy())
        labels_train.append(labels.numpy())
features_train = np.concatenate(features_train)
labels_train = np.concatenate(labels_train)

# 使用SVM进行分类
svm_model = SVC(kernel='linear')  # 可以尝试不同的核函数
svm_model.fit(features_train, labels_train)

# 这里没有展示测试阶段的特征提取和预测过程，但步骤类似
# 你需要重复提取特征的步骤，然后用训练好的SVM模型进行预测