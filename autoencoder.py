import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


# 定义超参数
batch_size = 64
learning_rate = 0.01
num_epochs = 30

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 下载并加载数据
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 定义网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 2)  # 压缩到2个特征，以便可视化
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(2, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Tanh()  # 输出值在-1到1之间，因为输入也被标准化到这个范围
        )

    def decode(self, z):
        x = self.decoder(z)
        x = x.view(-1, 1, 28, 28)  # 重塑为图像的形状
        return x

    def forward(self, x):
        # 定义前向传播
        x = x.view(-1, 28*28)
        #x = torch.relu(self.fc1(x))
        y = self.encoder(x)
        x = self.decoder(y)
        x = x.view(-1, 1, 28, 28)
        return x,y
def train_with_code_loss():
    # 实例化网络
    model = Net()

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    ones = torch.ones((1,2)) * 0.1

    # 训练模型
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # 前向传播
            outputs, code = model(images)
            #print(f"code.shape: {code.shape}")
            loss = criterion(outputs, images)
            code_loss = torch.sum(code**2)/code.size(0)
            if i == 0:
                print(f"code_loss: {code_loss}")
            if code_loss > 200:
                loss = loss + code_loss

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i == 0 or (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    # 保存模型
    torch.save(model.state_dict(), 'mnist_model_with_code_loss.pth')
def train():
    # 实例化网络
    model = Net()

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # 前向传播
            outputs, code = model(images)
            loss = criterion(outputs, images)
            #loss += torch.sqrt(torch.sum(code**2))

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    # 保存模型
    torch.save(model.state_dict(), 'mnist_model.pth')

def show_images():
    # 实例化网络
    model = Net()
    model.load_state_dict(torch.load('mnist_model_with_code_loss.pth'))
    # 测试模型
    model.eval()
    with torch.no_grad():
        # 可视化一些测试结果
        dataiter = iter(train_loader)
        images, _ = next(dataiter)
        print(f"images shape: {images.shape}, size:{images.size()}")
        output,code = model(images)
        # 显示原始图像和重建图像
        for i in range(6):
            # 原始图像
            plt.subplot(2, 6, i + 1)
            original = images[i].squeeze()  # 去除通道维度，从(1, 28, 28)变为(28, 28)
            plt.imshow(original.numpy(), cmap='gray')
            plt.axis('off')
            # 重建图像
            plt.subplot(2, 6, i + 7)
            reconstructed = output[i].squeeze()  # 去除通道维度，从(1, 28, 28)变为(28, 28)
            plt.imshow(reconstructed.detach().numpy(), cmap='gray')
            plt.axis('off')
        plt.show()

def show_code():
    # 实例化网络
    model = Net()
    model.load_state_dict(torch.load('mnist_model_with_code_loss.pth'))
    # 确保模型处于评估模式
    model.eval()
    # 存储编码器的二维向量和对应的标签
    codes = []
    labels = []
    with torch.no_grad():
        for data in test_loader:
            images, label = data
            _, code = model(images)
            codes.append(code)
            labels.append(label)
    # 将所有批次的结果合并
    codes = torch.cat(codes, dim=0)
    labels = torch.cat(labels, dim=0)
    # 绘制二维向量
    for i in range(10):
        indices = labels == i
        plt.scatter(codes[indices, 0], codes[indices, 1], label=str(i), alpha=0.5)
    plt.legend()
    plt.xlabel('Code Dimension 1')
    plt.ylabel('Code Dimension 2')
    plt.title('2D Codes from Autoencoder')
    plt.show()

def eval_code():
    # 实例化网络
    model = Net()
    model.load_state_dict(torch.load('mnist_model.pth'))
    # 确保模型处于评估模式
    model.eval()
    # 生成向量网格
    x = np.linspace(-0.01, 0.02, 10)
    y = np.linspace(-0.02, 0.01, 10)
    xx, yy = np.meshgrid(x, y)
    grid = np.vstack([xx.ravel(), yy.ravel()]).T
    # 将numpy数组转换为torch张量
    grid_tensor = torch.from_numpy(grid).float()
    # 通过解码器生成图像
    with torch.no_grad():
        generated_images = model.decode(grid_tensor)
    # 将生成的图像显示出来
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated_images[i].squeeze(), cmap='gray')
        ax.axis('off')
    plt.show()
if __name__ == '__main__':
    #train_with_code_loss()
    show_images()
    show_code()