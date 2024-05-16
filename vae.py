import os

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    #transforms.Normalize((0.5,), (0.5,))
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
        self.fc1 = nn.Linear(28 * 28, 400)
        self.fc21 = nn.Linear(400, 2)  # 均值
        self.fc22 = nn.Linear(400, 2)  # 对数方差

        # 解码器
        self.fc3 = nn.Linear(2, 400)
        self.fc4 = nn.Linear(400, 28 * 28)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        y = torch.sigmoid(self.fc4(h3))
        y = y.view(-1, 1, 28, 28)
        return y

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 28*28))
        z = self.reparameterize(mu, logvar)
        y = self.decode(z)
        return y, mu, logvar, z

# 定义损失函数
def loss_function(recon_x, x, mu, logvar):
    #print(f"recon_x: {recon_x}, x:{x}")
    # 重构损失
    #BCE = F.binary_cross_entropy(recon_x.view(-1, 28*28), x.view(-1, 28*28), reduction='sum')
    loss = F.mse_loss(recon_x,x, reduction='sum')
    # KL散度损失
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #return (BCE + KLD)/ recon_x.size(0)
    return (loss + KLD)/ recon_x.size(0)


def train():
    # 实例化网络
    model = Net()

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            #print(f"images:{images}, labels:{labels}")
            # 前向传播
            outputs, mean, logvar, code = model(images)
            loss = loss_function(outputs, images, mean, logvar)
            #kl_loss = torch.sum((torch.exp(logvar)-(1 + logvar) + mean.pow(2)))/outputs.shape[0]
            #loss += kl_loss

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    # 保存模型
    torch.save(model.state_dict(), 'mnist_vae_model.pth')

def show_images():
    # 实例化网络
    model = Net()
    model.load_state_dict(torch.load('mnist_vae_model.pth'))
    # 测试模型
    model.eval()
    with torch.no_grad():
        # 可视化一些测试结果
        dataiter = iter(train_loader)
        images, _ = next(dataiter)
        print(f"images shape: {images.shape}, size:{images.size()}")
        output,mean,logvar, code = model(images)
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
    model.load_state_dict(torch.load('mnist_vae_model.pth'))
    # 确保模型处于评估模式
    model.eval()
    # 存储编码器的二维向量和对应的标签
    codes = []
    labels = []
    with torch.no_grad():
        for data in test_loader:
            images, label = data
            _, _,_,code = model(images)
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
    model.load_state_dict(torch.load('mnist_vae_model.pth'))
    # 确保模型处于评估模式
    model.eval()
    # 生成向量网格
    x = np.linspace(-5, 5, 10)
    y = np.linspace(-5, 5, 10)
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
    train()
    show_images()
    show_code()
    eval_code()
