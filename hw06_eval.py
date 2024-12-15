from PIL import Image
import os
import numpy as np

dataset_path = "faces"

# 筛选出0.jpg到25.jpg的文件路径
image_paths = sorted(
    [os.path.join(dataset_path, x) for x in os.listdir(dataset_path)
     if x.endswith(".jpg") and x.split(".")[0].isdigit() and 0 <= int(x.split(".")[0]) <= 999]
)

# 将图片调整为 64x64 的大小并转换为 NumPy 数组
target_size = (64, 64)  # 目标大小
real_images = [np.array(Image.open(path).convert("RGB").resize(target_size, Image.LANCZOS)) for path in image_paths]

# 转换为 NumPy 数组并打印形状
real_images = np.array(real_images)
print(real_images.shape)  # 输出形状 (25, 64, 64, 3)
import torch
from torchvision.transforms import functional as F


def preprocess_image(image):
    image = torch.tensor(image).unsqueeze(0)
    image = image.permute(0, 3, 1, 2) / 255.0
    return F.center_crop(image, (256, 256))

real_images = torch.cat([preprocess_image(image) for image in real_images])
print(real_images.shape)
# torch.Size([10, 3, 256, 256])

def fid_submission():
    dataset_path = "submission"

    # 筛选出0.jpg到25.jpg的文件路径
    image_paths = sorted(
        [os.path.join(dataset_path, x) for x in os.listdir(dataset_path)
         if x.endswith(".jpg") and x.split(".")[0].isdigit() and 0 <= int(x.split(".")[0]) <= 1000]
    )

    # 将图片调整为 64x64 的大小并转换为 NumPy 数组
    target_size = (64, 64)  # 目标大小
    fake_images = [np.array(Image.open(path).convert("RGB").resize(target_size, Image.LANCZOS)) for path in image_paths]

    # 转换为 NumPy 数组并打印形状
    fake_images = np.array(fake_images)
    print(fake_images.shape)  # 输出形状 (25, 64, 64, 3)
    import torch
    from torchvision.transforms import functional as F

    def preprocess_image(image):
        image = torch.tensor(image).unsqueeze(0)
        image = image.permute(0, 3, 1, 2) / 255.0
        return F.center_crop(image, (256, 256))

    fake_images = torch.cat([preprocess_image(image) for image in fake_images])
    print(fake_images.shape)
    # torch.Size([10, 3, 256, 256])

    from torchmetrics.image.fid import FrechetInceptionDistance

    fid = FrechetInceptionDistance(normalize=True)
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)

    print(f"submission, FID: {float(fid.compute())}")
def fid_results():
    milestone = 1

    import math
    import numpy as np
    from PIL import Image
    for milestone in range(1, 11):
        # 假设 PNG 文件路径
        file_path = str(f'results/sample-{milestone}.png')

        # 加载保存的图片
        image = Image.open(file_path).convert("RGB")
        image_array = np.array(image)

        # 获取图片大小
        image_height, image_width, _ = image_array.shape
        num_samples = 25
        grid_size = int(math.sqrt(num_samples))  # 假设网格是 sqrt(num_samples) x sqrt(num_samples)
        sample_height, sample_width = 64, 64  # 单个样本的目标大小
        padding = 2  # 保存图片时的 padding 值

        # 计算实际单个样本的大小（含 padding）
        sample_with_padding_height = sample_height + padding
        sample_with_padding_width = sample_width + padding

        # 去掉黑边（忽略 padding 的像素）
        clean_images = []
        for i in range(grid_size):
            for j in range(grid_size):
                # 计算单个样本的实际像素范围（跳过 padding）
                y_start = i * sample_with_padding_height + padding
                y_end = y_start + sample_height
                x_start = j * sample_with_padding_width + padding
                x_end = x_start + sample_width

                # 提取并存储干净的图像块
                clean_images.append(image_array[y_start:y_end, x_start:x_end, :])

        # 转换为 NumPy 数组并打印形状
        clean_images = np.array(clean_images)
        print(clean_images.shape)  # 输出 (25, 64, 64, 3)

        clean_images = torch.cat([preprocess_image(image) for image in clean_images])
        print(clean_images.shape)

        from torchmetrics.image.fid import FrechetInceptionDistance

        fid = FrechetInceptionDistance(normalize=True)
        fid.update(real_images, real=True)
        fid.update(clean_images, real=False)

        print(f"milestone:{milestone}, FID: {float(fid.compute())}")
fid_submission()
# 目前最好的分数是在啥也没有修改的情况下：
# milestone:5, FID: 71.642578125
# submission, FID: 65.6946792602539
# 加了颜色，随机裁剪、旋转等变换后，fid有了提高，但是人眼看不如之前的。
# submission, FID: 45.190025329589844
# 去掉了颜色、随机裁剪，fid提高
# submission, FID: 35.223941802978516
# timestep从100加到了300，fid并没有提高
# submission, FID: 36.82044219970703
