import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
import netvlad
from torchvision.models import VGG16_Weights
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='Event Camera Training or Testing')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
parser.add_argument('--num_clusters', type=int, default=32, help='number of clusters for NetVLAD')
parser.add_argument('--pool_dim', type=int, default=512, help='dimension of NetVLAD output')
parser.add_argument('--model', type=str, default="train", help='train or test')

opt = parser.parse_args()

# 数据集类
class EventCameraDataset(Dataset):
    def __init__(self, file_paths, has_labels=True):
        self.file_paths = file_paths
        self.has_labels = has_labels  # 是否包含标签
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),  
            transforms.ToTensor(),  # 转换为 tensor（默认为RGB）
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_path = self.file_paths[idx]
        image = Image.open(image_path).convert('RGB')  # 打开为 RGB 图像
        image = self.transform(image)

        if self.has_labels:
            # 如果数据集包含标签，从文件名中提取标签
            filename = os.path.basename(image_path)  # 获取文件名，形如 'label_0_timestamp_1697816830748499.png'
            label_str = filename.split('_')[1]  # 提取 '0' 部分，假设标签在第二个位置
            
            try:
                label = int(label_str)  # 标签值 '0' 会被转换为整数 0
            except ValueError:
                label = 0  # 如果标签无法转换为整数，则设为默认值
        else:
            label = torch.tensor([])

        return image, label

# 训练函数
def train(epoch):
    epoch_loss = 0
    model.train()

    for iteration, (inputs, labels) in enumerate(train_loader):

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        encoded = model.encoder(inputs) 
        vlad_encoding = model.pool(encoded)  # shape [batch_size * 50, pool_dim]
        outputs = model.fc(vlad_encoding)  # shape [batch_size, num_classes]
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if iteration % 50 == 0:
            print(f"Epoch [{epoch + 1}/{opt.nEpochs}], Iteration [{iteration + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_loss = epoch_loss / len(train_loader)
    print(f"===> Epoch {epoch + 1} Complete: Avg. Loss: {avg_loss:.4f}")

    if (epoch + 1) % 20 == 0:
            checkpoint_path = f'/cluster/home2/zbs/icra/models/checkpoint_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")    

def get_png_files(data_dir):
    file_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.png')]
    return file_paths


# 模型初始化
def init_model(pretrained=False):
    global model, criterion, optimizer

    encoder_dim = 512
    encoder = models.vgg16(pretrained=True).features
    layers = list(encoder.children())[:-2]
    encoder = nn.Sequential(*layers)
    
    net_vlad = netvlad.NetVLAD(num_clusters=opt.num_clusters, dim=encoder_dim)

    model = nn.Module()
    model.add_module('encoder', encoder)
    model.add_module('pool', net_vlad)
    model.add_module('fc', nn.Linear(encoder_dim * opt.num_clusters, 16))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

def evaluate():
    model.eval()  # 设置为评估模式
    all_feature_vectors = []  # 用来存储所有批次的特征向量

    with torch.no_grad():
        for inputs, _ in val_loader:  # 在验证集中不需要标签（即只关心特征向量）
            inputs = inputs.to(device)
            
            # 通过模型获取NetVLAD层的输出
            encoded = model.encoder(inputs)
            vlad_encoding = model.pool(encoded)  # NetVLAD 输出特征向量

            # 将每个批次的特征向量添加到列表中
            all_feature_vectors.append(vlad_encoding.cpu().numpy())

    # 将所有特征向量合并为一个 NumPy 数组
    feature_vectors = np.concatenate(all_feature_vectors, axis=0)

    # 保存特征向量为npy文件
    np.save('features.npy', feature_vectors)
    print(f"Features saved to features.npy with shape {feature_vectors.shape}")


data_dir = r'/cluster/home2/zbs/icra/data/rgb/train/rgb'
file_paths = get_png_files(data_dir)

train_dataset = EventCameraDataset(file_paths)
train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=4)


val_data_dir = r'/cluster/home2/zbs/icra/data/rgb/test/passway'
val_file_paths = get_png_files(val_data_dir)
val_dataset = EventCameraDataset(val_file_paths,has_labels=False)
val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False, num_workers=4)


# 主程序
if __name__ == '__main__':
    if opt.model == "train":
        init_model()
        for epoch in range(opt.nEpochs):
            train(epoch)
    else:
        init_model(pretrained=False)
        pth = '/cluster/home2/zbs/icra/models/checkpoint_epoch_100.pth'
        model.load_state_dict(torch.load(pth))
        evaluate()
