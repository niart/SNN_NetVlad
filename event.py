import argparse
import torch
import torch.nn as nn
import netvlad
from spikingjelly.activation_based import neuron, layer , surrogate
from copy import deepcopy
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import torch.optim as optim
import snntorch as snn
from snntorch import surrogate
from netvlad import NetVLAD  

# 配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
opt = argparse.Namespace(batchSize=64, nEpochs=400, lr=0.001, num_clusters=16)

relu = nn.ReLU(inplace=True)
class LIFNeuron(snn.Leaky):
    def __init__(self, beta=0.9, learn_beta=True, threshold=1.0, reset_mechanism="subtract"):
        super(LIFNeuron, self).__init__(beta=beta, learn_beta=learn_beta, threshold=threshold, reset_mechanism=reset_mechanism)
        
    def forward(self, input_, mem):
        # 如果膜电位是None（初始状态），则初始化为0
        if mem is None:
            mem = torch.zeros_like(input_)
            
        # 使用父类的forward函数计算脉冲和更新后的膜电位
        spike, mem_next = super(LIFNeuron, self).forward(input_, mem)
        
        # 根据重置机制来重置膜电位，但不断开计算图
        if self.reset_mechanism == "subtract":
            # 创建膜电位的副本，以避免原地操作
            mem_reset = mem_next - spike * self.threshold
        elif self.reset_mechanism == "zero":
            # 当有脉冲发生时，将膜电位置为0
            mem_reset = mem_next * (1 - spike)
        else:
            mem_reset = mem_next
            
        return spike, mem_reset

class SpikingVGG(nn.Module):
    def __init__(self, timesteps=50, num_clusters=16, num_classes=16, input_size=128):
        super(SpikingVGG, self).__init__()
        self.timesteps = timesteps
        self.input_size = input_size
        
        # 卷积层
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        
        # 计算卷积和池化后的特征图尺寸
        feature_size = input_size // 8
        self.feature_size = feature_size
        
        # 脉冲神经元激活函数
        self.lif1 = LIFNeuron(beta=0.9, learn_beta=True, threshold=1.0, reset_mechanism="subtract")
        self.lif2 = LIFNeuron(beta=0.9, learn_beta=True, threshold=1.0, reset_mechanism="subtract")
        self.lif3 = LIFNeuron(beta=0.9, learn_beta=True, threshold=1.0, reset_mechanism="subtract")
        
        # NetVLAD 层
        self.netvlad = NetVLAD(num_clusters=num_clusters, dim=256)
        
        # 分类器
        self.classifier = nn.Linear(256 * num_clusters, num_classes)
        
        print(f"模型初始化完成，输入尺寸：{input_size}，特征图尺寸：{feature_size}")
        
    def forward(self, x):
        batch_size = x.size(0)
        device = x.device
        
        # 初始化膜电位，使用None而不是为其分配张量
        mem1 = mem2 = mem3 = None
        
        # 存储最后时间步的脉冲
        final_spike = None
        
        # 为每个时间步处理输入
        for t in range(self.timesteps):
            # 获取当前时间步的输入
            x_t = x[:, t]
            
            # 第一层 - 卷积+LIF
            conv1_out = self.conv1(x_t)
            spike1, mem1 = self.lif1(conv1_out, mem1)
            
            # 第二层 - 卷积+LIF
            conv2_out = self.conv2(spike1)
            spike2, mem2 = self.lif2(conv2_out, mem2)
            
            # 第三层 - 卷积+LIF
            conv3_out = self.conv3(spike2)
            spike3, mem3 = self.lif3(conv3_out, mem3)
            
            # 保存最后一个时间步的输出
            if t == self.timesteps - 1:
                final_spike = spike3
        
        # 使用NetVLAD对最后时间步的脉冲进行特征提取
        vlad_features = self.netvlad(final_spike)
        
        # 分类
        output = self.classifier(vlad_features)
        
        return output

class EventCameraDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx], allow_pickle=True).item()
        image = data['data'] 
        label = int(data['label'])
        image = torch.tensor(image, dtype=torch.float32)  
        label = torch.tensor(label, dtype=torch.long)
        return image, label

# 训练函数
def train(epoch, max_grad_norm=1.0):
    epoch_loss = 0
    model.train()

    for iteration, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, labels)
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        epoch_loss += loss.item()

        if iteration % 50 == 0:
            print(f"Epoch [{epoch + 1}/{opt.nEpochs}], Iteration [{iteration + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_loss = epoch_loss / len(train_loader)
    print(f"===> Epoch {epoch + 1} Complete: Avg. Loss: {avg_loss:.4f}")

    # 每 100 个 epoch 保存一次模型
    if (epoch + 1) % 10 == 0:
        checkpoint_path = f"/cluster/home2/zbs/icra/models/event_epoch_{epoch + 1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")


import os

def test(test_dir):
    model.eval()  # 设置为评估模式
    all_feature_vectors = []  # 用来存储所有批次的特征向量

    # 获取最后一级目录的名称
    last_dir = os.path.basename(test_dir)
    print(f"Testing on {last_dir}...")
    with torch.no_grad():
        for inputs, _ in test_loader:  # 在验证集中不需要标签（即只关心特征向量）
            inputs = inputs.to(device)
            # 通过模型获取NetVLAD层的输出
            # 需要去掉最后的全连接层，只获取NetVLAD的输出
            batch_size = inputs.size(0)
            mem1 = mem2 = mem3 = None
            final_spike = None

            # 为每个时间步处理输入
            for t in range(model.timesteps):
                x_t = inputs[:, t]

                # 第一层 - 卷积+LIF
                conv1_out = model.conv1(x_t)
                spike1, mem1 = model.lif1(conv1_out, mem1)

                # 第二层 - 卷积+LIF
                conv2_out = model.conv2(spike1)
                spike2, mem2 = model.lif2(conv2_out, mem2)

                # 第三层 - 卷积+LIF
                conv3_out = model.conv3(spike2)
                spike3, mem3 = model.lif3(conv3_out, mem3)

                # 保存最后一个时间步的输出
                if t == model.timesteps - 1:
                    final_spike = spike3

            # 使用NetVLAD对最后时间步的脉冲进行特征提取
            vlad_features = model.netvlad(final_spike)

            # 将每个批次的特征向量添加到列表中
            all_feature_vectors.append(vlad_features.cpu().numpy())

    # 将所有特征向量合并为一个 NumPy 数组
    feature_vectors = np.concatenate(all_feature_vectors, axis=0)

    # 保存特征向量为npy文件
    np.save(f'features_spikeVGG_{last_dir}.npy', feature_vectors)
    print(f"Features saved to features_spikeVGG_{last_dir}.npy with shape {feature_vectors.shape}")



# 获取数据
def get_npy_files(data_dir):
    file_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.npy')]
    return file_paths

data_dir = r'/cluster/home2/zbs/icra/data/event/train/chop_still_50_firstcamera_0'
file_paths = get_npy_files(data_dir)
train_dataset = EventCameraDataset(file_paths)
train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=4)

test_dir = r'/cluster/home2/zbs/icra/data/event/test/printer'
test_dataset = EventCameraDataset(file_paths)
test_loader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=4)
model_dir = r'/cluster/home2/zbs/icra/models/event_best_model_acc.pth'

# 使用 Spiking VGG 和 NetVLAD
spiking_neuron = neuron.LIFNode
model = SpikingVGG(timesteps =50).to(device)  # 使用 **kwargs


criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=opt.lr)

if __name__ == '__main__':
    model.load_state_dict(torch.load(model_dir)['model_state_dict'])

    test(test_dir)
    print("Testing complete.")
