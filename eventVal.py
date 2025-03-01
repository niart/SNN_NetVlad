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
import logging
import time
from datetime import datetime

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

log_dir = "/cluster/home2/zbs/icra/logs"
os.makedirs(log_dir, exist_ok=True)

current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = os.path.join(log_dir, f"training_log_{current_time}.txt")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
# 配置
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
opt = argparse.Namespace(batchSize=4, nEpochs=400, lr=0.001, num_clusters=8)


relu = nn.ReLU(inplace=True)
class LIFNeuron(snn.Leaky):
    def __init__(self, beta=0.9, learn_beta=True, threshold=1.0, reset_mechanism="subtract"):
        super(LIFNeuron, self).__init__(beta=beta, learn_beta=learn_beta, threshold=threshold, reset_mechanism=reset_mechanism)
        
    def forward(self, input_, mem):
        if mem is None:
            mem = torch.zeros_like(input_)
        spike, mem_next = super(LIFNeuron, self).forward(input_, mem)
        if self.reset_mechanism == "subtract":
            mem_reset = mem_next - spike * self.threshold
        elif self.reset_mechanism == "zero":
            mem_reset = mem_next * (1 - spike)
        else:
            mem_reset = mem_next
            
        return spike, mem_reset

class SpikingVGG(nn.Module):
    def __init__(self, timesteps=50, num_clusters=16, num_classes=16, input_size=128):
        super(SpikingVGG, self).__init__()
        self.timesteps = timesteps
        self.input_size = input_size
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        feature_size = input_size // 8
        self.feature_size = feature_size
        self.lif1 = LIFNeuron(beta=0.9, learn_beta=True, threshold=1.0, reset_mechanism="subtract")
        self.lif2 = LIFNeuron(beta=0.9, learn_beta=True, threshold=1.0, reset_mechanism="subtract")
        self.lif3 = LIFNeuron(beta=0.9, learn_beta=True, threshold=1.0, reset_mechanism="subtract")
        self.netvlad = NetVLAD(num_clusters=num_clusters, dim=256)
        self.classifier = nn.Linear(256 * num_clusters, num_classes)
        
        print(f"模型初始化完成，输入尺寸：{input_size}，特征图尺寸：{feature_size}")
        
    def forward(self, x):
        batch_size = x.size(0)
        device = x.device
        mem1 = mem2 = mem3 = None
        final_spike = None
        for t in range(self.timesteps):
            x_t = x[:, t]
            conv1_out = self.conv1(x_t)
            spike1, mem1 = self.lif1(conv1_out, mem1)
            conv2_out = self.conv2(spike1)
            spike2, mem2 = self.lif2(conv2_out, mem2)
            conv3_out = self.conv3(spike2)
            spike3, mem3 = self.lif3(conv3_out, mem3)
            if t == self.timesteps - 1:
                final_spike = spike3
        vlad_features = self.netvlad(final_spike)
        output = self.classifier(vlad_features)
        return output

# 数据集
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


# 获取数据
def get_npy_files(data_dir):
    file_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.npy')]
    return file_paths


data_dir = r'/cluster/home2/zbs/icra/data/event/train/chop_still_50_firstcamera_0'
file_paths = get_npy_files(data_dir)

train_dataset = EventCameraDataset(file_paths)
train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=4)

val_dir = r'/cluster/home2/zbs/icra/vprsnn/VPRSNN-main/data/event/val/chop_still_50_firstcamera_1'

# 使用 Spiking VGG 和 NetVLAD
spiking_neuron = neuron.LIFNode
model = SpikingVGG(timesteps =50) .to(device)  # 使用 **kwargs

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=opt.lr)

# Modify the train function to log information
def train(epoch, max_grad_norm=1.0):
    epoch_start_time = time.time()
    epoch_loss = 0
    model.train()

    for iteration, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, labels)
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        epoch_loss += loss.item()

        if iteration % 50 == 0:
            log_message = f"Epoch [{epoch + 1}/{opt.nEpochs}], Iteration [{iteration + 1}/{len(train_loader)}], Loss: {loss.item():.4f}"
            print(log_message)
            logging.info(log_message)

    avg_loss = epoch_loss / len(train_loader)
    epoch_time = time.time() - epoch_start_time
    log_message = f"===> Epoch {epoch + 1} Complete: Avg. Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s"
    print(log_message)
    logging.info(log_message)

    # Save model checkpoint every 20 epochs
    if (epoch + 1) % 20 == 0:
        checkpoint_path = f"/cluster/home2/zbs/icra/models/spikeVGG/event_epoch_{epoch + 1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        logging.info(f"Checkpoint saved at {checkpoint_path}")


def validate(epoch):
    val_dataset = EventCameraDataset(get_npy_files(val_dir))
    val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False, num_workers=4)
    
    model.eval()
    val_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            
            # Collect all predictions and labels for calculating metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert lists to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = 100.0 * np.mean(all_predictions == all_labels)
    
    # For multi-class classification, we need to specify the averaging method
    # 'macro' gives equal weight to each class
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    avg_val_loss = val_loss / len(val_loader)
    
    # Log all metrics
    log_message = f"\nValidation Results - Epoch: {epoch+1}"
    log_message += f"\n  Loss: {avg_val_loss:.4f}"
    log_message += f"\n  Accuracy: {accuracy:.2f}%"
    log_message += f"\n  Precision: {precision:.4f}"
    log_message += f"\n  Recall: {recall:.4f}"
    log_message += f"\n  F1-score: {f1:.4f}"
    
    print(log_message)
    logging.info(log_message)
    
    # Log confusion matrix (as a formatted string)
    logging.info("Confusion Matrix:")
    cm_string = np.array2string(conf_matrix, separator=', ')
    logging.info(cm_string)
    
    # Save metrics to a separate CSV file for easy plotting
    metrics_dir = "/cluster/home2/zbs/icra/logs/metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_file = os.path.join(metrics_dir, "validation_metrics.csv")
    
    # Create file with headers if it doesn't exist
    if not os.path.exists(metrics_file):
        with open(metrics_file, 'w') as f:
            f.write("epoch,loss,accuracy,precision,recall,f1_score\n")
    
    # Append metrics for this epoch
    with open(metrics_file, 'a') as f:
        f.write(f"{epoch+1},{avg_val_loss:.6f},{accuracy:.6f},{precision:.6f},{recall:.6f},{f1:.6f}\n")
    
    # Also calculate per-class metrics for more detailed analysis
    # This is especially useful for imbalanced datasets
    num_classes = len(np.unique(all_labels))
    per_class_precision = precision_score(all_labels, all_predictions, average=None, zero_division=0)
    per_class_recall = recall_score(all_labels, all_predictions, average=None, zero_division=0)
    per_class_f1 = f1_score(all_labels, all_predictions, average=None, zero_division=0)
    
    # Log per-class metrics
    logging.info("Per-class metrics:")
    for i in range(num_classes):
        class_metrics = f"Class {i}: Precision={per_class_precision[i]:.4f}, Recall={per_class_recall[i]:.4f}, F1-score={per_class_f1[i]:.4f}"
        logging.info(class_metrics)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'loss': avg_val_loss
    }

# Update the main training loop to include validation
if __name__ == '__main__':
    logging.info("Starting training...")
    best_metrics = {
        'accuracy': 0.0,
        'f1': 0.0,
        'epoch': 0
    }
    
    summary_file = os.path.join(log_dir, f"training_summary_{current_time}.txt")
    
    for epoch in range(opt.nEpochs):
        train(epoch)
        
        # Run validation every 5 epochs
        if (epoch + 1) % 5 == 0:
            metrics = validate(epoch)
            
            # Save model if accuracy improves
            if metrics['accuracy'] > best_metrics['accuracy']:
                best_metrics['accuracy'] = metrics['accuracy']
                best_metrics['precision'] = metrics['precision']
                best_metrics['recall'] = metrics['recall']
                best_metrics['f1'] = metrics['f1']
                best_metrics['loss'] = metrics['loss']
                best_metrics['epoch'] = epoch + 1
                
                best_model_path = f"/cluster/home2/zbs/icra/models/spikeVGG/event_best_model_acc.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': metrics,
                }, best_model_path)
                logging.info(f"New best model (accuracy) saved with accuracy: {metrics['accuracy']:.2f}%")
            
            # Also save model if F1 score improves (especially important for imbalanced datasets)
            if metrics['f1'] > best_metrics['f1']:
                if best_metrics['epoch'] != epoch + 1:  # Don't duplicate log if same as accuracy best
                    best_model_path_f1 = f"/cluster/home2/zbs/icra/models/event_best_model_f1.pth"
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'metrics': metrics,
                    }, best_model_path_f1)
                    logging.info(f"New best model (F1-score) saved with F1: {metrics['f1']:.4f}")
                best_metrics['f1'] = metrics['f1']
    
    # At the end of training, save a summary with the best results
    with open(summary_file, 'w') as f:
        f.write("=== Training Summary ===\n")
        f.write(f"Total epochs: {opt.nEpochs}\n")
        f.write(f"Best results (at epoch {best_metrics['epoch']}):\n")
        f.write(f"  Accuracy: {best_metrics['accuracy']:.2f}%\n")
        f.write(f"  Precision: {best_metrics['precision']:.4f}\n")
        f.write(f"  Recall: {best_metrics['recall']:.4f}\n")
        f.write(f"  F1-score: {best_metrics['f1']:.4f}\n")
        f.write(f"  Loss: {best_metrics['loss']:.4f}\n")
    
    logging.info(f"Training completed. Best validation accuracy: {best_metrics['accuracy']:.2f}% at epoch {best_metrics['epoch']}")
    logging.info(f"Best F1-score: {best_metrics['f1']:.4f}")
    logging.info(f"Training summary saved to {summary_file}")
