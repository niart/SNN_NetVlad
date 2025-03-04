# SNN for visual place recognition with NetVlad:
## A visual place recognition method based on spiking VGG16 and event camera.
This repository is part of project [Hybrid Guided VAE for Rapid Visual Place Recognition](https://github.com/niart/fzj_vpr). 
A spiking VGG attached NetVlad layer is trained with Backpropagation Through Time (BPTT) to recognize 16 classes of indoor visual scenes in an artificial office environment captured by event camera, and tested for generalization on three additional unseen places. Download our dataset **[HERE](https://drive.google.com/drive/folders/1oC8KnzzZXLAF_QzLBpGEebBqCXU_yTTT?usp=sharing)**. 
We also trained a normal VGG16 attached NetVlad layer on our RGB dataset for comparison. 
Contact us at **niwang.cs@gmail.com** if you have any inquiry.

### Steps of implementing this repository:
#### 1. Setup environment and dependencies: 
@Boshi please complete this part by making a .yml file for Anaconda.
#### 2. Test this repository with the model trained on our dataset: 
```python eventVal.py``` 

@Boshi please complete this part. This step should output classification accuracy for testing and a .npy file containing representation vectors.
#### 3. Train this model yourself with our training set: 
@Boshi please complete this step. This step should output training graphs and a .pth file.
#### 4. Test the generalization capability of the trained model on our additonal dataset:   
<p align="center">
<img src="https://github.com/niart/SNN_NetVlad/blob/961c990b1358c30304af79b7fcd4911d8ecdb0d7/Pasted%20image.png" width=100% height=60%>
</p>

1. Get representation vectors:
```python event.py``` 
2. visualize the distribution of representation vectors with TSNE:
```python generalization_tsne.py``` 
A typical result of this step:
<p align="center">
<img src="https://github.com/niart/spiking_VGG16_NetVlad/blob/41f7d54ff79bc87830d4819ba7d17ccc2ac938db/tsne_pictures/Screenshot%20from%202025-02-27%2005-45-24.png" width=60% height=60%>
</p>

#### 5. Train/test NetVlad with VGG16 based on our RGB dataset:
```python rbg.py``` 

Modify the path to feature vectors and visualize the distribution of representation vectors with TSNE:
```python generalization_tsne.py``` 
A typical result of this step:
<p align="center">
<img src="https://github.com/niart/SNN_NetVlad/blob/fbacdd9686ab8fcd10f0230fe261060d72a7c240/tsne_pictures/rgb_features.png" width=60% height=60%>
</p>

If you use this repository or our dataset for academic work which results in publication, please cite:
```
@misc{triplesumo,
  howpublished = {Wang, N., Das, G.P., Millard, A.G. (2022). Learning Cooperative Behaviours in Adversarial Multi-agent Systems. In: Pacheco-Gutierrez, S., Cryer, A., Caliskanelli, I., Tugal, H., Skilton, R. (eds) Towards Autonomous Robotic Systems. TAROS 2022. Lecture Notes in Computer Science(), vol 13546. Springer, Cham. https://doi.org/10.1007/978-3-031-15908-4_15} 
```  
