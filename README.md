# SNN for visual place recognition with NetVlad:
## A visual place recognition method based on spiking VGG16 and event camera. 
This repository is part of project [Hybrid Guided VAE for Rapid Visual Place Recognition](https://github.com/niart/fzj_vpr). 
Download our dataset **[HERE](https://drive.google.com/drive/folders/1oC8KnzzZXLAF_QzLBpGEebBqCXU_yTTT?usp=sharing)**. 
Contact us at **niwang.cs@gmail.com** if you have any inquiry.

### Steps of implementing this repository:
#### 1. Setup environment and dependencies: @Boshi please complete this part by making a yml file.
#### 2. Test this repository with the model trained on our dataset: @Boshi please complete this part. Ideally it
#### 3. Train this model yourself with our training set: @Boshi
#### 4. Test the generalization capability of the trained model on our additonal dataset:   
1. Get representation vectors:
2. visualize the distribution of representation vectors with TSNE:
```python generalization_tsne.py``` 


to your ```~/.bashrc```, and then ```source ~/.bashrc```
3. Use Anaconda to create a virtual environment 'triple_sumo' with ```conda env create -f triplesumo2021.yml```; Then ```conda activate triple_sumo```.
4. ```git clone https://github.com/niart/triplesumo.git``` and ```cd triplesumo```
5. Use the ```envs``` foler of this repository to replace the ```gym/envs``` installed in your conda environment triplesumo. 
6. To train blue agent in an ongoing game between red and green, run ```cd train_bug```, then```python runmain2.py```. 
7. If you meet error ```Creating window glfw ... ERROR: GLEW initalization error: Missing GL version```, you may add ```export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so``` to ```~/.bashrc```, then ```source ~/.bashrc```. 

key algorithm:
The reward function is in ```gym/envs/mojuco/triant.py```;
The training algorithm is in ```train_bug/DDPG4.py```.

If you use this repository or our dataset for academic work which results in publication, please cite:
```
@misc{triplesumo,
  howpublished = {Wang, N., Das, G.P., Millard, A.G. (2022). Learning Cooperative Behaviours in Adversarial Multi-agent Systems. In: Pacheco-Gutierrez, S., Cryer, A., Caliskanelli, I., Tugal, H., Skilton, R. (eds) Towards Autonomous Robotic Systems. TAROS 2022. Lecture Notes in Computer Science(), vol 13546. Springer, Cham. https://doi.org/10.1007/978-3-031-15908-4_15} 
```  

Test for generalization of the newly trained model on three new datasets:
<p align="center">
<img src="https://github.com/niart/spiking_VGG16_NetVlad/blob/41f7d54ff79bc87830d4819ba7d17ccc2ac938db/tsne_pictures/Screenshot%20from%202025-02-27%2005-45-24.png" width=60% height=60%>
</p>

