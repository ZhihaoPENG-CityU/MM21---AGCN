# Attention-driven Graph Clustering Network
DOI: 10.1145/3474085.3475276

URL: https://dl.acm.org/doi/abs/10.1145/3474085.3475276

We have added comments in the code, the specific details can correspond to the explanation in the paper.

We appreciate it if you use this code and cite our paper, which can be cited as follows,
> @article{peng2021attention, <br>
>   author={Peng, Zhihao and Liu, Hui and Jia, Yuheng and Hou, Junhui}, <br>
>   journal={arXiv preprint arXiv:2108.05499},  <br>
>   title={Attention-driven Graph Clustering Network},  <br>
>   doi={10.1145/3474085.3475276} <br>
> } <br>

# Environment
+ Python[3.6.12]
+ Pytorch[1.7.1]
+ GPU (GeForce RTX 2080 Ti) & (NVIDIA GeForce RTX 3090) & (Quadro RTX 8000)

# Hyperparameters
+ The learning rates of USPS, HHAR, ACM, and DBLP datasets are set to 0.001, and the learning rates of Reuters and CiteSeer datasets are set to 0.0001. lambda1 and lambda2 are set to {1000, 1000} for USPS, {1, 0.1} for HHAR, {10, 10} for Reuters, and {0.1, 0.01} for graph datasets.

# To run code
+ python AGCN.py --name [data_name]
+ For examle, if u would like to run AGCN on the usps dataset, the command is as follows,
  + python AGCN.py --name usps

# Data
Due to the limitation of GitHub, we share the data in [<a href="https://drive.google.com/drive/folders/1swVtlqQkLFEmu9l2QXEQS6Hmw20q0QTc?usp=sharing">here</a>].
