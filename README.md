# Attention-driven Graph Clustering Network

[stars-img]: https://img.shields.io/github/stars/ZhihaoPENG-CityU/MM21---AGCN?color=yellow
[stars-url]: https://github.com/ZhihaoPENG-CityU/MM21---AGCN/stargazers
[fork-img]: https://img.shields.io/github/forks/ZhihaoPENG-CityU/MM21---AGCN?color=lightblue&label=fork
[fork-url]: https://github.com/ZhihaoPENG-CityU/MM21---AGCN/network/members
[visitors-img]: https://github.com/ZhihaoPENG-CityU/MM21---AGCN/watchers
[adgc-url]: https://github.com/ZhihaoPENG-CityU/MM21---AGCN/watchers

[![GitHub stars][stars-img]][stars-url]
[![GitHub forks][fork-img]][fork-url]
[![visitors][visitors-img]][adgc-url]

DOI: 10.1145/3474085.3475276

URL: https://dl.acm.org/doi/abs/10.1145/3474085.3475276

VIDEO: https://dl.acm.org/doi/abs/10.1145/3474085.3475276 

We have added comments in the code, the specific details can correspond to the explanation in the paper.

We appreciate it if you use this code and cite our paper, which can be cited as follows,
> @inproceedings{peng2021attention, <br>
>   title={Attention-driven Graph Clustering Network}, <br>
>   author={Peng, Zhihao and Liu, Hui and Jia, Yuheng and Hou, Junhui},  <br>
>   booktitle={Proceedings of the 29th ACM International Conference on Multimedia},  <br>
>   pages={935--943}, <br>
>   year={2021}
> } <br>

# Environment
+ Python[3.6.12]
+ Pytorch[1.7.1]
+ GPU (GeForce RTX 2080 Ti) & (NVIDIA GeForce RTX 3090) & (Quadro RTX 8000)

# Hyperparameters
+ The learning rates of USPS, HHAR, ACM, and DBLP datasets are set to 0.001, and the learning rates of Reuters and CiteSeer datasets are set to 0.0001. lambda1 and lambda2 are set to {1000, 1000} for USPS, {1, 0.1} for HHAR, {10, 10} for Reuters, and {0.1, 0.01} for graph datasets.

# To run code
+ Step 1: set the hyperparameters for the specific dataset;
+ Step 2: python AGCN.py --name [data_name]
* For examle, if u would like to run AGCN on the usps dataset, u need to
* first set {1000, 1000} for USPS;
* then run the command "python AGCN.py --name usps"

# Data
Due to the limitation of GitHub, we share the data in [<a href="https://drive.google.com/drive/folders/1D_kH2loUTH6fHfdwnVElUHVw1kHfflVV?usp=sharing">here</a>].
