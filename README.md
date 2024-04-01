# Attention-driven Graph Clustering Network

[python-img]: https://img.shields.io/github/languages/top/ZhihaoPENG-CityU/MM21---AGCN?color=lightgrey
[stars-img]: https://img.shields.io/github/stars/ZhihaoPENG-CityU/MM21---AGCN?color=yellow
[stars-url]: https://github.com/ZhihaoPENG-CityU/MM21---AGCN/stargazers
[fork-img]: https://img.shields.io/github/forks/ZhihaoPENG-CityU/MM21---AGCN?color=lightblue&label=fork
[fork-url]: https://github.com/ZhihaoPENG-CityU/MM21---AGCN/network/members
[visitors-img]: https://visitor-badge.glitch.me/badge?page_id=ZhihaoPENG-CityU.MM21---AGCN
[agcn-url]: https://github.com/ZhihaoPENG-CityU/MM21---AGCN

[![Made with Python][python-img]][agcn-url]
[![GitHub stars][stars-img]][stars-url]
[![GitHub forks][fork-img]][fork-url]
[![visitors][visitors-img]][agcn-url]



DOI: 10.1145/3474085.3475276

URL: https://dl.acm.org/doi/abs/10.1145/3474085.3475276

VIDEO: https://dl.acm.org/doi/abs/10.1145/3474085.3475276 

We have added comments in the code, and the specific details can correspond to the explanation in the paper. Please get in touch with me (zhihapeng3-c@my.cityu.edu.hk) if you have any issues.

We appreciate it if you use this code and cite our related papers, which can be cited as follows,

> @article{peng2023egrc, <br>
>   title={EGRC-Net: Embedding-Induced Graph Refinement Clustering Network}, <br>
>   author={Peng, Zhihao and Liu, Hui and Jia, Yuheng and Hou, Junhui},  <br>
>   journal={IEEE Transactions on Image Processing}, <br>
>   volume={32}, <br>
>   pages={6457--6468}, <br>
>   year={2023}, <br>
>   publisher={IEEE}
> } <br>

> @article{peng2022deep, <br>
>   title={Deep Attention-guided Graph Clustering with Dual Self-supervision}, <br>
>   author={Peng, Zhihao and Liu, Hui and Jia, Yuheng and Hou, Junhui},  <br>
>   journal={IEEE Transactions on Circuits and Systems for Video Technology},  <br>
>   year={2022}, <br>
>   publisher={IEEE}
> } <br>

> @inproceedings{peng2021attention, <br>
>   title={Attention-driven graph clustering network}, <br>
>   author={Peng, Zhihao and Liu, Hui and Jia, Yuheng and Hou, Junhui},  <br>
>   booktitle={Proceedings of the 29th ACM International Conference on Multimedia},  <br>
>   pages={935--943},  <br>
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

# Evaluation
+ evaluation.py
👉
The commonly used clustering metrics, such as acc, nmi, ari, and f1, etc.
+ get_net_par_num.py
👉
Get the network parameters by `print(num_net_parameter(model))', where model is the designed network.

# Data
Due to the limitation of GitHub, we share the data in [<a href="https://drive.google.com/drive/folders/1D_kH2loUTH6fHfdwnVElUHVw1kHfflVV?usp=sharing">here</a>].


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ZhihaoPENG-CityU/MM21---AGCN&type=Date)](https://star-history.com/#ZhihaoPENG-CityU/MM21---AGCN&Date)
