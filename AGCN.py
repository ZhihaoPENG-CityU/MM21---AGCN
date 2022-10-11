from __future__ import print_function, division
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from utils import load_data, load_graph
from GNN import GNNLayer
from evaluation import eva
from collections import Counter
from datetime import datetime
import time
import scipy.io as scio
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from get_net_par_num import num_net_parameter

tic = time.time()
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

class AE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        # encoder
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        # extracted feature by AE
        self.z_layer = Linear(n_enc_3, n_z)
        # decoder
        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)
    def forward(self, x):
        enc_z2 = F.relu(self.enc_1(x))
        enc_z3 = F.relu(self.enc_2(enc_z2))
        enc_z4 = F.relu(self.enc_3(enc_z3))
        z = self.z_layer(enc_z4)
        dec_z2 = F.relu(self.dec_1(z))
        dec_z3 = F.relu(self.dec_2(dec_z2))
        dec_z4 = F.relu(self.dec_3(dec_z3))
        x_bar = self.x_bar_layer(dec_z4)

        return x_bar, enc_z2, enc_z3, enc_z4, z

class MLP_L(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_L, self).__init__()
        self.wl = Linear(n_mlp, 5)
    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.wl(mlp_in)), dim=1)
        
        return weight_output

class MLP_1(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_1, self).__init__()
        self.w1 = Linear(n_mlp,2)
    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w1(mlp_in)), dim=1) 
        
        return weight_output

class MLP_2(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_2, self).__init__()
        self.w2 = Linear(n_mlp, 2)
    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w2(mlp_in)), dim=1)
        
        return weight_output

class MLP_3(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_3, self).__init__()
        self.w3 = Linear(n_mlp, 2)
    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w3(mlp_in)), dim=1)  
        
        return weight_output

class AGCN(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, 
                n_input, n_z, n_clusters, v=1):
        super(AGCN, self).__init__()

        # AE
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        self.agcn_0 = GNNLayer(n_input, n_enc_1)
        self.agcn_1 = GNNLayer(n_enc_1, n_enc_2)
        self.agcn_2 = GNNLayer(n_enc_2, n_enc_3)
        self.agcn_3 = GNNLayer(n_enc_3, n_z)
        self.agcn_z = GNNLayer(3020,n_clusters)

        self.mlp = MLP_L(3020)

        # attention on [Z_i || H_i]
        self.mlp1 = MLP_1(2*n_enc_1)
        self.mlp2 = MLP_2(2*n_enc_2)
        self.mlp3 = MLP_3(2*n_enc_3)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj):
        # AE Module
        x_bar, h1, h2, h3, z = self.ae(x)

        x_array = list(np.shape(x))
        n_x = x_array[0]

        # # AGCN-H
        z1 = self.agcn_0(x, adj)
        # z2
        m1 = self.mlp1( torch.cat((h1,z1), 1) )
        m1 = F.normalize(m1,p=2)
        m11 = torch.reshape(m1[:,0], [n_x, 1])
        m12 = torch.reshape(m1[:,1], [n_x, 1])
        m11_broadcast =  m11.repeat(1,500)
        m12_broadcast =  m12.repeat(1,500)
        z2 = self.agcn_1( m11_broadcast.mul(z1)+m12_broadcast.mul(h1), adj)
        # z3
        m2 = self.mlp2( torch.cat((h2,z2),1) )     
        m2 = F.normalize(m2,p=2)
        m21 = torch.reshape(m2[:,0], [n_x, 1])
        m22 = torch.reshape(m2[:,1], [n_x, 1])
        m21_broadcast = m21.repeat(1,500)
        m22_broadcast = m22.repeat(1,500)
        z3 = self.agcn_2( m21_broadcast.mul(z2)+m22_broadcast.mul(h2), adj)
        # z4
        m3 = self.mlp3( torch.cat((h3,z3),1) )# self.mlp3(h2)      
        m3 = F.normalize(m3,p=2)
        m31 = torch.reshape(m3[:,0], [n_x, 1])
        m32 = torch.reshape(m3[:,1], [n_x, 1])
        m31_broadcast = m31.repeat(1,2000)
        m32_broadcast = m32.repeat(1,2000)
        z4 = self.agcn_3( m31_broadcast.mul(z3)+m32_broadcast.mul(h3), adj)

        # # AGCN-S
        u  = self.mlp(torch.cat((z1,z2,z3,z4,z),1))
        u = F.normalize(u,p=2) 
        u0 = torch.reshape(u[:,0], [n_x, 1])
        u1 = torch.reshape(u[:,1], [n_x, 1])
        u2 = torch.reshape(u[:,2], [n_x, 1])
        u3 = torch.reshape(u[:,3], [n_x, 1])
        u4 = torch.reshape(u[:,4], [n_x, 1])

        tile_u0 = u0.repeat(1,500)
        tile_u1 = u1.repeat(1,500)
        tile_u2 = u2.repeat(1,2000)
        tile_u3 = u3.repeat(1,10)
        tile_u4 = u4.repeat(1,10)

        net_output = torch.cat((tile_u0.mul(z1), tile_u1.mul(z2), tile_u2.mul(z3), tile_u3.mul(z4), tile_u4.mul(z)), 1 )   
        net_output = self.agcn_z(net_output, adj, active=False) 
        predict = F.softmax(net_output, dim=1)
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z, net_output

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def train_AGCN(dataset):

    dataname = 'usps'
    eprm_state = 'result'

    file_out = open('./output/'+dataname+'_'+eprm_state+'.out', 'a')
    print("The experimental results", file=file_out)

    # hyper parameters
    lambda_1 = [1000] #[0.001,0.01,0.1,1,10,100,1000]
    lambda_2 = [1000] #[0.001,0.01,0.1,1,10,100,1000]
    for ld1 in lambda_1:
        for ld2 in lambda_2:
            print("lambda_1: ", ld1, "lambda_2: ", ld2, file=file_out)
            model = AGCN(500, 500, 2000, 2000, 500, 500,
                        n_input=args.n_input,
                        n_z=args.n_z,
                        n_clusters=args.n_clusters,
                        v=1.0).cuda()

            # Get the network parameters
            print(num_net_parameter(model))

            optimizer = Adam(model.parameters(), lr=args.lr)

            # KNN Graph
            adj = load_graph(args.name, args.k)
            adj = adj.cuda()

            # cluster parameter initiate
            data = torch.Tensor(dataset.x).cuda()
            y = dataset.y
            with torch.no_grad():
                _, _, _, _, z = model.ae(data)

            iters10_kmeans_iter_Q = []
            iters10_NMI_iter_Q = []
            iters10_ARI_iter_Q = []
            iters10_F1_iter_Q = []

            iters10_kmeans_iter_Z = []
            iters10_NMI_iter_Z = []
            iters10_ARI_iter_Z = []
            iters10_F1_iter_Z = []

            iters10_kmeans_iter_P = []
            iters10_NMI_iter_P = []
            iters10_ARI_iter_P = []
            iters10_F1_iter_P = []

            z_1st = z

            for i in range(1):

                kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
                y_pred = kmeans.fit_predict(z_1st.data.cpu().numpy())
                y_pred_last = y_pred
                model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).cuda()
                acc,nmi,ari,f1 = eva(y, y_pred, 'pae')

                # get the value
                kmeans_iter_Q = []
                NMI_iter_Q = []
                ARI_iter_Q = []
                F1_iter_Q = []

                kmeans_iter_Z = []
                NMI_iter_Z = []
                ARI_iter_Z = []
                F1_iter_Z = []

                kmeans_iter_P = []
                NMI_iter_P = []
                ARI_iter_P = []
                F1_iter_P = []

                for epoch in range(200):

                    if epoch % 1 == 0:
                        _, tmp_q, pred, _, _ = model(data, adj)
                        tmp_q = tmp_q.data
                        p = target_distribution(tmp_q)
                    
                        res1 = tmp_q.cpu().numpy().argmax(1)       #Q
                        res2 = pred.data.cpu().numpy().argmax(1)   #Z
                        res3 = p.data.cpu().numpy().argmax(1)      #P

                        acc,nmi,ari,f1 = eva(y, res1, str(epoch) + 'Q')
                        kmeans_iter_Q.append(acc)
                        NMI_iter_Q.append(nmi)
                        ARI_iter_Q.append(ari)
                        F1_iter_Q.append(f1 )

                        acc,nmi,ari,f1 = eva(y, res2, str(epoch) + 'Z')
                        kmeans_iter_Z.append(acc)
                        NMI_iter_Z.append(nmi)
                        ARI_iter_Z.append(ari)
                        F1_iter_Z.append(f1)

                        acc,nmi,ari,f1 = eva(y, res3, str(epoch) + 'P')
                        kmeans_iter_P.append(acc)
                        NMI_iter_P.append(nmi)
                        ARI_iter_P.append(ari)
                        F1_iter_P.append(f1)

                    x_bar, q, pred, _, _ = model(data, adj)

                    kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
                    ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
                    re_loss = F.mse_loss(x_bar, data)

                    loss = ld1 * kl_loss + ld2 * ce_loss + re_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # _Q
                kmeans_max= np.max(kmeans_iter_Q)
                nmi_max= np.max(NMI_iter_Q)
                ari_max= np.max(ARI_iter_Q)
                F1_max= np.max(F1_iter_Q)
                iters10_kmeans_iter_Q.append(round(kmeans_max,5))
                iters10_NMI_iter_Q.append(round(nmi_max,5))
                iters10_ARI_iter_Q.append(round(ari_max,5))
                iters10_F1_iter_Q.append(round(F1_max,5))

                # _Z
                kmeans_max= np.max(kmeans_iter_Z)
                nmi_max= np.max(NMI_iter_Z)
                ari_max= np.max(ARI_iter_Z)
                F1_max= np.max(F1_iter_Z)
                iters10_kmeans_iter_Z.append(round(kmeans_max,5))
                iters10_NMI_iter_Z.append(round(nmi_max,5))
                iters10_ARI_iter_Z.append(round(ari_max,5))
                iters10_F1_iter_Z.append(round(F1_max,5))

                # _P
                kmeans_max= np.max(kmeans_iter_P)
                nmi_max= np.max(NMI_iter_P)
                ari_max= np.max(ARI_iter_P)
                F1_max= np.max(F1_iter_P)
                iters10_kmeans_iter_P.append(round(kmeans_max,5))
                iters10_NMI_iter_P.append(round(nmi_max,5))
                iters10_ARI_iter_P.append(round(ari_max,5))
                iters10_F1_iter_P.append(round(F1_max,5))

            print("#####################################", file=file_out)
            print("kmeans Z mean",round(np.mean(iters10_kmeans_iter_Z),5),"max",np.max(iters10_kmeans_iter_Z),"\n",iters10_kmeans_iter_Z, file=file_out)
            print("NMI mean",round(np.mean(iters10_NMI_iter_Z),5),"max",np.max(iters10_NMI_iter_Z),"\n",iters10_NMI_iter_Z, file=file_out)
            print("ARI mean",round(np.mean(iters10_ARI_iter_Z),5),"max",np.max(iters10_ARI_iter_Z),"\n",iters10_ARI_iter_Z, file=file_out)
            print("F1  mean",round(np.mean(iters10_F1_iter_Z),5),"max",np.max(iters10_F1_iter_Z),"\n",iters10_F1_iter_Z, file=file_out)
            print(':acc, nmi, ari, f1: \n{:.4f}\n{:.4f}\n{:.4f}\n{:.4f}'.format(round(np.mean(iters10_kmeans_iter_Z),5),round(np.mean(iters10_NMI_iter_Z),5),round(np.mean(iters10_ARI_iter_Z),5),round(np.mean(iters10_F1_iter_Z),5)), file=file_out)

    file_out.close()

if __name__ == "__main__":
    # iters
    iters = 10

    for iter_num in range(iters):
        print(iter_num)
        parser = argparse.ArgumentParser(
            description='train',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--name', type=str, default='usps')
        parser.add_argument('--k', type=int, default=3)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--n_clusters', default=3, type=int)
        parser.add_argument('--n_z', default=10, type=int)
        parser.add_argument('--pretrain_path', type=str, default='pkl')
        args = parser.parse_args()
        args.cuda = torch.cuda.is_available()
        print("use cuda: {}".format(args.cuda))
        device = torch.device("cuda" if args.cuda else "cpu")

        args.pretrain_path = 'data/{}.pkl'.format(args.name)
        dataset = load_data(args.name)

        if args.name == 'usps' or args.name == 'usps_for_np':
            args.n_clusters = 10
            args.n_input = 256

        if args.name == 'hhar':
            args.k = 5
            args.n_clusters = 6
            args.n_input = 561

        if args.name == 'reut':
            args.lr = 1e-4
            args.n_clusters = 4
            args.n_input = 2000

        if args.name == 'acm'or args.name == 'acm_for_np':
            args.k = None
            args.n_clusters = 3
            args.n_input = 1870

        if args.name == 'dblp' or args.name == 'dblp_for_np':
            args.k = None
            args.n_clusters = 4
            args.n_input = 334

        if args.name == 'cite':
            args.lr = 1e-4
            args.k = None
            args.n_clusters = 6
            args.n_input = 3703

        print(args)
        train_AGCN(dataset)

    toc = time.time()
    print("Time:", (toc - tic))
    
