import warnings
warnings.filterwarnings("ignore")

import torch
import dgl.data
import argparse
import itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from src.graphModel import SAGEConv
from src.data_input import load_data
from sklearn.metrics import roc_auc_score,average_precision_score



device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--data_evaluation', type=str, default='mHSC-E', help='data_evaluation type',required=False)
parser.add_argument('--net', type=str, default='Specific', help='network type',required=False)
parser.add_argument('--num', type=int, default= 500, help='network scale',required=False)
parser.add_argument('--cell_size',type=float,default=0,help=' the drop rate of scRNA ',required=False)
parser.add_argument('--train_size',type=float,default=1,help='',required=False)
args = parser.parse_args()


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(1,h_feats, 'mean',3,1,0,bias=False)
        self.conv2 = SAGEConv(h_feats, h_feats*2, 'mean',3,1,0,bias=False)
        self.conv3 = SAGEConv(h_feats*2, h_feats*4, 'mean',3,1,0,bias=False)
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = nn.MaxPool2d(2)(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = nn.MaxPool2d(2)(h)
        h = self.conv3(g, h)
        h = F.relu(h)
        h = nn.MaxPool2d(2)(h)
        h = nn.Flatten()(h)
        return h


class MLPPredictor(nn.Module):
    def __init__(self, h_feats,out_put):
        super().__init__()
        self.W1 = nn.Linear(2*h_feats, out_put)
        self.W2 = nn.Linear(out_put, 1)

    def apply_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': torch.sigmoid(self.W2(torch.relu(self.W1(h)))).squeeze(1)}


    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']

def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]).to(device), torch.zeros(neg_score.shape[0]).to(device)])
    #labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy(scores, labels)


def compute_auc(pos_score, neg_score):
    pos_score = pos_score.to('cpu')
    neg_score = neg_score.to('cpu')
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores),average_precision_score(labels,scores)


net = args.net
type = args.data_evaluation
num = args.num
cell_size = args.cell_size
size = 32
flatten_size = 512
train_data,test_data,features,nums = load_data(net,type,num,size,cell_size,args.train_size)
src_pos,dst_pos = train_data[0],train_data[1]

train_g = dgl.graph((src_pos, dst_pos), num_nodes=nums)
features_ = torch.from_numpy(features)
features_ = features_.to(torch.float32)
train_g = dgl.to_bidirected(train_g)
train_g.ndata['feature'] = features_



train_pos_u, train_pos_v = src_pos,dst_pos
train_neg_u, train_neg_v = train_data[2],train_data[3]

test_pos_u, test_pos_v = test_data[0],test_data[1]
test_neg_u, test_neg_v = test_data[2],test_data[3]

train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=train_g.number_of_nodes())
train_pos_g = dgl.to_bidirected(train_pos_g)
train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=train_g.number_of_nodes())
train_neg_g = dgl.to_bidirected(train_neg_g)

test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=train_g.number_of_nodes())
test_pos_g = dgl.to_bidirected(test_pos_g)
test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=train_g.number_of_nodes())
test_neg_g = dgl.to_bidirected(test_neg_g)

total_auc = []
total_auprc = []

for i in range(1):
    model = GraphSAGE(train_g.ndata['feature'].shape[-1], 32)
    # 可以使用自定义的MLPPredictor代替DotPredictor

    pred = MLPPredictor(flatten_size,218)  # [32,512]  [64,4608]
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.001)

    #GPU Setting#
    train_g = train_g.to(device)
    train_pos_g = train_pos_g.to(device)
    train_neg_g = train_neg_g.to(device)
    test_neg_g = test_neg_g.to(device)
    test_pos_g = test_pos_g.to(device)
    model = model.to(device)
    pred = pred.to(device)

    for e in range(1):
        # 前向传播
        h = model(train_g, train_g.ndata['feature'])
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)
        loss = compute_loss(pos_score, neg_score)

        # 后向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if(loss<0.5):
        #     break;
        # if e % 5 == 0:
        #     with torch.no_grad():
        #         pos_score = pred(test_pos_g, h)
        #         neg_score = pred(test_neg_g, h)
        #         auc = compute_auc(pos_score, neg_score)
        #         print("epochs{} AUC: {}".format(e, auc))

    # 检测结果准确

    with torch.no_grad():
        #h = model(train_g, train_g.ndata['feature'])
        pos_score = pred(test_pos_g, h)
        neg_score = pred(test_neg_g, h)
        auc,aupr = compute_auc(pos_score, neg_score)
        total_auc.append(auc)
        total_auprc.append(aupr)
        #print("{} {} {} AUC: {}  AUPR{}".format(net,type,num,auc,aupr))

total_auc = np.array(total_auc)
total_auprc = np.array(total_auprc)
auc = total_auc.mean()
aupr = total_auprc.mean()
print("%s,%s,%d,%.4f,%.4f,%.4f,%.4f" % (net,type,num,cell_size,args.train_size,auc,aupr))
# df_auc = pd.DataFrame(total_auc)
# df_aupr = pd.DataFrame(total_auprc)
# df_auc.to_csv("./output/{}_{}_{}_auroc.csv".format(net,type,num))
# df_aupr.to_csv("./output/{}_{}_{}_auprc.csv".format(net,type,num))