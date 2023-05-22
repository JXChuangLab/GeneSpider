import sys
import random
import numpy as np
import pandas as pd

from pyts.image import GramianAngularField


def map_1d_to_2d(vec, m):
    # 对向量中的元素进行排序
    n = len(vec)

    # 将向量平均分成 m x m 份
    part_size = n // (m * m)
    parts = []
    for i in range(m * m):
        start = i * part_size
        end = start + part_size
        if i == m * m - 1:
            end = n
        parts.append(vec[start:end])

    # 对每份中的元素进行排序
    sorted_parts = parts

    # 将有序向量填充到二维矩阵中
    matrix = np.zeros((m, m))
    for i in range(m * m):
        row = i // m
        col = i % m
        idx = i // part_size
        matrix[row, col] = sorted_parts[idx][i % part_size]

    return matrix

def getmatrix(vec,m):
    ret = np.zeros((len(vec),1,m,m))
    for i in range(len(vec)):
        tmp = map_1d_to_2d(vec[i,:],m)
        ret[i,0,:,:] = tmp
    return ret
def drop_exp(row,rate):
    if rate == 0:
        return row
    size = len(row)
    idxs = [i for i in range(size)]
    idxs = random.sample(idxs,int(size*rate))
    row[idxs] = 0
    return row
def load_data(net,type,nums,size=32,rate=0,train_size=1,netcode=False):

    exp_path = "./data_evaluation/Benchmark Dataset/"+net+" Dataset/"+type+"/TFs+"+str(nums)+"/BL--ExpressionData.csv"
    geneids_path = "./data_evaluation/Benchmark Dataset/"+net+" Dataset/" + type + "/TFs+" + str(nums) + "/Target.csv"

    train_set_path = "./data_evaluation/TrainTest/"+net+"/"+type+" "+str(nums)+"/"+"Train_set.csv"
    valid_set_path = "./data_evaluation/TrainTest/"+net+"/"+type+" "+str(nums)+"/"+"Validation_set.csv"
    test_set_path = "./data_evaluation/TrainTest/"+net+"/"+type+" "+str(nums)+"/"+"Test_set.csv"

    exp = pd.read_csv(exp_path,sep=',',index_col=0)
    l = len(exp.columns)
    idxs = [i for i in range(l)]
    idxs = random.sample(idxs, int(l * rate))
    exp.values[:,idxs] = 0.0
    #exp = exp.apply(drop_exp,axis=1,rate=rate)

    train = pd.read_csv(train_set_path,sep=',',index_col=0)
    valid = pd.read_csv(valid_set_path,sep=',',index_col=0)
    train = pd.concat([train,valid],axis=0)
    test = pd.read_csv(test_set_path,sep=',',index_col=0)

    genes_ids = pd.read_csv(geneids_path,sep=',',index_col=0)
    features = genes_ids['Gene'].values.tolist()

    features = exp.loc[features]
    if netcode == False:
        gas = GramianAngularField(image_size=size, method='s')
        features = gas.fit_transform(features)
        features = np.expand_dims(features, axis=1)

    src_pos_ids = train[train['Label']==1]['TF'].values
    dst_pos_ids = train[train['Label']==1]['Target'].values
    src_neg_ids = train[train['Label']==0]['TF'].values
    dst_neg_ids = train[train['Label']==0]['Target'].values

    if(train_size > 1 or train_size < 0):
        print("train_size out of bound")
        sys.exit(1)
    # 打乱顺序
    indexs_pos = np.arange(len(src_pos_ids))
    indexs_pos = np.random.choice(indexs_pos, int(len(src_pos_ids) * train_size),replace=False).tolist()
    indexs_neg = np.arange(len(src_neg_ids))
    indexs_neg = np.random.choice(indexs_neg, int(len(src_neg_ids) * train_size), replace=False).tolist()
    train_data = []
    train_data.append(src_pos_ids[indexs_pos].tolist())
    train_data.append(dst_pos_ids[indexs_pos].tolist())
    train_data.append(src_neg_ids[indexs_neg].tolist())
    train_data.append(dst_neg_ids[indexs_neg].tolist())

    test_data = []
    test_src_pos_ids = test[test['Label']==1]['TF'].values.tolist()
    test_dst_pos_ids = test[test['Label']==1]['Target'].values.tolist()
    test_src_neg_ids = test[test['Label']==0]['TF'].values.tolist()
    test_dst_neg_ids = test[test['Label']==0]['Target'].values.tolist()

    test_data.append(test_src_pos_ids)
    test_data.append(test_dst_pos_ids)
    test_data.append(test_src_neg_ids)
    test_data.append(test_dst_neg_ids)

    return train_data,test_data,features,len(genes_ids)

def load_breast_cancer():
    data = pd.read_csv("./data_evaluation/Breast Cancer/CCLE_expression_withinSample.csv",index_col=0)
    prio_net = pd.read_csv("./data_evaluation/Breast Cancer/train_df.csv",index_col=0)
    valid_net = pd.read_csv("./data_evaluation/Breast Cancer/valid_df.csv",index_col=0)
    prio_net['value'] = prio_net['value'].astype(int)
    valid_net['value'] = valid_net['value'].astype(int)

    prio_net = prio_net[prio_net['Tf']!="HKR1"]
    prio_net = prio_net[prio_net['Target'] != "HKR1"]
    valid_net = valid_net[valid_net['Tf'] != "HKR1"]
    valid_net = valid_net[valid_net['Target'] != "HKR1"]

    TFs = prio_net['Tf'].values.tolist()
    TFs.extend(valid_net['Tf'].values.tolist())

    Tatgets = prio_net['Target'].values.tolist()
    Tatgets.extend(valid_net['Target'].values.tolist())

    genes = set(TFs+Tatgets)
    genes = list(genes)
    data = data.loc[genes]
    gas = GramianAngularField(image_size=32, method='s')
    features = gas.fit_transform(data.values)
    features = np.expand_dims(features, axis=1)

    genes = pd.DataFrame(genes,columns=['genes'])
    genes['id'] = np.arange(len(genes))
    genes.set_index(["genes"], inplace=True)

    id_to_genes = pd.DataFrame(genes.index.tolist(),columns=['genes'])
    id_to_genes['id'] = np.arange(len(genes))
    id_to_genes.set_index(["id"], inplace=True)

    pos = prio_net[prio_net['value']==1]
    neg = prio_net[prio_net['value']==0]
    train_data = []
    train_data.append(genes.loc[pos['Tf'].values.tolist()]['id'].values.tolist())
    train_data.append(genes.loc[pos['Target'].values.tolist()]['id'].values.tolist())
    train_data.append(genes.loc[neg['Tf'].values.tolist()]['id'].values.tolist())
    train_data.append(genes.loc[neg['Target'].values.tolist()]['id'].values.tolist())

    test_data = []
    test_data.append(genes.loc[valid_net['Tf'].values.tolist()]['id'].values.tolist())
    test_data.append(genes.loc[valid_net['Target'].values.tolist()]['id'].values.tolist())
    test_data.append([])
    test_data.append([])

    return train_data, test_data, features, len(features),id_to_genes
