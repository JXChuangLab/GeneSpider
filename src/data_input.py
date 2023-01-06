import random
import numpy as np
import pandas as pd

from pyts.image import GramianAngularField


def drop_exp(row,rate):
    size = len(row)
    idxs = [i for i in range(size)]
    idxs = random.sample(idxs,int(size*rate))
    row[idxs] = 0
    return row

def load_data(net,type,nums,size=32,rate=0):

    exp_path = "./data_evaluation/Benchmark Dataset/"+net+" Dataset/"+type+"/TFs+"+str(nums)+"/BL--ExpressionData.csv"
    geneids_path = "./data_evaluation/Benchmark Dataset/"+net+" Dataset/" + type + "/TFs+" + str(nums) + "/Target.csv"

    train_set_path = "./data_evaluation/TrainTest/"+net+"/"+type+" "+str(nums)+"/"+"Train_set.csv"
    valid_set_path = "./data_evaluation/TrainTest/"+net+"/"+type+" "+str(nums)+"/"+"Validation_set.csv"
    test_set_path = "./data_evaluation/TrainTest/"+net+"/"+type+" "+str(nums)+"/"+"Test_set.csv"

    exp = pd.read_csv(exp_path,sep=',',index_col=0)
    exp = exp.apply(drop_exp,axis=1,rate=rate)

    train = pd.read_csv(train_set_path,sep=',',index_col=0)
    valid = pd.read_csv(valid_set_path,sep=',',index_col=0)
    train = pd.concat([train,valid],axis=0)
    test = pd.read_csv(test_set_path,sep=',',index_col=0)

    genes_ids = pd.read_csv(geneids_path,sep=',',index_col=0)
    features = genes_ids['Gene'].values.tolist()

    features = exp.loc[features]

    gas = GramianAngularField(image_size=size, method='d')
    features = gas.fit_transform(features)
    features = np.expand_dims(features, axis=1)

    src_pos_ids = train[train['Label']==1]['TF'].values
    dst_pos_ids = train[train['Label']==1]['Target'].values
    src_neg_ids = train[train['Label']==0]['TF'].values
    dst_neg_ids = train[train['Label']==0]['Target'].values

    indexs_pos = np.arange(len(src_pos_ids))
    indexs_pos = np.random.choice(indexs_pos,int(len(src_pos_ids)*0.7),replace=False).tolist()
    indexs_neg = np.arange(len(src_neg_ids))
    indexs_neg = np.random.choice(indexs_neg, int(len(src_neg_ids) * 0.7), replace=False).tolist()

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