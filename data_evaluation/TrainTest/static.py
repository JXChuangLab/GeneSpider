import pandas as pd

netwoprks = ["STRING","Specific","Non-Specific"]
nums = ["500","1000"]
types = ["hESC","hHEP","mDC","mESC","mHSC-E","mHSC-GM","mHSC-L"]


def caculate(net,type,num):
    path = "./"+net+"/"+type+" "+num+"/"
    data_1 = pd.read_csv(path+"Train_set.csv")
    data_2 = pd.read_csv(path+"Validation_set.csv")
    data_3 = pd.read_csv(path+"Test_set.csv")
    data = pd.concat([data_1,data_2,data_3],axis=0)

    TFs = set(data['TF'].values.tolist())
    TFs = len(TFs)
    data = data[data['Label']==1]
    pairs = len(data)
    print("%s,%s,%s,%d,%d" % (net,type,num,TFs,pairs))
print("network,type,num,TFs,pairs")
for net in netwoprks:
    for type in types:
        for num in nums:
            caculate(net,type,num)