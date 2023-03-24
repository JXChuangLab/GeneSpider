import pandas as pd
net = ['STRING','Specific','Non-Specific']
cells = ['hESC','hHEP','mDC','mESC','mHSC-E','mHSC-GM','mHSC-L']

for n in net:
    for c in cells:
        train_500 = pd.read_csv("./TrainTest/"+n+"/"+c+" 500/Train_set.csv")
        train_1000 = pd.read_csv("./TrainTest/"+n+"/"+c+" 500/Train_set.csv")
        pos_500 = train_500[train_500['Label']==1]
        neg_500 = train_500[train_500['Label']==0]
        pos_1000 = train_1000[train_1000['Label'] == 1]
        neg_1000 = train_1000[train_1000['Label'] == 0]
        print(n+"_"+c+"500 rate:{} : {}".format(len(pos_500),len(neg_500)))
        print(n + "_" + c + "1000 rate:{} : {}".format(len(pos_1000), len(neg_1000)))