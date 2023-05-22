import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

names = ['hESC','hHEP','mDC','mESC','mHSC-E','mHSC-GM','mHSC-L']
sns.set(style='whitegrid')

def addlabels(y):
    for i in range(len(y)):
        plt.text(i,y[i],'%.2f'%y[i],ha='center')
def plot_bars( x, y, hue,name):
    sns.barplot(x=x, y=y, hue=hue)
    addlabels(y.values)
    plt.ylim(0,1)
    plt.title(name)
    plt.xlabel("Cell size")
    plt.ylabel("AUROC")
    plt.xticks(np.arange(5), ("100%", "80%", "60%", "40%", "20%"))
    plt.legend(loc='lower right')
    plt.show()

def plot_boxs(x,y,hue,name):
    sns.boxplot(x=x, y=y, hue=hue,palette=sns.color_palette('hls',2))
    plt.title(name)
    plt.xlabel("Train size")
    plt.ylabel("AUROC")
    plt.xticks(np.arange(5), ("20%", "40%", "60%", "80%", "100%"))
    plt.legend(loc='lower right')
    plt.show()

def all_plots(data,names):
    for name in names:
        data_tmp = data[data[2] == name]
        plot_boxs(data_tmp[5], data_tmp[6], data_tmp[3], name)

def all_bar(data,names):
    for name in names:
        data_tmp = data[data[2] == name]
        data_1 = data_tmp[data_tmp[3] == 500]
        data_2 = data_tmp[data_tmp[3] == 1000]
        data_1 = data_1.groupby(4).mean()
        data_2 = data_2.groupby(4).mean()
        plot_bars(data_1.index, data_1[6], data_1[3], name)
        plot_bars(data_2.index, data_2[6], data_2[3], name)

def plot_bars2( x, y, hue,name):
    sns.barplot(x=x, y=y, hue=hue)
    #plt.ylim(0,1)
    plt.title(name)
    plt.xlabel("")
    plt.ylabel("AUPRC")
    plt.xticks(np.arange(7), ('hESC','hHEP','mDC','mESC','mHSC-E','mHSC-GM','mHSC-L'))
    plt.legend(loc='lower right')
    plt.show()

def all_bar2(data,names):
    for name in names:
        tmp_data = data[data[2]==name]
        if name == 'Specific':
            name = 'Cell-type-specific'
        if name == 'Non-Specific':
            name = 'Non-specific'
        data_1 = tmp_data[tmp_data[4] == 500]
        data_2 = tmp_data[tmp_data[4] == 1000]
        data_1 = data_1.groupby([1,3],as_index=False).mean()
        data_2 = data_2.groupby([1,3],as_index=False).mean()

        tmp_1 = data_1.groupby([1]).mean()[8]
        print(name+"_500_increased:%.2f" % ((tmp_1[0]-tmp_1[1])/tmp_1[0]*100))
        tmp_2 = data_2.groupby([1]).mean()[8]
        print(name + "_1000_increased:%.2f" % ((tmp_2[0] - tmp_2[1]) / tmp_2[0] * 100))
        plot_bars2(data_1[3], data_1[8],data_1[1],name+"_500")
        plot_bars2(data_2[3], data_2[8], data_2[1],name+"_1000")
# train_size = pd.read_csv("./train_size.csv",sep=',',header=None)
# train_size = train_size[train_size[1]=='Specific']
# all_plots(train_size,names)

# cell_size = pd.read_csv("./cell_size.csv", sep=',', header=None)
# cell_size = cell_size[cell_size[1]=='Specific']
# all_bar(cell_size,names)

neigh = pd.read_csv("./neigh.csv",sep=',',header=None)
names = ['STRING','Specific','Non-Specific']
all_bar2(neigh,names)

